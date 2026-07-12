"""Training loop for PIDS models.

Handles model training with:
- Self-supervised pretraining with multiple objectives
- Optional few-shot fine-tuning for attack detection
- Gradient accumulation for large graphs
- Early stopping with patience
- Memory tracking (GPU and CPU)
- Validation-based model selection
"""

import copy
import os
import random
import tracemalloc
from time import perf_counter as timer

import numpy as np
import torch
import wandb

from pidsmaker.factory import (
    build_model,
    optimizer_factory,
    optimizer_few_shot_factory,
)
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.utils import get_device, log, log_start, log_tqdm, set_seed

from . import inference_loop


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state):
    if not isinstance(rng_state, dict):
        raise RuntimeError("Full checkpoint is missing RNG state")
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"])
    if rng_state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def load_checkpoint_model_state(model, checkpoint_state):
    current_state = model.state_dict()
    adjusted_keys = []
    model_state = {}
    for key, value in checkpoint_state.items():
        current_value = current_state.get(key)
        try:
            current_shape = tuple(current_value.shape) if current_value is not None else None
        except RuntimeError:
            current_shape = None
        try:
            current_numel = current_value.numel() if current_value is not None else None
        except (RuntimeError, ValueError):
            current_numel = None
        if (
            current_value is not None
            and value.numel() == 0
            and key.endswith("lin_edge.weight")
            and (current_shape is None or current_numel == 0)
        ):
            adjusted_keys.append(key)
            continue
        if (
            current_value is not None
            and current_shape is not None
            and tuple(value.shape) != current_shape
            and value.numel() == 0
            and current_numel == 0
        ):
            adjusted_keys.append(key)
            continue
        else:
            model_state[key] = value
    incompatible = model.load_state_dict(model_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = [key for key in incompatible.missing_keys if key not in set(adjusted_keys)]
    if unexpected or missing:
        raise RuntimeError(
            "Error(s) in loading checkpoint state: missing={} unexpected={}".format(
                missing, unexpected
            )
        )
    return adjusted_keys


def normalize_zero_sized_optimizer_state(optimizer):
    adjusted = []
    for group_index, group in enumerate(optimizer.param_groups):
        for param_index, param in enumerate(group["params"]):
            state = optimizer.state.get(param)
            if not isinstance(state, dict) or param.numel() != 0:
                continue
            for key, value in list(state.items()):
                if not torch.is_tensor(value) or value.numel() != 0 or tuple(value.shape) == tuple(param.shape):
                    continue
                state[key] = torch.zeros_like(param)
                adjusted.append("group{}.param{}.{}".format(group_index, param_index, key))
    return adjusted


def checkpoint_training_state(
    epoch_times,
    peak_train_cpu_mem,
    peak_train_gpu_mem,
    test_stats,
    patience_counter,
    all_test_stats,
    global_best_val_score,
    best_val_score,
    best_epoch,
    best_model,
):
    return {
        "epoch_times": list(epoch_times),
        "peak_train_cpu_mem": peak_train_cpu_mem,
        "peak_train_gpu_mem": peak_train_gpu_mem,
        "test_stats": test_stats,
        "patience_counter": patience_counter,
        "all_test_stats": list(all_test_stats),
        "global_best_val_score": global_best_val_score,
        "best_val_score": best_val_score,
        "best_epoch": best_epoch,
        "best_model_state_dict": best_model,
    }


def apply_checkpoint_training_state(state):
    if not isinstance(state, dict):
        raise RuntimeError("Full checkpoint is missing training state")
    return {
        "epoch_times": list(state.get("epoch_times") or []),
        "peak_train_cpu_mem": float(state.get("peak_train_cpu_mem") or 0),
        "peak_train_gpu_mem": float(state.get("peak_train_gpu_mem") or 0),
        "test_stats": state.get("test_stats"),
        "patience_counter": int(state.get("patience_counter") or 0),
        "all_test_stats": list(state.get("all_test_stats") or []),
        "global_best_val_score": float(state.get("global_best_val_score", float("-inf"))),
        "best_val_score": float(state.get("best_val_score", float("-inf"))),
        "best_epoch": state.get("best_epoch"),
        "best_model": state.get("best_model_state_dict"),
    }


def main(cfg):
    """Main training loop executing self-supervised pretraining and optional few-shot fine-tuning.

    Training process:
    1. Self-supervised pretraining on reconstruction/prediction objectives
    2. Optional few-shot fine-tuning on labeled attack data
    3. Validation-based model selection (best epoch or each epoch)
    4. Early stopping with configurable patience

    Args:
        cfg: Configuration with training hyperparameters (epochs, lr, patience, etc.)

    Returns:
        float: Best validation score achieved during training
    """
    set_seed(cfg, seed=cfg.training.seed)

    log_start(__file__)
    device = get_device(cfg)
    use_cuda = device == torch.device("cuda")

    # Reset the peak memory usage counter
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)
    tracemalloc.start()

    train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(cfg)

    model = build_model(
        data_sample=train_data[0][0], device=device, cfg=cfg, max_node_num=max_node_num
    )
    optimizer = optimizer_factory(cfg, parameters=list(model.parameters()))

    resume_checkpoint = os.environ.get("PIDSMAKER_RESUME_CHECKPOINT")
    save_checkpoint = os.environ.get("PIDSMAKER_SAVE_CHECKPOINT")
    resume_full_state = env_flag("PIDSMAKER_RESUME_FULL_STATE")
    save_full_state = env_flag("PIDSMAKER_CHECKPOINT_FULL_STATE")
    resume_optimizer = env_flag("PIDSMAKER_RESUME_OPTIMIZER") or resume_full_state
    resume_training_state = None
    start_epoch = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        if resume_full_state and checkpoint.get("checkpoint_mode") != "full_state":
            raise RuntimeError(
                "PIDSMAKER_RESUME_FULL_STATE=1 requires a full_state checkpoint: "
                f"{resume_checkpoint}"
            )
        adjusted_model_keys = load_checkpoint_model_state(model, checkpoint["model_state_dict"])
        if resume_optimizer:
            if checkpoint.get("optimizer_state_dict") is None:
                raise RuntimeError(f"Checkpoint has no optimizer state: {resume_checkpoint}")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            adjusted_optimizer_keys = normalize_zero_sized_optimizer_state(optimizer)
        else:
            adjusted_optimizer_keys = []
        if resume_full_state:
            restore_rng_state(checkpoint.get("rng_state"))
            resume_training_state = apply_checkpoint_training_state(checkpoint.get("training_state"))
        if adjusted_model_keys:
            log(
                "Adjusted zero-sized checkpoint tensors: " + ", ".join(adjusted_model_keys),
                return_line=True,
            )
        if adjusted_optimizer_keys:
            log(
                "Adjusted zero-sized optimizer checkpoint tensors: " + ", ".join(adjusted_optimizer_keys),
                return_line=True,
            )
        start_epoch = int(checkpoint.get("last_epoch", -1)) + 1
        log(
            f"Loaded training checkpoint from {resume_checkpoint}; continuing at epoch {start_epoch}",
            return_line=True,
        )

    run_evaluation = cfg.training_loop.run_evaluation
    assert run_evaluation in ["best_epoch", "each_epoch"], (
        f"Invalid run evaluation {run_evaluation}"
    )
    best_epoch_mode = run_evaluation == "best_epoch"

    num_epochs = cfg.training.num_epochs
    tot_loss = 0.0
    epoch_times = []
    peak_train_cpu_mem = 0
    peak_train_gpu_mem = 0
    test_stats = None
    patience = cfg.training.patience
    patience_counter = 0
    all_test_stats = []
    global_best_val_score = float("-inf")
    best_val_score, best_model, best_epoch = float("-inf"), None, None
    use_few_shot = cfg.training.decoder.use_few_shot
    grad_acc = cfg.training.grad_accumulation

    if resume_training_state:
        epoch_times = resume_training_state["epoch_times"]
        peak_train_cpu_mem = resume_training_state["peak_train_cpu_mem"]
        peak_train_gpu_mem = resume_training_state["peak_train_gpu_mem"]
        test_stats = resume_training_state["test_stats"]
        patience_counter = resume_training_state["patience_counter"]
        all_test_stats = resume_training_state["all_test_stats"]
        global_best_val_score = resume_training_state["global_best_val_score"]
        best_val_score = resume_training_state["best_val_score"]
        best_epoch = resume_training_state["best_epoch"]
        best_model = resume_training_state["best_model"]

    if use_few_shot:
        num_epochs += 1  # in few-shot, the first epoch is without ssl training

    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, num_epochs):
        if not use_few_shot or (use_few_shot and epoch > 0):
            start = timer()
            tracemalloc.start()

            # Before each epoch, we reset the memory
            model.reset_state()
            model.to_fine_tuning(False)

            loss_acc = torch.zeros(1, device=device)
            tot_loss = 0
            for dataset in train_data:
                for i, g in enumerate(log_tqdm(dataset, "Training")):
                    g.to(device=device)
                    g = remove_attacks_if_needed(g, cfg)
                    model.train()
                    optimizer.zero_grad()

                    results = model(g)
                    loss = results["loss"]
                    loss_acc += loss
                    tot_loss += loss.item()

                    if (i + 1) % grad_acc == 0:
                        loss_acc.backward()
                        optimizer.step()
                        loss_acc = torch.zeros(1, device=device)

                    g.to("cpu")
                    if use_cuda:
                        torch.cuda.empty_cache()

                # Last batch
                if loss_acc > 0:
                    loss_acc.backward()
                    optimizer.step()

            tot_loss /= sum(len(dataset) for dataset in train_data)
            epoch_times.append(timer() - start)

            _, peak_inference_cpu_memory = tracemalloc.get_traced_memory()
            peak_train_cpu_mem = max(peak_train_cpu_mem, peak_inference_cpu_memory / (1024**3))
            tracemalloc.stop()

            if use_cuda:
                peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (
                    1024**3
                )
                peak_train_gpu_mem = max(peak_train_gpu_mem, peak_inference_gpu_memory)
                torch.cuda.reset_peak_memory_stats(device=device)

            log(
                f"[@epoch{epoch:02d}] Training finished - GPU memory: {peak_train_gpu_mem:.2f} GB | CPU memory: {peak_train_cpu_mem:.2f} GB | Mean Loss: {tot_loss:.4f}",
                return_line=True,
            )

        # Few-shot learning fine tuning
        if use_few_shot:
            model.to_fine_tuning(True)
            optimizer = optimizer_few_shot_factory(cfg, parameters=list(model.parameters()))

            num_epochs_few_shot = cfg.training.decoder.few_shot.num_epochs_few_shot
            patience_few_shot = cfg.training.decoder.few_shot.patience_few_shot

            for tuning_epoch in range(0, num_epochs_few_shot):
                model.reset_state()

                loss_acc = torch.zeros(1, device=device)
                tot_loss = 0
                for dataset in train_data:
                    for g in log_tqdm(dataset, "Fine-tuning"):
                        if 1 in g.y:
                            g.to(device=device)
                            model.train()
                            optimizer.zero_grad()

                            results = model(g)
                            loss = results["loss"]
                            loss_acc += loss
                            tot_loss += loss.item()

                            if (i + 1) % grad_acc == 0:
                                loss_acc.backward()
                                optimizer.step()
                                loss_acc = torch.zeros(1, device=device)

                            g.to("cpu")
                            if use_cuda:
                                torch.cuda.empty_cache()

                    # Last batch
                    if loss_acc > 0:
                        loss_acc.backward()
                        optimizer.step()

                tot_loss /= sum(len(dataset) for dataset in train_data)

                # Validation
                val_stats = inference_loop.main(
                    cfg=cfg,
                    model=model,
                    val_data=val_data,
                    test_data=test_data,
                    epoch=epoch,
                    split="val",
                    logging=False,
                )
                val_loss = val_stats["val_loss"]
                val_score = val_stats["val_score"]

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                    patience_counter = 0
                else:
                    patience_counter += 1

                if val_score > global_best_val_score:
                    global_best_val_score = val_score
                    best_epoch = epoch

                log(
                    f"[@epoch{tuning_epoch:02d}] Fine-tuning - Train Loss: {tot_loss:.5f} | Val Loss: {val_loss:.4f}",
                    return_line=True,
                )

                if patience_counter >= patience_few_shot:
                    log(f"Early stopping: best few-shot loss is {best_val_score:.4f}")
                    break

            model.load_state_dict(best_model)
            model.to_device(device)

        # model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
        # save_model(model, model_path, cfg)

        # Test
        if (epoch + 1) % 2 == 0 or epoch == 0:
            test_stats = inference_loop.main(
                cfg=cfg,
                model=model,
                val_data=val_data,
                test_data=test_data,
                epoch=epoch,
                split="all",
            )
            all_test_stats.append(test_stats)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_epoch": epoch,
                    "train_loss": round(tot_loss, 4),
                    "val_score": round(test_stats["val_score"], 4),
                    "val_loss": round(test_stats["val_loss"], 4),
                    "test_loss": round(test_stats["test_loss"], 4),
                }
            )
        last_completed_epoch = epoch

    if save_checkpoint:
        os.makedirs(os.path.dirname(save_checkpoint), exist_ok=True)
        checkpoint = {
            "checkpoint_schema_version": 2,
            "checkpoint_mode": "full_state" if save_full_state else "model_optimizer",
            "last_epoch": last_completed_epoch,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "training_num_epochs": num_epochs,
        }
        if save_full_state:
            checkpoint["rng_state"] = capture_rng_state()
            checkpoint["training_state"] = checkpoint_training_state(
                epoch_times,
                peak_train_cpu_mem,
                peak_train_gpu_mem,
                test_stats,
                patience_counter,
                all_test_stats,
                global_best_val_score,
                best_val_score,
                best_epoch,
                best_model,
            )
        torch.save(checkpoint, save_checkpoint)
        log(f"Saved training checkpoint to {save_checkpoint}", return_line=True)

    # After training
    if best_epoch_mode:
        model.load_state_dict(best_model)
        test_stats = inference_loop.main(
            cfg=cfg,
            model=model,
            val_data=val_data,
            test_data=test_data,
            epoch=best_epoch,
            split="test",
        )

    wandb.log(
        {
            "best_epoch": best_epoch,
            "train_epoch_time": round(np.mean(epoch_times), 2),
            "val_score": round(best_val_score, 5),
            "peak_train_cpu_memory": round(peak_train_cpu_mem, 3),
            "peak_train_gpu_memory": round(peak_train_gpu_mem, 3),
            "peak_inference_cpu_memory": round(
                np.max([d["peak_inference_cpu_memory"] for d in all_test_stats]), 3
            ),
            "peak_inference_gpu_memory": round(
                np.max([d["peak_inference_gpu_memory"] for d in all_test_stats]), 3
            ),
            "time_per_batch_inference": round(
                np.mean([d["time_per_batch_inference"] for d in all_test_stats]), 3
            ),
        }
    )

    return best_val_score


def remove_attacks_if_needed(graph, cfg):
    """Remove attack edges from graph for self-supervised training if configured.

    Args:
        graph: Graph batch with labels in graph.y
        cfg: Configuration with few_shot.include_attacks_in_ssl_training setting

    Returns:
        graph: Original graph or filtered graph without attacks (y=1)
    """
    if not cfg.training.decoder.few_shot.include_attacks_in_ssl_training:
        if 1 in graph.y:
            return graph.clone()[graph.y != 1]
    return graph
