import os

from pidsmaker.config import update_cfg_for_multi_dataset
from pidsmaker.preprocessing.build_graph_methods import (
    build_default_graphs,
    build_magic_graphs,
)
from pidsmaker.utils.orange_export import is_orange_dataset, replace_with_symlink
from pidsmaker.utils.utils import get_multi_datasets, log


def import_orange_construction(cfg):
    export_dir = getattr(cfg.dataset, "orange_export_dir", "")
    if not export_dir:
        raise ValueError("Missing dataset.orange_export_dir for Orange construction import.")

    source_construction_dir = os.path.join(export_dir, "construction")
    if not os.path.isdir(source_construction_dir):
        raise FileNotFoundError(
            "Missing Orange construction directory: {}".format(source_construction_dir)
        )

    os.makedirs(cfg.construction._task_path, exist_ok=True)
    os.makedirs(cfg.construction._tw_labels, exist_ok=True)

    for source_name, destination in [
        ("nx", cfg.construction._graphs_dir),
        ("indexid2msg", cfg.construction._dicts_dir),
        ("node_id_to_path", cfg.construction._node_id_to_path),
    ]:
        source = os.path.join(source_construction_dir, source_name)
        replace_with_symlink(source, destination)

    log("Linked prebuilt Orange construction artifacts from '{}'".format(source_construction_dir))


def main_from_config(cfg):
    if is_orange_dataset(cfg):
        import_orange_construction(cfg)
        return

    graph_method = cfg.construction.used_method
    if graph_method == "default":
        build_default_graphs.main(cfg)
    elif graph_method == "magic":
        build_default_graphs.main(cfg)
        build_magic_graphs.main(cfg)
    else:
        raise ValueError(f"Unrecognized graph method: {graph_method}")


def main(cfg):
    multi_datasets = get_multi_datasets(cfg)
    if "none" in multi_datasets:
        main_from_config(cfg)

    # Multi-dataset mode
    else:
        for dataset in multi_datasets:
            updated_cfg, should_restart = update_cfg_for_multi_dataset(cfg, dataset)

            if should_restart["construction"]:
                main_from_config(updated_cfg)
