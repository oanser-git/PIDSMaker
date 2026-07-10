"""Utilities for PIDSMaker runs backed by Orange export artifacts."""

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


ORANGE_EXPORT_ROOT = os.environ.get("ORANGE_EXPORT_ROOT", "/home/pids/capture_export/pidsmaker_export")

ORANGE_RAW_NODE_TYPES = [
    "address",
    "argv",
    "block",
    "char",
    "device",
    "directory",
    "file",
    "iattr",
    "inode_unknown",
    "link",
    "machine",
    "named pipe",
    "netflow",
    "network",
    "packet",
    "path",
    "pipe",
    "process",
    "process_memory",
    "regular file",
    "socket",
    "super block",
    "symlink",
    "task",
    "unknown",
    "xattr",
]

ORANGE_RAW_EDGE_LABELS = [
    "Used:accept",
    "Used:bind",
    "Used:bind_addr",
    "Used:chown",
    "Used:connect",
    "Used:connect_addr",
    "Used:exec",
    "Used:file_rcv",
    "Used:file_receive",
    "Used:getattr",
    "Used:getxattr",
    "Used:inode_readlink",
    "Used:listen",
    "Used:listxattr",
    "Used:memory_read",
    "Used:mmap",
    "Used:mmap_private",
    "Used:open",
    "Used:path_chroot",
    "Used:path_rename_read",
    "Used:perm",
    "Used:perm_check",
    "Used:ptrace_access",
    "Used:ptrace_read",
    "Used:read",
    "Used:read_ioctl",
    "Used:read_link",
    "Used:receive",
    "Used:receive_msg",
    "Used:sb_set_mnt_opts",
    "Used:sb_statfs",
    "Used:sb_umount",
    "Used:shutdown",
    "Used:socket_msg",
    "WasAssociatedWith:ran_on",
    "WasDerivedFrom:accept_socket",
    "WasDerivedFrom:addressed",
    "WasDerivedFrom:arg",
    "WasDerivedFrom:capset",
    "WasDerivedFrom:exec",
    "WasDerivedFrom:free",
    "WasDerivedFrom:getxattr_inode",
    "WasDerivedFrom:named",
    "WasDerivedFrom:receive_packet",
    "WasDerivedFrom:receive_unix",
    "WasDerivedFrom:send_unix",
    "WasDerivedFrom:setattr_inode",
    "WasDerivedFrom:setgid",
    "WasDerivedFrom:setuid",
    "WasDerivedFrom:setxattr_inode",
    "WasDerivedFrom:sh_read",
    "WasDerivedFrom:sh_write",
    "WasDerivedFrom:terminate_proc",
    "WasDerivedFrom:version",
    "WasDerivedFrom:version_entity",
    "WasGeneratedBy:bind",
    "WasGeneratedBy:clone",
    "WasGeneratedBy:clone_mem",
    "WasGeneratedBy:connect",
    "WasGeneratedBy:connect_unix_stream",
    "WasGeneratedBy:exec_task",
    "WasGeneratedBy:file_lock",
    "WasGeneratedBy:inode_create",
    "WasGeneratedBy:link",
    "WasGeneratedBy:listen",
    "WasGeneratedBy:memory_write",
    "WasGeneratedBy:munmap",
    "WasGeneratedBy:path_chmod",
    "WasGeneratedBy:path_rename_write",
    "WasGeneratedBy:rename",
    "WasGeneratedBy:sb_mount",
    "WasGeneratedBy:sb_pivotroot",
    "WasGeneratedBy:send",
    "WasGeneratedBy:send_msg",
    "WasGeneratedBy:setattr",
    "WasGeneratedBy:setpgid",
    "WasGeneratedBy:setuid",
    "WasGeneratedBy:setxattr",
    "WasGeneratedBy:socket_create",
    "WasGeneratedBy:socket_pair_create",
    "WasGeneratedBy:unlink",
    "WasGeneratedBy:write",
    "WasGeneratedBy:write_ioctl",
    "WasInformedBy:clone",
    "WasInformedBy:kill",
    "WasInformedBy:prctl",
    "WasInformedBy:ptrace_read_task",
    "WasInformedBy:terminate_task",
    "WasInformedBy:version_activity",
    "WasInvalidatedBy:path_unlink",
]

ORANGE_RAW_TOOLS = ("camflow", "conprov", "provbpf", "recap")


def is_orange_dataset(cfg_or_dataset: Any) -> bool:
    dataset = getattr(cfg_or_dataset, "dataset", cfg_or_dataset)
    return bool(getattr(dataset, "orange_export_dir", ""))


def orange_dataset_configs() -> Dict[str, Dict[str, Any]]:
    configs = {}
    for tool in ORANGE_RAW_TOOLS:
        dataset = "ORANGE_{}_RAW".format(tool.upper())
        database = "orange_{}_raw".format(tool)
        configs[dataset] = {
            "raw_dir": "",
            "database": database,
            "database_all_file": database,
            "num_node_types": len(ORANGE_RAW_NODE_TYPES),
            "num_edge_types": len(ORANGE_RAW_EDGE_LABELS),
            "start_date": "train",
            "end_date": "test",
            "train_dates": ["train"],
            "val_dates": ["val"],
            "test_dates": ["test"],
            "unused_dates": [],
            "ground_truth_relative_path": [],
            "attack_to_time_window": [],
            "orange_export_dir": os.path.join(ORANGE_EXPORT_ROOT, tool, "raw"),
        }
    return configs


@lru_cache(maxsize=None)
def load_export_index(export_dir: str) -> Optional[Dict[str, Any]]:
    path = Path(export_dir) / "export_index.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_export_index_or_raise(cfg_or_dataset: Any) -> Dict[str, Any]:
    dataset = getattr(cfg_or_dataset, "dataset", cfg_or_dataset)
    export_dir = str(getattr(dataset, "orange_export_dir", ""))
    if not export_dir:
        raise ValueError("Missing dataset.orange_export_dir for Orange dataset.")
    export_index = load_export_index(export_dir)
    if export_index is None:
        raise FileNotFoundError("Missing Orange export index: {}".format(Path(export_dir) / "export_index.json"))
    return export_index


def get_node_types(cfg_or_dataset: Any) -> List[str]:
    dataset = getattr(cfg_or_dataset, "dataset", cfg_or_dataset)
    export_dir = str(getattr(dataset, "orange_export_dir", ""))
    export_index = load_export_index(export_dir) if export_dir else None
    if export_index is not None and export_index.get("node_types"):
        return [str(value) for value in export_index["node_types"]]
    return list(ORANGE_RAW_NODE_TYPES)


def get_edge_labels(cfg_or_dataset: Any) -> List[str]:
    dataset = getattr(cfg_or_dataset, "dataset", cfg_or_dataset)
    export_dir = str(getattr(dataset, "orange_export_dir", ""))
    export_index = load_export_index(export_dir) if export_dir else None
    if export_index is not None and export_index.get("edge_labels"):
        return [str(value) for value in export_index["edge_labels"]]
    return list(ORANGE_RAW_EDGE_LABELS)


def apply_dataset_metadata(dataset: Any) -> None:
    if not is_orange_dataset(dataset):
        return
    dataset.num_node_types = len(get_node_types(dataset))
    dataset.num_edge_types = len(get_edge_labels(dataset))


def bidirectional_map(labels: List[str], start: int = 1) -> Dict[Any, Any]:
    mapping = {}
    for index, label in enumerate(labels, start=start):
        mapping[index] = label
        mapping[label] = index
    return mapping


def replace_with_symlink(source: str, destination: str) -> None:
    src = Path(source).resolve()
    dst = Path(destination)
    if not src.exists():
        raise FileNotFoundError("Missing Orange export artifact: {}".format(src))
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(src), str(dst))
