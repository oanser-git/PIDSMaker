import csv
import os.path
from collections import defaultdict

import torch

from .orange_export import get_export_index_or_raise, is_orange_dataset
from pidsmaker.utils.utils import (
    datetime_to_ns_time_US,
    get_all_graphs_for_dates,
    init_database_connection,
    log,
)


_ORANGE_ATTACK_TO_NIDS_CACHE = {}
_ORANGE_ATTACK_TO_EDGES_CACHE = {}


def _orange_test_graph_paths(cfg):
    return get_all_graphs_for_dates(cfg.construction._graphs_dir, cfg.dataset.test_dates)


def _orange_malicious_windows(export_index):
    return {
        str(window["file_name"])
        for run in export_index.get("splits", {}).get("test", [])
        for window in run.get("windows", [])
        if int(window.get("y", 0)) == 1 or int(window.get("attack_edge_count", 0)) > 0
    }


def get_orange_tw_to_malicious_nodes(cfg):
    export_index = get_export_index_or_raise(cfg)
    malicious_window_names = _orange_malicious_windows(export_index)
    tw_to_malicious_nodes = {}
    num_malicious_edges = 0

    for tw, path in enumerate(_orange_test_graph_paths(cfg)):
        if os.path.basename(path) not in malicious_window_names:
            continue

        graph = torch.load(path)
        node_to_count = defaultdict(int)
        for src, dst, _, attrs in graph.edges(data=True, keys=True):
            if int(attrs.get("y", 0)) != 1:
                continue
            node_to_count[str(src)] += 1
            node_to_count[str(dst)] += 1
            num_malicious_edges += 1

        if node_to_count:
            tw_to_malicious_nodes[tw] = dict(node_to_count)

    log(
        "Loaded Orange labels for {} malicious time window(s) and {} malicious edge(s).".format(
            len(tw_to_malicious_nodes), num_malicious_edges
        )
    )
    return tw_to_malicious_nodes


def get_orange_GP_of_each_attack(cfg):
    cache_key = (cfg.dataset.orange_export_dir, cfg.construction._graphs_dir)
    if cache_key in _ORANGE_ATTACK_TO_NIDS_CACHE:
        return _ORANGE_ATTACK_TO_NIDS_CACHE[cache_key]

    export_index = get_export_index_or_raise(cfg)
    attack_id_by_window = {}
    attack_to_nids = {}

    for run in export_index.get("splits", {}).get("test", []):
        malicious_windows = [
            window
            for window in run.get("windows", [])
            if int(window.get("y", 0)) == 1 or int(window.get("attack_edge_count", 0)) > 0
        ]
        if not malicious_windows:
            continue

        attack_id = len(attack_to_nids)
        start_ns = min(int(window["start_ns"]) for window in malicious_windows)
        end_ns = max(int(window["end_ns"]) for window in malicious_windows)
        attack_to_nids[attack_id] = {"nids": set(), "time_range": [start_ns, end_ns]}
        for window in malicious_windows:
            attack_id_by_window[str(window["file_name"])] = attack_id

    for path in _orange_test_graph_paths(cfg):
        attack_id = attack_id_by_window.get(os.path.basename(path))
        if attack_id is None:
            continue

        graph = torch.load(path)
        for src, dst, _, attrs in graph.edges(data=True, keys=True):
            if int(attrs.get("y", 0)) != 1:
                continue
            attack_to_nids[attack_id]["nids"].add(int(src))
            attack_to_nids[attack_id]["nids"].add(int(dst))

    _ORANGE_ATTACK_TO_NIDS_CACHE[cache_key] = attack_to_nids
    return attack_to_nids


def get_orange_attack_to_mal_edges(cfg):
    cache_key = (cfg.dataset.orange_export_dir, cfg.construction._graphs_dir)
    if cache_key in _ORANGE_ATTACK_TO_EDGES_CACHE:
        return _ORANGE_ATTACK_TO_EDGES_CACHE[cache_key]

    export_index = get_export_index_or_raise(cfg)
    attack_id_by_window = {}
    attack_to_mal_edges = {}

    for run in export_index.get("splits", {}).get("test", []):
        malicious_windows = [
            window
            for window in run.get("windows", [])
            if int(window.get("y", 0)) == 1 or int(window.get("attack_edge_count", 0)) > 0
        ]
        if not malicious_windows:
            continue

        attack_id = len(attack_to_mal_edges)
        attack_to_mal_edges[attack_id] = set()
        for window in malicious_windows:
            attack_id_by_window[str(window["file_name"])] = attack_id

    for path in _orange_test_graph_paths(cfg):
        attack_id = attack_id_by_window.get(os.path.basename(path))
        if attack_id is None:
            continue

        graph = torch.load(path)
        for src, dst, _, attrs in graph.edges(data=True, keys=True):
            if int(attrs.get("y", 0)) != 1:
                continue
            attack_to_mal_edges[attack_id].add(
                (str(src), str(dst), int(attrs["time"]), str(attrs["label"]))
            )

    log(
        "Loaded Orange edge ground truth with {} malicious edge(s) across {} attack run(s).".format(
            sum(len(edges) for edges in attack_to_mal_edges.values()), len(attack_to_mal_edges)
        )
    )
    _ORANGE_ATTACK_TO_EDGES_CACHE[cache_key] = attack_to_mal_edges
    return attack_to_mal_edges


def get_orange_ground_truth_edges(cfg):
    malicious_edges = set()
    for edges_set in get_orange_attack_to_mal_edges(cfg).values():
        malicious_edges |= edges_set
    return malicious_edges


def get_orange_ground_truth(cfg):
    attack_to_nids = get_orange_GP_of_each_attack(cfg)
    ground_truth_nids = set()
    for attack in attack_to_nids.values():
        ground_truth_nids.update(attack["nids"])

    ground_truth_paths = {node_id: str(node_id) for node_id in ground_truth_nids}
    uuid_to_node_id = {str(node_id): str(node_id) for node_id in ground_truth_nids}

    log(
        "Loaded Orange ground truth with {} malicious node(s) across {} attack run(s).".format(
            len(ground_truth_nids), len(attack_to_nids)
        )
    )
    return ground_truth_nids, ground_truth_paths, uuid_to_node_id


def get_ground_truth(cfg):
    if is_orange_dataset(cfg):
        return get_orange_ground_truth(cfg)

    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    ground_truth_nids, ground_truth_paths = [], {}
    uuid_to_node_id = {}
    for file in cfg.dataset.ground_truth_relative_path:
        with open(os.path.join(cfg._ground_truth_dir, file), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(int(node_id))
                ground_truth_paths[int(node_id)] = node_labels
                uuid_to_node_id[node_uuid] = str(node_id)

    mimicry_edge_num = cfg.construction.mimicry_edge_num
    if mimicry_edge_num is not None and mimicry_edge_num > 0:
        num_GPs = len(ground_truth_nids)
        for file in cfg.dataset.ground_truth_relative_path:
            file_name = file.split("/")[-1]
            with open(os.path.join(cfg.construction._mimicry_dir, file_name), "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    node_uuid, node_labels, _ = row[0], row[1], row[2]
                    node_id = uuid2nids[node_uuid]
                    ground_truth_nids.append(int(node_id))
                    ground_truth_paths[int(node_id)] = node_labels
                    uuid_to_node_id[node_uuid] = str(node_id)
        num_mimicry_GPs = len(ground_truth_nids) - num_GPs
        log(f"{num_mimicry_GPs} mimicry ground truth nodes loaded")

    return set(ground_truth_nids), ground_truth_paths, uuid_to_node_id


def get_GP_of_each_attack(cfg):
    if is_orange_dataset(cfg):
        return get_orange_GP_of_each_attack(cfg)

    cur, connect = init_database_connection(cfg)
    uuid2nids, _ = get_uuid2nids(cur)

    attack_to_nids = {}

    for i, (path, attack_to_time_window) in enumerate(
        zip(cfg.dataset.ground_truth_relative_path, cfg.dataset.attack_to_time_window)
    ):
        attack_to_nids[i] = {}
        attack_to_nids[i]["nids"] = set()
        attack_to_nids[i]["time_range"] = [
            datetime_to_ns_time_US(tw)
            for tw in [attack_to_time_window[1], attack_to_time_window[2]]
        ]

        with open(os.path.join(cfg._ground_truth_dir, path), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                attack_to_nids[i]["nids"].add(int(node_id))

        mimicry_edge_num = cfg.construction.mimicry_edge_num
        if mimicry_edge_num is not None and mimicry_edge_num > 0:
            num_mimicry_GPs = 0
            with open(os.path.join(cfg.construction._mimicry_dir, path.split("/")[-1]), "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    num_mimicry_GPs += 1
                    node_uuid, node_labels, _ = row[0], row[1], row[2]
                    node_id = uuid2nids[node_uuid]
                    attack_to_nids[i]["nids"].add(int(node_id))
            log(f"{num_mimicry_GPs} mimicry ground truth nodes loaded")
    return attack_to_nids


def get_uuid2nids(cur):
    queries = {
        "file": "SELECT index_id, node_uuid FROM file_node_table;",
        "netflow": "SELECT index_id, node_uuid FROM netflow_node_table;",
        "subject": "SELECT index_id, node_uuid FROM subject_node_table;",
    }
    uuid2nids = {}
    nid2uuid = {}
    for node_type, query in queries.items():
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            uuid2nids[row[1]] = row[0]
            nid2uuid[row[0]] = row[1]

    return uuid2nids, nid2uuid


def get_events(
    cur,
    start_time,
    end_time,
):
    # malicious_nodes_str = ', '.join(f"'{node}'" for node in malicious_nodes)
    # sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}' AND src_index_id IN ({malicious_nodes_str});"
    sql = f"SELECT * FROM event_table WHERE timestamp_rec BETWEEN '{start_time}' AND '{end_time}';"

    cur.execute(sql)
    rows = cur.fetchall()
    return rows


def get_t2malicious_node(cfg) -> dict[list]:
    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    t_to_node = defaultdict(list)

    for attack_tuple in cfg.dataset.attack_to_time_window:
        attack = attack_tuple[0]
        start_time = datetime_to_ns_time_US(attack_tuple[1])
        end_time = datetime_to_ns_time_US(attack_tuple[2])

        ground_truth_nids = set()
        with open(os.path.join(cfg._ground_truth_dir, attack), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.add(str(node_id))

        mimicry_edge_num = cfg.construction.mimicry_edge_num
        if mimicry_edge_num is not None and mimicry_edge_num > 0:
            num_GPs = len(ground_truth_nids)
            with open(
                os.path.join(cfg.construction._mimicry_dir, attack.split("/")[-1]),
                "r",
            ) as f:
                reader = csv.reader(f)
                for row in reader:
                    node_uuid, node_labels, _ = row[0], row[1], row[2]
                    node_id = uuid2nids[node_uuid]
                    ground_truth_nids.add(str(node_id))
            num_mimicry_GPs = len(ground_truth_nids) - num_GPs
            log(f"{num_mimicry_GPs} mimicry nodes loaded")

        rows = get_events(cur, start_time, end_time)
        for row in rows:
            src_id = row[1]
            dst_id = row[4]
            t = row[6]
            if src_id in ground_truth_nids:
                t_to_node[int(t)].append(nid2uuid[int(src_id)])
            if dst_id in ground_truth_nids:
                t_to_node[int(t)].append(nid2uuid[int(dst_id)])

    return t_to_node


def get_attack_to_mal_edges(cfg) -> dict[list]:
    if is_orange_dataset(cfg):
        return get_orange_attack_to_mal_edges(cfg)

    cur, connect = init_database_connection(cfg)
    uuid2nids, nid2uuid = get_uuid2nids(cur)

    malicious_edge_selection = cfg.evaluation.edge_evaluation.malicious_edge_selection

    attack_to_mal_edges = defaultdict(set)
    for i, (path, attack_to_time_window) in enumerate(
        zip(cfg.dataset.ground_truth_relative_path, cfg.dataset.attack_to_time_window)
    ):
        start_time = datetime_to_ns_time_US(attack_to_time_window[1])
        end_time = datetime_to_ns_time_US(attack_to_time_window[2])

        ground_truth_nids = []
        with open(os.path.join(cfg._ground_truth_dir, path), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(str(node_id))
        ground_truth_nids = set(ground_truth_nids)

        rows = get_events(cur, start_time, end_time)
        for row in rows:
            src_idx_id = row[1]
            ope = row[2]
            dst_idx_id = row[4]
            event_uuid = row[5]
            timestamp_rec = row[6]

            condition = None
            if malicious_edge_selection == "src_node":
                condition = src_idx_id in ground_truth_nids
            elif malicious_edge_selection == "dst_node":
                condition = dst_idx_id in ground_truth_nids
            elif malicious_edge_selection == "both_nodes":
                condition = src_idx_id in ground_truth_nids and dst_idx_id in ground_truth_nids
            elif malicious_edge_selection == "either_node":
                condition = src_idx_id in ground_truth_nids or dst_idx_id in ground_truth_nids
            else:
                raise ValueError(
                    "`malicious_edge_selection` must be one of 'src_node', 'dst_node', 'both_nodes', 'either_node"
                )

            if condition:
                attack_to_mal_edges[i].add((src_idx_id, dst_idx_id, timestamp_rec, ope))

    return attack_to_mal_edges


def get_ground_truth_edges(cfg) -> set:
    if is_orange_dataset(cfg):
        return get_orange_ground_truth_edges(cfg)

    attack_to_mal_edges = get_attack_to_mal_edges(cfg)

    malicious_edges = set()
    for attack, edges_set in attack_to_mal_edges.items():
        malicious_edges |= edges_set

    return malicious_edges
