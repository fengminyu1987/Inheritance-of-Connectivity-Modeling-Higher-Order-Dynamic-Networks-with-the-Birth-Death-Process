"""Read incidence data and compute common topology and distribution statistics."""
import hashlib
import json
import csv
import math
from itertools import combinations
from pathlib import Path
import numpy as np
import networkx as nx
from ibad_functions import clusering


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_graph(nodes, hyperedges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for n in G:
        G.nodes[n]['E'] = []
    E = {int(e): list(members) for e, members in hyperedges.items()}
    for e, members in E.items():
        if len(members) != len(set(members)):
            raise ValueError('Duplicate node within a hyperedge')
        for n in members:
            if n not in G:
                raise ValueError('Hyperedge refers to a node outside the node list')
            G.nodes[n]['E'].append(e)
    return G, E


def topology(G, E, compute_apl=True):

    projection = nx.Graph()
    projection.add_nodes_from(G.nodes)
    for members in E.values():
        projection.add_edges_from(combinations(members, 2))
    sizes = [len(e) for e in E.values()]
    L = sum(sizes)
    N, M = len(G), len(E)
    if N:
        component = max(nx.connected_components(projection), key=len)
        plcc = len(component)/N
        apl = float(nx.average_shortest_path_length(projection.subgraph(component))) if compute_apl else None
    else:
        plcc, apl = 0.0, 0.0 if compute_apl else None
    return dict(N=N, E=M, L=L, density=L/(N*M) if N and M else 0.0,
                node_mean_degree=L/N if N else 0.0, edge_mean_size=L/M if M else 0.0,
                C=float(clusering(G, E)), PLCC=plcc, APL_LCC=apl,
                N_triadic=sum(s*(s-1)//2 for s in sizes),

                node_triples_with_multiplicity=sum(math.comb(s, 3) for s in sizes if s>=3),
                projected_edges=projection.number_of_edges())


def read_network(config, data_dir):
    meta = dict(config)
    sources = []
    if config['kind'] == 'matrix':
        path = Path(data_dir)/config['file']
        A = np.loadtxt(path, delimiter=',', ndmin=2, encoding='utf-8-sig')
        if not np.all(np.isfinite(A)) or np.any(A < 0):
            raise ValueError(f'{path}: Matrix contains nonfinite or negative values')
        meta['original_matrix_shape'] = list(A.shape)
        meta['first_row_positive_entries'] = int(np.count_nonzero(A[0] > config['threshold']))
        if config['drop_first_row']:
            A = A[1:]
        if config['nodes_axis'] not in ('rows', 'columns'):
            raise ValueError('nodes_axis must be rows or columns')
        B = (A > config['threshold'])
        if config['nodes_axis'] == 'columns':
            B = B.T
        original_nodes = list(range(B.shape[0]))
        hyperedges = {e: np.flatnonzero(B[:, e]).tolist() for e in range(B.shape[1])}
        if config['drop_isolated_nodes']:
            original_nodes = np.flatnonzero(B.any(axis=1)).tolist()
        node_map = {old: i for i, old in enumerate(original_nodes)}
        hyperedges = {e: [node_map[n] for n in members] for e, members in hyperedges.items()}
        nodes = list(range(len(original_nodes)))
        meta['binary_rule'] = f'weight > {config["threshold"]}; weights are not incidence multiplicities'
        meta['species_labels_available'] = False
        sources.append(dict(file=path.name, sha256=hash_file(path)))
    else:
        sizes_path, ids_path = Path(data_dir)/config['sizes'], Path(data_dir)/config['vertices']
        sizes = np.loadtxt(sizes_path, dtype=np.int64, ndmin=1)
        ids = np.loadtxt(ids_path, dtype=np.int64, ndmin=1)
        if np.any(sizes < 0) or int(sizes.sum()) != len(ids):
            raise ValueError('Email files disagree: sum(nverts) must equal the number of simplex entries')
        if not 0 < config['fraction'] <= 1:
            raise ValueError('fraction must be in (0,1]')
        limit = (int(len(sizes)*config['fraction']) if config['edge_limit'] is None else int(config['edge_limit']))
        if limit < 1 or limit > len(sizes):
            raise ValueError('Email record count is out of bounds')
        members_list, offset, removed_duplicates = [], 0, 0
        seen = set()
        for size in sizes[:limit]:
            members = sorted(set(int(n) for n in ids[offset:offset+int(size)]))
            removed_duplicates += int(size)-len(members)
            offset += int(size)
            if not config['keep_singletons'] and len(members) <= 1:
                continue
            key = tuple(members)
            if config['deduplicate_hyperedges'] and key in seen:
                continue
            seen.add(key)
            members_list.append(members)
        original_nodes = sorted({n for members in members_list for n in members})
        node_map = {old: i for i, old in enumerate(original_nodes)}
        nodes = list(range(len(original_nodes)))
        hyperedges = {e: [node_map[n] for n in members] for e, members in enumerate(members_list)}
        meta.update(total_records=len(sizes), selected_records=limit,
                    original_id_count=len(ids), used_id_count=offset,
                    repeated_ids_removed_within_edges=removed_duplicates,
                    retained_identical_hyperedge_records=len(members_list)-len(set(map(tuple,members_list))),
                    selection='file-order prefix by hyperedge records; no timestamps supplied')
        sources = [dict(file=p.name, sha256=hash_file(p)) for p in (sizes_path, ids_path)]
    G, E = make_graph(nodes, hyperedges)
    meta.update(N=len(G), E=len(E), sources=sources,
                original_node_ids=original_nodes, retained_empty_edges=sum(not e for e in E.values()))
    return G, E, meta


def histogram(G, E):
    return (np.bincount([len(G.nodes[n]['E']) for n in G], minlength=1),
            np.bincount([len(e) for e in E.values()], minlength=1))


def probability(counts):
    a = np.asarray(counts, dtype=float)
    if a.sum() == 0:
        return [0.0]*len(a)
    return (a/a.sum()).tolist()


def mean_lists(rows):
    a = np.zeros(max(map(len, rows)))
    for row in rows:
        a[:len(row)] += row
    return (a/len(rows)).tolist()


def distribution_distances(p, q, epsilon=1e-10):
    # Align the full integer support, including zero degree. KL uses real || simulation.

    length = max(len(p), len(q))
    p, q = np.pad(p, (0,length-len(p))), np.pad(q, (0,length-len(q)))
    if p.sum() <= 0 or q.sum() <= 0:
        return dict(KL_real_sim=None, KL_infinite=False, KL_smoothed=None, JS=None,
                    log_base='e', smoothing_epsilon=epsilon)
    p, q = p/p.sum(), q/q.sum()
    positive = p > 0
    infinite = bool(np.any((q == 0) & positive))
    kl = None if infinite else float(np.sum(p[positive]*np.log(p[positive]/q[positive])))
    middle = (p+q)/2
    js = .5*sum(float(np.sum(x[x>0]*np.log(x[x>0]/middle[x>0]))) for x in (p,q))
    if epsilon <= 0:
        raise ValueError('KL smoothing epsilon must be positive')
    ps, qs = (p+epsilon)/(1+epsilon*length), (q+epsilon)/(1+epsilon*length)
    return dict(KL_real_sim=kl, KL_infinite=infinite,
                KL_smoothed=float(np.sum(ps*np.log(ps/qs))), JS=js,
                log_base='e', smoothing_epsilon=epsilon)


def descriptive_fit(p, bounds):

    k = np.arange(len(p)); p = np.array(p, dtype=float)
    mask = (k>0) & (p>0) & (k>=bounds[0])
    if bounds[1] is not None:
        mask &= k<=bounds[1]
    if mask.sum()<3:
        return dict(gamma=None, c=None, k_min=None, k_max=None, points=int(mask.sum()), R2_log=None)
    x,y=np.log(k[mask]),np.log(p[mask])
    c,slope=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
    total=np.sum((y-y.mean())**2)
    return dict(gamma=float(-slope), c=float(c), k_min=int(k[mask].min()), k_max=int(k[mask].max()),
                points=len(x), R2_log=float(1-np.sum((y-c-slope*x)**2)/total) if total>0 else None)
