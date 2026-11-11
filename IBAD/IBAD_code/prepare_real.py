"""Convert empirical data to hypergraphs and compute their reference statistics."""
from datetime import datetime, timezone
from real_config import DATASETS, DATA_DIR, REAL_DIR, NETWORKS, COMPUTE_APL
from real_utils import read_network, histogram, probability, topology, save_json, write_csv


def prepare(networks=None):
    summary=[]
    for name in (NETWORKS if networks is None else networks):
        G,E,meta=read_network(DATASETS[name],DATA_DIR)
        counts_n,counts_e=histogram(G,E)
        metrics=topology(G,E,COMPUTE_APL)
        output=dict(schema_version='real_hypergraph_v1', dataset=name, preprocessing=meta,
                    nodes=list(G),hyperedges=E, node_counts=counts_n.tolist(),edge_counts=counts_e.tolist(),
                    P_node=probability(counts_n),P_edge=probability(counts_e),metrics=metrics,
                    prepared_utc=datetime.now(timezone.utc).isoformat(),
                    metric_notes=dict(C='projected clustering excluding triples contained in one hyperedge',
                        N_triadic='sum_e choose(size(e),2), pair incidences with hyperedge multiplicity',
                        APL_LCC='average shortest path restricted to largest node-projection component'))
        save_json(REAL_DIR/f'{name}.json',output)
        summary.append(dict(dataset=name,**metrics))
        print(name,meta.get('original_matrix_shape'),metrics)
    write_csv(REAL_DIR/'real_summary.csv',summary)
    print('Empirical references saved in: ',REAL_DIR)

if __name__=='__main__':
    prepare()
