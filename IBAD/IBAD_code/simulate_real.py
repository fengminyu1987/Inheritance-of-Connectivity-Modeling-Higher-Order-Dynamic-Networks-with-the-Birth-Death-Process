"""Simulate from a small hypergraph and record post-burn-in observations."""
import json
import hashlib
from pathlib import Path
from array import array
from datetime import datetime, timezone
from time import perf_counter
import numpy as np
import networkx as nx
import ibad_functions as model
import real_config as cfg
from real_utils import make_graph, histogram, topology, save_json, probability


def real_identity(real):
    payload={k:real[k] for k in ('dataset','preprocessing','nodes','hyperedges')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def parameters_for(real, use_selected=None):
    p=dict(cfg.PARAMETERS[real['dataset']])
    if p.pop('match_population'):
        if not real['metrics']['N'] or not real['metrics']['E']:
            raise ValueError('Cannot match an empty population')
        p['mu_n']=p['lambda_n']/real['metrics']['N']
        p['mu_e']=p['lambda_e']/real['metrics']['E']
    p.update(a_n=cfg.A_N,a_e=cfg.A_E,min_history=cfg.MIN_HISTORY,
             deletion_mode=cfg.DELETION_MODE,similarity_rule=cfg.SIMILARITY_RULE,
             similarity_sigma=cfg.SIMILARITY_SIGMA,T_pre=cfg.T_PRE,T=cfg.T,
             sample_every=cfg.SAMPLE_EVERY,trace_every=cfg.TRACE_EVERY,
             compute_apl=cfg.COMPUTE_APL,initial_network=cfg.INITIAL_NETWORK,
             initial_nodes=cfg.INITIAL_SIZES[real['dataset']][0],
             initial_edges=cfg.INITIAL_SIZES[real['dataset']][1])
    if use_selected is None:
        use_selected = cfg.USE_SEARCH_PARAMETERS
    if use_selected:
        path = cfg.SELECTED_DIR / (real['dataset'] + '.json')
        selected = json.loads(path.read_text(encoding='utf-8'))
        if selected['real_signature'] != real_identity(real):
            raise ValueError('Selected parameters refer to different empirical preprocessing')
        p = dict(selected['parameters'])
    return p


def initialize(real,p,seed):
    for key in ('lambda_n','lambda_e','mu_n','mu_e','a_n','a_e','similarity_sigma'):
        if not np.isfinite(p[key]) or p[key]<=0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('m_n','m_e','T_pre'):
        if not isinstance(p[key],int) or isinstance(p[key],bool) or p[key]<0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('T','sample_every','trace_every','min_history'):
        if not isinstance(p[key],int) or isinstance(p[key],bool) or p[key]<1:
            raise ValueError(f'{key} must be a positive integer')
    if p['min_history']<3:
        raise ValueError('min_history must be at least 3')
    if p['similarity_rule'] not in ('history_rbf_raw','history_rbf','positive','shifted'):
        raise ValueError('Unknown similarity_rule')
    if p['initial_network'] != 'small':
        raise ValueError('initial_network must be small')
    for key in ('initial_nodes', 'initial_edges'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    nodes = list(range(p['initial_nodes']))
    model.G,model.E=make_graph(nodes,{e:list(nodes) for e in range(p['initial_edges'])})
    for key,value in p.items():
        setattr(model,key,value)
    model.l_N=max(model.G.nodes,default=-1)
    model.l_E=max(model.E,default=-1)
    model.random.seed(seed)
    np.random.seed(seed)
    model.time_list={'birth_n':['birth_n',np.random.exponential(1/p['lambda_n'])],
                     'birth_e':['birth_e',np.random.exponential(1/p['lambda_e'])]}
    for n in model.G:
        model.time_list[f'{n}n']=['n',np.random.exponential(1/p['mu_n'])]
    for e in model.E:
        model.time_list[f'{e}e']=['e',np.random.exponential(1/p['mu_e'])]
    model.his_n={n:array('d',[len(model.G.nodes[n]['E'])]) for n in model.G}
    model.his_e={e:array('d',[len(model.E[e])]) for e in model.E}
    model.inheritance_stats={kind+'_'+item:0 for kind in ('node','edge')
        for item in ('lost','success','duplicate','no_candidate','direct',
                     'undefined_pairs','nonpositive_pairs','positive_pairs','history_pairs')}
    deletion_functions={
        'direct':(model.del_node_1,model.del_E),
        'uniform_single':(model.del_N_random,model.del_E_random),
        'uniform_positive':(model.del_N_random_positive,model.del_E_random_positive),
        'ibad_single':(model.del_N_H,model.del_E_H),
        'ibad_feasible':(model.del_N_feasible,model.del_E_feasible)}
    if p['deletion_mode'] not in deletion_functions:
        raise ValueError('Unknown deletion_mode')
    return deletion_functions[p['deletion_mode']]


def next_event(p,del_node,del_edge):
    do_=min(model.time_list,key=lambda x:model.time_list[x][1])
    dt=model.time_list[do_][1]
    if dt<0:
        raise RuntimeError('Negative residual event time')
    for key in model.time_list:
        model.time_list[key][1]-=dt
    event=model.time_list[do_][0]
    if event=='birth_n':
        model.get_node(p['m_n'])
        model.time_list['birth_n']=['birth_n',np.random.exponential(1/p['lambda_n'])]
    elif event=='birth_e':
        model.E=model.get_E(p['m_e'])
        model.time_list['birth_e']=['birth_e',np.random.exponential(1/p['lambda_e'])]
    elif event=='n':
        del_node(int(do_[:-1]))
    else:
        model.E=del_edge(int(do_[:-1]))
    if p['deletion_mode']!='direct':
        for n in model.G:
            if p['deletion_mode']!='uniform_single' or len(model.his_n[n])<p['min_history']:
                model.his_n[n].append(len(model.G.nodes[n]['E']))
        for e in model.E:
            if p['deletion_mode']!='uniform_single' or len(model.his_e[e])<p['min_history']:
                model.his_e[e].append(len(model.E[e]))
    return dt


def add_histogram(total,values,weight=1.0):
    if len(values)>len(total):
        total=np.pad(total,(0,len(values)-len(total)))
    total[:len(values)]+=weight*values
    return total


def run_one(real,p,seed,save=True):
    start=perf_counter()
    del_node,del_edge=initialize(real,p,seed)
    initial_nodes,initial_edges=set(model.G),set(model.E)
    warm_trace=[]
    time_now=0.0
    def trace(event):
        warm_trace.append(dict(event=event,time=time_now,N=len(model.G),E=len(model.E),
                               L=sum(map(len,model.E.values()))))
    trace(0)
    for t in range(p['T_pre']):
        time_now+=next_event(p,del_node,del_edge)
        if (t+1)%p['trace_every']==0 or t==p['T_pre']-1:
            trace(t+1)
            print(real['dataset'],seed,'warmup',t+1,'/',p['T_pre'],flush=True)
    warm_stats=dict(model.inheritance_stats)
    model.inheritance_stats=dict.fromkeys(model.inheritance_stats,0)
    time_start=time_now
    residual=dict(initial_nodes_remaining=len(initial_nodes & set(model.G)),
                  initial_edges_remaining=len(initial_edges & set(model.E)))
    event_counts_n,event_counts_e=np.zeros(1),np.zeros(1)
    time_p_n,time_p_e=np.zeros(1),np.zeros(1)
    nonempty_n=nonempty_e=0.0
    physical_N=physical_E=physical_L=0.0
    d1_before,d2_before=histogram(model.G,model.E)
    snapshots=[]
    number_n,number_e,L_series,density_series,t_degree=[],[],[],[],[]
    final_checkpoint=None
    for t in range(p['T']):

        dt=min(value[1] for value in model.time_list.values())
        N,M=len(model.G),len(model.E)
        L=int(np.arange(len(d1_before))@d1_before)
        physical_N+=dt*N;physical_E+=dt*M;physical_L+=dt*L
        if N:
            time_p_n=add_histogram(time_p_n,d1_before,dt/N);nonempty_n+=dt
        if M:
            time_p_e=add_histogram(time_p_e,d2_before,dt/M);nonempty_e+=dt
        time_now+=next_event(p,del_node,del_edge)
        d1_before,d2_before=histogram(model.G,model.E)
        event_counts_n=add_histogram(event_counts_n,d1_before)
        event_counts_e=add_histogram(event_counts_e,d2_before)
        N,M=len(model.G),len(model.E)
        L=int(np.arange(len(d1_before))@d1_before)
        number_n.append(N);number_e.append(M);L_series.append(L)
        density_series.append(L/(N*M) if N and M else 0.0)
        t_degree.append(time_now)
        if (t+1)%p['sample_every']==0 or t==p['T']-1:
            values=topology(model.G,model.E,p['compute_apl'])
            snapshots.append(dict(event=t+1,time=time_now,**values))
            final_checkpoint=values
            print(real['dataset'],seed,'sample',t+1,'/',p['T'],'C=',round(values['C'],4),flush=True)
    duration=time_now-time_start
    half=max(1,p['T']//2)
    block_check={}
    for key,series in [('N',number_n),('E',number_e),('L',L_series)]:
        block_check[key]=dict(first_half_mean=float(np.mean(series[:half])),
                              second_half_mean=float(np.mean(series[half:])) if series[half:] else None)
    for key in ('C','PLCC'):
        vals=[v[key] for v in snapshots];split=max(1,len(vals)//2)
        block_check[key]=dict(first_half_mean=float(np.mean(vals[:split])),
                              second_half_mean=float(np.mean(vals[split:])) if vals[split:] else None)

    result=dict(schema_version='real_simulation_v1',implementation_revision='ibad_empirical_v1',
        dataset=real['dataset'],real_signature=real_identity(real),parameters=dict(p,seed=seed),
        real_reference={k:real[k] for k in ('dataset','preprocessing','P_node','P_edge','metrics')},
        node_counts_mean=(event_counts_n/p['T']).tolist(),edge_counts_mean=(event_counts_e/p['T']).tolist(),
        P_node_time=(time_p_n/nonempty_n).tolist() if nonempty_n else probability(time_p_n),
        P_edge_time=(time_p_e/nonempty_e).tolist() if nonempty_e else probability(time_p_e),
        node_nonempty_time=nonempty_n,edge_nonempty_time=nonempty_e,
        number_n=number_n,number_e=number_e,L_series=L_series,density_series=density_series,t_degree=t_degree,
        topology_samples=snapshots,warmup_trace=warm_trace,
        warmup_time=time_start,observation_duration=duration,
        stationary_population_targets=dict(N=p['lambda_n']/p['mu_n'],E=p['lambda_e']/p['mu_e']),
        physical_time_means=dict(N=physical_N/duration,E=physical_E/duration,L=physical_L/duration),
        initialization_survivors_after_warmup=residual,half_window_diagnostic=block_check,
        inheritance_stats_warmup=warm_stats,inheritance_stats_sampling=dict(model.inheritance_stats),
        final_metrics=final_checkpoint,simulation_seconds=perf_counter()-start,
        notes=dict(stationarity='warmup length is user set; diagnostics are not a stationarity proof',
                   degree='event mean counts and physical-time mean PMFs stored separately; k=0 retained',
                   topology='post-event samples every sample_every events plus final event, not exact time average',
                   initial_history='starts with the degree of the small initial network',
                   comparison='post-burn-in averages from small-network initialization',
                   N_triadic='sum_e choose(size(e),2), not number of three-node triangles'))
    if cfg.SAVE_FINAL_NETWORK:
        result['final_nodes']=list(model.G)
        result['final_hyperedges']=model.E
    if save:
        fingerprint=hashlib.sha256(json.dumps(dict(parameters=p,real_signature=result['real_signature']),sort_keys=True).encode()).hexdigest()[:12]
        folder=cfg.RESULT_DIR/real['dataset']/f"{p['deletion_mode']}_{p['similarity_rule']}_{fingerprint}"/(
            f"seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}")
        save_json(folder/'results.json',result)
        print('Saved: ',folder/'results.json',flush=True)
    return result


def main():
    for name in cfg.NETWORKS:
        path=cfg.REAL_DIR/f'{name}.json'
        if not path.exists():
            raise SystemExit('Run prepare_real.py first: '+str(path))
        real=json.loads(path.read_text(encoding='utf-8'))
        if any(real['preprocessing'].get(k)!=v for k,v in cfg.DATASETS[name].items()):
            raise SystemExit('Preprocessing changed; rerun prepare_real.py: '+name)
        p=parameters_for(real)
        print(name,'mu: ',p['mu_n'],p['mu_e'],'population targets: ',p['lambda_n']/p['mu_n'],p['lambda_e']/p['mu_e'])
        for seed in cfg.SEEDS:
            run_one(real,p,seed)

if __name__=='__main__':
    main()
