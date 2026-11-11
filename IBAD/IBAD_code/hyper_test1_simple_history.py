"""Simulate one IBAD hypergraph run and save list-based JSON results."""
import networkx as nx
import random
import numpy as np
import json
from pathlib import Path
from array import array
from bisect import bisect_right
from itertools import accumulate
from time import perf_counter


def pearson_fast(x,y,x_cache=None):


    if x_cache is None:
        x_cache={}
    length=len(x)
    if length not in x_cache:
        x=x-x.mean()
        xmax=np.max(np.abs(x))
        norm_x=xmax*np.sqrt(np.sum((x/xmax)**2)) if xmax else 0.0
        x_cache[length]=x/norm_x if norm_x else None
    x=x_cache[length]
    if x is None:
        return float('nan')
    y=y-y.mean()
    ymax=np.max(np.abs(y))
    if ymax==0:
        return float('nan')
    norm_y=ymax*np.sqrt(np.sum((y/ymax)**2))
    r=float(np.dot(x,y/norm_y))
    return max(-1.0,min(1.0,r)) if np.isfinite(r) else float('nan')


def history_distance_squared(x,y,x_cache=None):
    # Compare the full common suffix; raw and log histories use different distance units.


    if x_cache is None:
        x_cache={}
    length=len(x)
    if length not in x_cache:
        x_cache[length]=np.log1p(x) if similarity_rule=='history_rbf' else x
    y=np.log1p(y) if similarity_rule=='history_rbf' else y
    difference=x_cache[length]-y
    return float(np.dot(difference,difference)/length)


def history_rbf_weights(distances):
    # A shared exponent shift avoids underflow without changing normalized probabilities.


    if not distances:
        return []
    distances=np.asarray(distances,dtype=float)
    with np.errstate(over='ignore',under='ignore'):
        weights=np.exp(-0.5*((distances-distances.min())/similarity_sigma)/similarity_sigma)
    return weights.tolist()


def get_node(m_n):
    # Preferential sampling of distinct hyperedges, with weights fixed before birth.
    global l_N
    edge_list=list(E)
    sum_e=list(accumulate(len(E[key])+a_e for key in edge_list))
    add_list=[]
    selected=set()
    for m in range(min(m_n,len(edge_list))):
        key=edge_list[min(bisect_right(sum_e,random.random()*sum_e[-1]),len(edge_list)-1)]
        while key in selected:
            key=edge_list[min(bisect_right(sum_e,random.random()*sum_e[-1]),len(edge_list)-1)]
        add_list.append(key)
        selected.add(key)
    l_N+=1
    G.add_node(l_N)
    G.nodes[l_N]['E']=add_list
    for key in add_list:
        E[key].append(l_N)
    time_list[f"{l_N}n"]=['n',np.random.exponential(1/mu_n)]
    his_n[l_N]=array('d')


def del_node_1(num_node,count_stats=True):
    if count_stats:
        inheritance_stats['node_lost']+=len(G.nodes[num_node]['E'])
        inheritance_stats['node_direct']+=len(G.nodes[num_node]['E'])
    for key in G.nodes[num_node]['E']:
        E[key].remove(num_node)
    G.remove_node(num_node)
    time_list.pop(f"{num_node}n")
    his_n.pop(num_node)


def get_E(m_e):
    global l_E
    l_E+=1
    node_list=list(G.nodes)
    sum_d=list(accumulate(len(G.nodes[n]['E'])+a_n for n in node_list))
    add_list=[]
    selected=set()
    for m in range(min(m_e,len(node_list))):
        i=min(bisect_right(sum_d,random.random()*sum_d[-1]),len(node_list)-1)
        while i in selected:
            i=min(bisect_right(sum_d,random.random()*sum_d[-1]),len(node_list)-1)
        add_list.append(i)
        selected.add(i)
    E[l_E]=[]
    for i in add_list:
        temp_n=node_list[i]
        G.nodes[temp_n]['E'].append(l_E)
        E[l_E].append(temp_n)
    time_list[f"{l_E}e"]=['e',np.random.exponential(1/mu_e)]
    his_e[l_E]=array('d')
    return E


def del_E(key_to_remove,count_stats=True):
    if count_stats:
        inheritance_stats['edge_lost']+=len(E[key_to_remove])
        inheritance_stats['edge_direct']+=len(E[key_to_remove])
    for node in E[key_to_remove]:
        G.nodes[node]['E'].remove(key_to_remove)
    del E[key_to_remove]
    time_list.pop(f"{key_to_remove}e")
    his_e.pop(key_to_remove)
    return E


def del_E_H(key_to_remove,uniform=False,feasible_only=False,positive_only=False):
    # Compute candidate weights once; a duplicate single proposal is skipped.
    r_list={}
    x=np.frombuffer(his_e[key_to_remove],dtype=float)
    x_cache={}
    use_history=(not uniform and not positive_only and
                 similarity_rule in ('history_rbf','history_rbf_raw'))
    undefined_pairs=nonpositive_pairs=positive_pairs=0
    for e in E:
        if e==key_to_remove:
            continue
        shorter=min(len(his_e[e]),len(x))
        if shorter<min_history:
            continue
        if uniform and not positive_only:
            r_list[e]=1.0
            continue
        if use_history:

            r_list[e]=history_distance_squared(
                x[-shorter:],np.frombuffer(his_e[e],dtype=float)[-shorter:],x_cache)
            continue
        r_=pearson_fast(
            x[-shorter:],np.frombuffer(his_e[e],dtype=float)[-shorter:],x_cache)
        if not np.isfinite(r_):
            undefined_pairs+=1
        elif r_>0:
            positive_pairs+=1
        else:
            nonpositive_pairs+=1
        if positive_only or similarity_rule=='positive':
            if np.isfinite(r_) and r_>0:
                r_list[e]=1.0 if uniform else r_
        else:
            r_=r_ if np.isfinite(r_) else 0.0
            if r_+1>0:
                r_list[e]=r_+1
    edge_list=list(r_list)
    sum_r=list(accumulate(history_rbf_weights(list(r_list.values())) if use_history else r_list.values()))
    if use_history:
        inheritance_stats['edge_history_pairs']+=len(r_list)
    success=duplicate=no_candidate=0
    for n in E[key_to_remove]:

        if feasible_only:
            edge_list=[e for e in r_list if e not in G.nodes[n]['E']]
            values=[r_list[e] for e in edge_list]
            sum_r=list(accumulate(history_rbf_weights(values) if use_history else values))
        if not sum_r:
            no_candidate+=1
            continue
        e=edge_list[min(bisect_right(sum_r,random.random()*sum_r[-1]),len(edge_list)-1)]
        if e in G.nodes[n]['E']:
            duplicate+=1
            continue
        G.nodes[n]['E'].append(e)
        E[e].append(n)
        success+=1
    inheritance_stats['edge_lost']+=len(E[key_to_remove])
    inheritance_stats['edge_success']+=success
    inheritance_stats['edge_duplicate']+=duplicate
    inheritance_stats['edge_no_candidate']+=no_candidate
    inheritance_stats['edge_undefined_pairs']+=undefined_pairs
    inheritance_stats['edge_nonpositive_pairs']+=nonpositive_pairs
    inheritance_stats['edge_positive_pairs']+=positive_pairs
    return del_E(key_to_remove,count_stats=False)


def del_N_H(num_node,uniform=False,feasible_only=False,positive_only=False):
    # Compute candidate weights once; a duplicate single proposal is skipped.
    r_list={}
    x=np.frombuffer(his_n[num_node],dtype=float)
    x_cache={}
    use_history=(not uniform and not positive_only and
                 similarity_rule in ('history_rbf','history_rbf_raw'))
    undefined_pairs=nonpositive_pairs=positive_pairs=0
    for n in G.nodes:
        if n==num_node:
            continue
        shorter=min(len(his_n[n]),len(x))
        if shorter<min_history:
            continue
        if uniform and not positive_only:
            r_list[n]=1.0
            continue
        if use_history:
            r_list[n]=history_distance_squared(
                x[-shorter:],np.frombuffer(his_n[n],dtype=float)[-shorter:],x_cache)
            continue
        r_=pearson_fast(
            x[-shorter:],np.frombuffer(his_n[n],dtype=float)[-shorter:],x_cache)
        if not np.isfinite(r_):
            undefined_pairs+=1
        elif r_>0:
            positive_pairs+=1
        else:
            nonpositive_pairs+=1
        if positive_only or similarity_rule=='positive':
            if np.isfinite(r_) and r_>0:
                r_list[n]=1.0 if uniform else r_
        else:
            r_=r_ if np.isfinite(r_) else 0.0
            if r_+1>0:
                r_list[n]=r_+1
    node_list=list(r_list)
    sum_r=list(accumulate(history_rbf_weights(list(r_list.values())) if use_history else r_list.values()))
    if use_history:
        inheritance_stats['node_history_pairs']+=len(r_list)
    success=duplicate=no_candidate=0
    for e in G.nodes[num_node]['E']:
        if feasible_only:
            node_list=[n for n in r_list if e not in G.nodes[n]['E']]
            values=[r_list[n] for n in node_list]
            sum_r=list(accumulate(history_rbf_weights(values) if use_history else values))
        if not sum_r:
            no_candidate+=1
            continue
        n=node_list[min(bisect_right(sum_r,random.random()*sum_r[-1]),len(node_list)-1)]
        if e in G.nodes[n]['E']:
            duplicate+=1
            continue
        G.nodes[n]['E'].append(e)
        E[e].append(n)
        success+=1
    inheritance_stats['node_lost']+=len(G.nodes[num_node]['E'])
    inheritance_stats['node_success']+=success
    inheritance_stats['node_duplicate']+=duplicate
    inheritance_stats['node_no_candidate']+=no_candidate
    inheritance_stats['node_undefined_pairs']+=undefined_pairs
    inheritance_stats['node_nonpositive_pairs']+=nonpositive_pairs
    inheritance_stats['node_positive_pairs']+=positive_pairs
    del_node_1(num_node,count_stats=False)
    return G


def del_N_random(num_node):
    return del_N_H(num_node,uniform=True)


def del_E_random(key_to_remove):
    return del_E_H(key_to_remove,uniform=True)


def del_N_random_positive(num_node):
    return del_N_H(num_node,uniform=True,positive_only=True)


def del_E_random_positive(key_to_remove):
    return del_E_H(key_to_remove,uniform=True,positive_only=True)


def del_N_feasible(num_node):
    return del_N_H(num_node,feasible_only=True)


def del_E_feasible(key_to_remove):
    return del_E_H(key_to_remove,feasible_only=True)


def degree(G,E):
    d1_=np.bincount([len(G.nodes[n]['E']) for n in G.nodes],minlength=1)
    d2_=np.bincount([len(E[e]) for e in E],minlength=1)
    return d1_,d2_

def clusering(G,E):
    # Exclude triples lying within a single hyperedge; count each projected triple once.


    node_list=list(G.nodes)
    neighbors={n:set() for n in node_list}
    node_edges={n:set() for n in node_list}
    order={n:i for i,n in enumerate(node_list)}
    for key in E.keys():
        members=set(E[key])
        for n in members:
            neighbors[n].update(members)
            neighbors[n].discard(n)
            node_edges[n].add(key)

    a=0
    b=sum(len(neighbors[n])*(len(neighbors[n])-1)//2 for n in node_list)
    for n1 in node_list:
        for n2 in neighbors[n1]:
            if order[n2]<=order[n1]:
                continue
            for n3 in neighbors[n1] & neighbors[n2]:
                if order[n3]<=order[n2]:
                    continue
                if node_edges[n1] & node_edges[n2] & node_edges[n3]:
                    b-=3
                else:
                    a+=3
    if b==0:
        return 0
    else:
        return a/b

def apl(E):
    global G

    G_now=nx.Graph()
    G_now.add_nodes_from(G.nodes)
    for key in E.keys():
        for i in range(len(E[key])):
            for j in range(i+1,len(E[key])):
                G_now.add_edge(E[key][i],E[key][j])
    if len(G_now.nodes)==0:
        return 0.0,0.0
    largest=max(nx.connected_components(G_now),key=len)
    G_largest=G_now.subgraph(largest)
    return(len(G_largest.nodes)/len(G_now.nodes),nx.average_shortest_path_length(G_largest))


def triadic(E):
    # Q2 counts within-hyperedge node pairs, retaining multiplicity across hyperedges.
    tri=0

    for key in E.keys():
        len_e=len(E[key])
        tri+=((len_e-1)*len_e//2)
    return tri


G = nx.Graph()
l_N=10
l_E=1
H = nx.path_graph(l_N)
G.add_nodes_from(H)

E={}
E[l_E]=list(G.nodes)

for i in range(len(G.nodes)):
    G.nodes[i]['E']=[l_E]
l_E+=1

node_num=len(G.nodes)


m_e=1
m_n=1
lambda_e=2
mu_e=1/80
lambda_n=2
mu_n=1/80


a_n=1.0
a_e=1.0
min_history=100
similarity_rule='history_rbf_raw'
similarity_sigma=1.0

T_pre=100000
T=20000
sample_every=400
seed=None
deletion_mode='ibad_single'
save_output=True
output_root=Path(__file__).resolve().parent/'results_simple'

# Optional overrides supplied by run_experiments.py.
for parameter, value in globals().get('run_parameters', {}).items():
    if parameter not in {'m_n','m_e','lambda_n','lambda_e','mu_n','mu_e','a_n','a_e',
                         'min_history','similarity_rule','similarity_sigma','T_pre','T',
                         'sample_every','seed','deletion_mode','save_output','output_root'}:
        raise ValueError(f'Unknown run parameter: {parameter}')
    globals()[parameter] = value

if seed is None:
    seed=random.SystemRandom().randrange(2**32)
random.seed(seed)
np.random.seed(seed)
if not all(np.isfinite(v) and v>0 for v in (lambda_n,lambda_e,mu_n,mu_e)):
    raise ValueError('lambda and mu must be finite positive rates')
if not all(np.isfinite(v) and v>0 for v in (a_n,a_e)):
    raise ValueError('a_n and a_e must be finite and positive')
if not isinstance(min_history,int) or isinstance(min_history,bool) or min_history<3:
    raise ValueError('min_history must be an integer >= 3')
if similarity_rule not in ('positive','shifted','history_rbf','history_rbf_raw'):
    raise ValueError('similarity_rule must be positive, shifted, history_rbf or history_rbf_raw')
if not np.isfinite(similarity_sigma) or similarity_sigma<=0:
    raise ValueError('similarity_sigma must be finite and positive')
if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in (m_n,m_e,T_pre)):
    raise ValueError('m_n, m_e and T_pre must be nonnegative integers')
if any(not isinstance(v,int) or isinstance(v,bool) or v<1 for v in (T,sample_every)):
    raise ValueError('T and sample_every must be positive integers')
del_node_function,del_edge_function={
    'ibad_single':(del_N_H,del_E_H),
    'direct':(del_node_1,del_E),
    'uniform_single':(del_N_random,del_E_random),
    'uniform_positive':(del_N_random_positive,del_E_random_positive),
    'ibad_feasible':(del_N_feasible,del_E_feasible)
}[deletion_mode]
inheritance_stats={kind+'_'+item:0 for kind in ('node','edge')
                   for item in ('lost','success','duplicate','no_candidate','direct',
                                'undefined_pairs','nonpositive_pairs','positive_pairs','history_pairs')}
simulation_start=perf_counter()

time_list={}
birth=np.random.exponential(1/lambda_n)
time_list['birth_n']=['birth_n',birth]
birth=np.random.exponential(1/lambda_e)
time_list['birth_e']=['birth_e',birth]
for i in range(len(list(G.nodes))):
    death=np.random.exponential(1/mu_n)
    time_list[f"{list(G.nodes)[i]}n"]=['n',death]
for i in E.keys():
    death=np.random.exponential(1/mu_e)
    time_list[f"{i}e"]=['e',death]

his_e={}
his_n={}

for n in list(G.nodes):
    his_n[n]=array('d',[len(G.nodes[n]['E'])])
for e in E.keys():
    his_e[e]=array('d',[len(E[e])])


number_e=[]
number_n=[]
time_now=0.0
t_population=[]

for t in range(T_pre):
    do_ = min (time_list, key = lambda x: time_list[x][1])
    time_gap=time_list[do_][1]
    time_now+=time_gap
    if time_list[do_][0]=='birth_e':
        for key in time_list.keys():
            time_list[key][1]-=time_gap
        E=get_E(m_e)
        birth=np.random.exponential(1/lambda_e)
        time_list['birth_e']=['birth_e',birth]
    elif time_list[do_][0]=='birth_n':
        for key in time_list.keys():
            time_list[key][1]-=time_gap
        get_node(m_n)
        birth=np.random.exponential(1/lambda_n)
        time_list['birth_n']=['birth_n',birth]
    elif time_list[do_][0]=='n':
        for key in time_list.keys():
            time_list[key][1]-=time_gap
        do_number=int(do_[:-1])
        del_node_function(do_number)
    elif time_list[do_][0]=='e':
        for key in time_list.keys():
            time_list[key][1]-=time_gap
        do_number=int(do_[:-1])
        E=del_edge_function(do_number)


    if deletion_mode!='direct':
        for n in G.nodes:
            if deletion_mode!='uniform_single' or len(his_n[n])<min_history:
                his_n[n].append(len(G.nodes[n]['E']))
        for e in E:
            if deletion_mode!='uniform_single' or len(his_e[e])<min_history:
                his_e[e].append(len(E[e]))

    number_e.append(len(his_e))
    number_n.append(len(his_n))
    t_population.append(time_now)


inheritance_stats_warmup=inheritance_stats.copy()
inheritance_stats=dict.fromkeys(inheritance_stats,0)
time_start=time_now

d1_before,d2_before=degree(G,E)
d1_integral=np.zeros(len(d1_before))
d2_integral=np.zeros(len(d2_before))
P_node_integral=np.zeros(len(d1_before))
P_edge_integral=np.zeros(len(d2_before))
node_nonempty_time=edge_nonempty_time=0.0
number_n_time={}
number_e_time={}

d1=[]
d2=[]
t_degree=[]
cluser_=[]
apl_=[]
plcc=[]
N_triadic=[]
t_topology=[]
L_series=[]
density_series=[]
for t in range(T):
    do_ = min (time_list, key = lambda x: time_list[x][1])
    time_gap=time_list[do_][1]
    number_n_time[len(G)]=number_n_time.get(len(G),0.0)+time_gap
    number_e_time[len(E)]=number_e_time.get(len(E),0.0)+time_gap

    for counts,total,prob_total in ((d1_before,d1_integral,P_node_integral),
                                    (d2_before,d2_integral,P_edge_integral)):
        total[:len(counts)]+=time_gap*counts
        population=int(counts.sum())
        if population:
            prob_total[:len(counts)]+=time_gap/population*counts
    if d1_before.sum()>0:
        node_nonempty_time+=time_gap
    if d2_before.sum()>0:
        edge_nonempty_time+=time_gap
    time_now+=time_gap
    if time_list[do_][0]=='birth_e':
        for key in time_list.keys():
            time_list[key][1]-=time_gap
        E=get_E(m_e)
        birth=np.random.exponential(1/lambda_e)
        time_list['birth_e']=['birth_e',birth]
    elif time_list[do_][0]=='birth_n':
        for key in time_list.keys():
            time_list[key][1]-=time_gap
        get_node(m_n)
        birth=np.random.exponential(1/lambda_n)
        time_list['birth_n']=['birth_n',birth]
    elif time_list[do_][0]=='n':
        for key in time_list.keys():
            time_list[key][1]-=time_gap
        do_number=int(do_[:-1])
        del_node_function(do_number)
    elif time_list[do_][0]=='e':
        for key in time_list.keys():
            time_list[key][1]-=time_gap
        do_number=int(do_[:-1])
        E=del_edge_function(do_number)


    if deletion_mode!='direct':
        for n in G.nodes:
            if deletion_mode!='uniform_single' or len(his_n[n])<min_history:
                his_n[n].append(len(G.nodes[n]['E']))
        for e in E:
            if deletion_mode!='uniform_single' or len(his_e[e])<min_history:
                his_e[e].append(len(E[e]))

    number_e.append(len(his_e))
    number_n.append(len(his_n))
    t_population.append(time_now)

    d1_,d2_=degree(G, E)
    d1.append(d1_.tolist())
    d2.append(d2_.tolist())
    L_series.append(int(np.dot(np.arange(len(d1_)),d1_)))
    density_series.append(L_series[-1]/(len(G)*len(E)) if len(G) and len(E) else 0.0)

    d1_before,d2_before=d1_,d2_
    if len(d1_)>len(d1_integral):
        d1_integral=np.pad(d1_integral,(0,len(d1_)-len(d1_integral)))
        P_node_integral=np.pad(P_node_integral,(0,len(d1_)-len(P_node_integral)))
    if len(d2_)>len(d2_integral):
        d2_integral=np.pad(d2_integral,(0,len(d2_)-len(d2_integral)))
        P_edge_integral=np.pad(P_edge_integral,(0,len(d2_)-len(P_edge_integral)))
    t_degree.append(time_now)
    if (t%sample_every)==0:
        cluser_.append(clusering(G, E))
        plcc_i,apl_i=apl(E)
        apl_.append(apl_i)
        plcc.append(plcc_i)
        N_triadic.append(triadic(E))
        t_topology.append(time_now)


max_d1=max(len(row) for row in d1)
max_d2=max(len(row) for row in d2)
for row in d1:
    row.extend([0]*(max_d1-len(row)))
for row in d2:
    row.extend([0]*(max_d2-len(row)))
# The final row is the observation-window mean count at each integer degree.
d1.append(np.mean(d1,axis=0).tolist())
d2.append(np.mean(d2,axis=0).tolist())


P_node_time=(P_node_integral/node_nonempty_time).tolist() if node_nonempty_time else P_node_integral.tolist()
P_edge_time=(P_edge_integral/edge_nonempty_time).tolist() if edge_nonempty_time else P_edge_integral.tolist()
duration=time_now-time_start
d1_time_average=(d1_integral/duration).tolist() if duration else d1_integral.tolist()
d2_time_average=(d2_integral/duration).tolist() if duration else d2_integral.tolist()

P_node_pooled=(d1_integral/d1_integral.sum()).tolist() if d1_integral.sum() else d1_integral.tolist()
P_edge_pooled=(d2_integral/d2_integral.sum()).tolist() if d2_integral.sum() else d2_integral.tolist()
simulation_seconds=perf_counter()-simulation_start


results={
    'schema_version':'simple_v2',
    'implementation_revision':'ibad_v1',
    'parameters':{'lambda_n':lambda_n,'lambda_e':lambda_e,'mu_n':mu_n,'mu_e':mu_e,
                  'm_n':m_n,'m_e':m_e,'T_pre':T_pre,'T':T,'sample_every':sample_every,
                  'seed':seed,'deletion_mode':deletion_mode,
                  'a_n':a_n,'a_e':a_e,'min_history':min_history,'similarity_rule':similarity_rule,
                  'similarity_sigma':similarity_sigma if similarity_rule.startswith('history_rbf') and deletion_mode in ('ibad_single','ibad_feasible') else None},
    'notes':{'rates':'lambda: birth rate; mu: per-object death rate; exponential scale=1/rate',
             'd1/d2':'T post-event count snapshots + one event-average count row; index is degree',
             'P_time':'time-average of within-state PMFs, conditional on nonempty population',
             'P_pooled':'normalized integrated counts (node-time / edge-time sampling)',
             'number_n/e':'includes warmup; formal post-event records start at index T_pre',
             'topology':'post-event samples, not physical-time averages',
             'birth_weights':'new node: edge size+a_e; new edge: node hyperdegree+a_n; distinct targets',
             'history':f'common full event history, at least {min_history} points; not independent samples',
             'similarity':{'positive':'finite r>0, weight=r; undefined Pearson excluded',
                           'shifted':'weight=r+1; undefined Pearson treated as r=0',
                           'history_rbf':'exp(-mean((log1p(k_i)-log1p(k_j))**2)/(2*sigma**2)); constants eligible',
                           'history_rbf_raw':'exp(-mean((k_i-k_j)**2)/(2*sigma**2)); constants eligible'}[similarity_rule],
             'similarity_scope':'similarity_rule applies to ibad_single/ibad_feasible; uniform_positive always uses finite positive Pearson; uniform_single and direct ignore similarity_rule',
             'weight_evaluation':'once per deletion; histories frozen; RBF weights subtract minimum squared distance before exponentiation; no similarity cutoff; feasible comparison rescales within each feasible subset',
             'uniform_single':'all other history-eligible objects, equal weights; no Pearson filter',
             'uniform_positive':'only finite r>0 history-eligible objects, equal weights',
             'inheritance':'one draw per lost incidence; existing incidence is skipped without redraw; no weight means no inheritance; ibad_feasible is a separate conditional-on-feasibility comparison',
             'pair_counters':'undefined/nonpositive/positive_pairs count Pearson comparisons; history_pairs counts history-distance comparisons; repeated across deletions, not unique objects; uniform_single/direct compute neither',
             'N_triadic':'sum_e size(e)*(size(e)-1)//2, with multiplicity across hyperedges',
             'clustering':'projected clustering excluding within-hyperedge triples; zero denominator = 0'},
    'number_n':number_n,'number_e':number_e,'t_population':t_population,
    'd1':d1,'d2':d2,'t_degree':t_degree,
    'P_node_time':P_node_time,'P_edge_time':P_edge_time,
    'P_node_pooled':P_node_pooled,'P_edge_pooled':P_edge_pooled,
    'd1_time_average':d1_time_average,'d2_time_average':d2_time_average,
    'time_start':time_start,'duration':duration,
    'node_nonempty_time':node_nonempty_time,'edge_nonempty_time':edge_nonempty_time,
    'node_empty_time':max(0.0,duration-node_nonempty_time),
    'edge_empty_time':max(0.0,duration-edge_nonempty_time),
    'number_n_time_distribution':{k:v/duration for k,v in number_n_time.items()} if duration else {},
    'number_e_time_distribution':{k:v/duration for k,v in number_e_time.items()} if duration else {},
    'cluser_':cluser_,'apl_':apl_,'plcc':plcc,'N_triadic':N_triadic,'t_topology':t_topology,
    'L_series':L_series,'density_series':density_series,
    'inheritance_stats_warmup':inheritance_stats_warmup,
    'inheritance_stats_sampling':inheritance_stats.copy(),
    'final_nodes':list(G.nodes),'final_hyperedges':E,
    'simulation_seconds':simulation_seconds
}
if save_output:

    parameter_name=(f'mn{m_n}_me{m_e}_ln{lambda_n!r}_le{lambda_e!r}'
                    f'_mun{mu_n!r}_mue{mu_e!r}'
                    f'_an{a_n!r}_ae{a_e!r}_h{min_history}_{similarity_rule}')
    if results['parameters']['similarity_sigma'] is not None:
        parameter_name+=f'_sig{similarity_sigma!r}'
    parent=Path(output_root)/'simple_v2'/deletion_mode/parameter_name
    parent.mkdir(parents=True,exist_ok=True)
    index=1
    while True:
        output_dir=parent/f'seed{seed}_pre{T_pre}_T{T}_sample{sample_every}_run{index:03d}'
        try:
            output_dir.mkdir()
            break
        except FileExistsError:
            index+=1
    result_file=output_dir/'results.json'
    temporary_file=output_dir/'results.tmp'
    with temporary_file.open('x',encoding='utf-8') as f:
        json.dump(results,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    temporary_file.replace(result_file)
    print('Result lists saved to: ',result_file.resolve())
print(f'Runtime {simulation_seconds:.3f} s; seed={seed}; mode={deletion_mode}')
print('Time-averaged node/edge probability sums: ',sum(P_node_time),sum(P_edge_time))
print(f'Settings: a_n={a_n}, a_e={a_e}, min_history={min_history}, similarity_rule={similarity_rule}, effective sigma={results["parameters"]["similarity_sigma"]}')
print(f'Observation duration {duration:.6g}; {duration*mu_n:.3f} node and {duration*mu_e:.3f} edge mean lifetimes')
print(f'Topology samples: {len(t_topology)}; degree counts use {T} post-event states')
