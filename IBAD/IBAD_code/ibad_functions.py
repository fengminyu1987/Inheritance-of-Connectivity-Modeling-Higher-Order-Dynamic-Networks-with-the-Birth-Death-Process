"""Birth, deletion, inheritance and topology functions for the empirical experiments."""
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
