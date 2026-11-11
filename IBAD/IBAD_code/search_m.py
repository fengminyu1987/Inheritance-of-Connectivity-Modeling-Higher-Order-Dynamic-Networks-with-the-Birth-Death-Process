"""Calibrate integer attachment parameters using node- and hyperedge-side JS divergence."""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
import numpy as np
import real_config as cfg
import simulate_real as sim
import ibad_functions as model
from real_utils import histogram, probability, distribution_distances, save_json, write_csv


SEARCH_NETWORKS = ['PPN']
M_N_BOUNDS = (1, 16)
M_E_BOUNDS = (1, 16)
COARSE_VALUES = [1, 2, 4, 8, 16]
LOCAL_CENTERS = 2
LOCAL_RADIUS = 1
FINALISTS = 3
SCREEN_T = 3000
SEARCH_SEEDS = [281176436, 281176437, 281176438]
VALIDATION_SEEDS = list(cfg.SEEDS)
NODE_WEIGHT = 0.5
EDGE_WEIGHT = 0.5
AUTO_WARMUP_FLOOR = True
INITIAL_SURVIVAL_TARGET = 0.001
MAX_WORKERS = 1
RUN_VALIDATION = True
SEARCH_DIR = cfg.ROOT / 'm_search'


SEARCH_VERSION = 'm_search_v1'


def atomic_save(path, data):
    path=Path(path)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    save_json(temporary,data)
    temporary.replace(path)


def score_result(real, result):
    losses=[]; js=[]
    for field in ('node','edge'):
        q=probability(result[field+'_counts_mean'])
        value=distribution_distances(real['P_'+field],q)['JS']
        js.append(value)
        losses.append(1.0 if value is None else max(0.,value/math.log(2)))

    return dict(JS_n=js[0],JS_e=js[1],
                score=(NODE_WEIGHT*losses[0]+EDGE_WEIGHT*losses[1])/(NODE_WEIGHT+EDGE_WEIGHT))


def screen_run(real, p, seed):

    started=perf_counter()
    del_node,del_edge=sim.initialize(real,p,seed)
    original_n,original_e=set(model.G),set(model.E)
    elapsed=0.
    for _ in range(p['T_pre']):
        elapsed+=sim.next_event(p,del_node,del_edge)
    warmup_time=elapsed
    remaining_n=len(original_n & set(model.G))/max(1,len(original_n))
    remaining_e=len(original_e & set(model.E))/max(1,len(original_e))
    totals=[np.zeros(1),np.zeros(1)]
    first=[np.zeros(1),np.zeros(1)]
    half=p['T']//2
    links=np.zeros(2); Ns=Es=0.
    for t in range(p['T']):
        elapsed+=sim.next_event(p,del_node,del_edge)
        dn,de=histogram(model.G,model.E)
        for i,counts in enumerate((dn,de)):
            totals[i]=sim.add_histogram(totals[i],counts)
            if t<half:first[i]=sim.add_histogram(first[i],counts)
        links[int(t>=half)]+=np.arange(len(dn))@dn
        Ns+=len(model.G);Es+=len(model.E)
    half_js=[]
    for total,a in zip(totals,first):
        a=np.pad(a,(0,len(total)-len(a)))
        half_js.append(distribution_distances(probability(a),probability(total-a))['JS'])
    means=links/np.array([half,p['T']-half])
    return dict(node_counts_mean=(totals[0]/p['T']).tolist(),
                edge_counts_mean=(totals[1]/p['T']).tolist(),
                N_mean=Ns/p['T'],E_mean=Es/p['T'],L_mean=float(sum(links)/p['T']),
                L_first_half=float(means[0]),L_second_half=float(means[1]),
                L_half_relative_change=float(abs(means[1]-means[0])/max(float(np.mean(means)),1.)),
                half_JS_n=half_js[0],half_JS_e=half_js[1],
                warmup_time=warmup_time,remaining_n_fraction=remaining_n,remaining_e_fraction=remaining_e,
                runtime_seconds=perf_counter()-started)


def run_job(job):
    path=Path(job['path'])
    if path.exists():
        result=json.loads(path.read_text(encoding='utf-8'))
        if result.get('search_cache_key')!=job['key']:
            raise ValueError('Cache signature mismatch: '+str(path))
        return result,True
    real=json.loads(Path(job['real_path']).read_text(encoding='utf-8'))
    if sim.real_identity(real)!=job['real_signature']:
        raise ValueError('Empirical data changed during calibration; restart the search')
    p=job['parameters']
    print(f"Starting {real['dataset']} m=({p['m_n']},{p['m_e']}) seed={job['seed']} "
          f"pre={p['T_pre']} T={p['T']} {'full validation' if job['full'] else 'distribution screening'}",flush=True)
    if job['full']:
        result=sim.run_one(real,job['parameters'],job['seed'],save=False)
    else:
        result=screen_run(real,job['parameters'],job['seed'])
        result.update(parameters=dict(job['parameters'],seed=job['seed']),dataset=real['dataset'])
    result['search_cache_key']=job['key']
    atomic_save(path,result)
    return result,False


def evaluate(real, real_path, base, pairs, seeds, observation, full, core_hash):
    jobs=[]
    for mn,me in sorted(set(pairs)):
        p=dict(base,m_n=mn,m_e=me,T=observation)
        for seed in seeds:
            identity=dict(version=SEARCH_VERSION,core=core_hash,real=sim.real_identity(real),
                          parameters=p,seed=seed,full=full)
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
            path=(cfg.RESULT_DIR/real['dataset']/('m_search_validation_'+key)/f'seed{seed}'/'results.json'
                  if full else SEARCH_DIR/'cache'/real['dataset']/(key+'.json'))
            jobs.append(dict(path=str(path),key=key,real_path=str(real_path),real_signature=sim.real_identity(real),
                             parameters=p,seed=seed,full=full))
    rows=[]
    executor=ProcessPoolExecutor(max_workers=MAX_WORKERS) if MAX_WORKERS>1 else None
    try:
        iterator=executor.map(run_job,jobs) if executor else map(run_job,jobs)
        for i,(job,(result,cached)) in enumerate(zip(jobs,iterator),1):
            scores=score_result(real,result)
            p=job['parameters']
            rows.append(dict(m_n=p['m_n'],m_e=p['m_e'],seed=job['seed'],**scores,
                cached=cached,path=job['path'],
                **{k:result.get(k) for k in ('N_mean','E_mean','L_mean','L_half_relative_change',
                    'half_JS_n','half_JS_e','warmup_time','remaining_n_fraction','remaining_e_fraction','runtime_seconds')}))
            print(f"{real['dataset']} {i}/{len(jobs)} m=({p['m_n']},{p['m_e']}) seed={job['seed']} "
                  f"score={scores['score']:.6f} {'[cached]' if cached else ''}",flush=True)
    finally:
        if executor:executor.shutdown(wait=True,cancel_futures=True)
    ranking=[]
    for mn,me in sorted(set(pairs)):
        selected=[r for r in rows if (r['m_n'],r['m_e'])==(mn,me)]
        scores=[r['score'] for r in selected]
        ranking.append(dict(m_n=mn,m_e=me,score_mean=float(np.mean(scores)),
            score_sd=float(np.std(scores,ddof=1)) if len(scores)>1 else None,runs=len(scores),
            JS_n_mean=float(np.mean([r['JS_n'] for r in selected])) if all(r['JS_n'] is not None for r in selected) else None,
            JS_e_mean=float(np.mean([r['JS_e'] for r in selected])) if all(r['JS_e'] is not None for r in selected) else None))
    ranking.sort(key=lambda row:(row['score_mean'],row['m_n'],row['m_e']))
    return ranking,rows


def main():
    for bound in (M_N_BOUNDS,M_E_BOUNDS):
        if len(bound)!=2 or any(type(x) is not int for x in bound) or not 0<=bound[0]<=bound[1]:
            raise ValueError('m bounds must be ordered nonnegative integers')
    if not SEARCH_SEEDS or len(set(SEARCH_SEEDS))!=len(SEARCH_SEEDS):raise ValueError('Search seeds are empty or duplicated')
    if len(set(VALIDATION_SEEDS))!=len(VALIDATION_SEEDS):raise ValueError('Validation seeds are duplicated')
    if set(SEARCH_SEEDS)&set(VALIDATION_SEEDS):raise ValueError('Validation seeds must differ from calibration seeds')
    if RUN_VALIDATION and not VALIDATION_SEEDS:raise ValueError('Missing validation seeds')
    if not all(type(x) is int and x>=1 for x in (MAX_WORKERS,LOCAL_CENTERS,FINALISTS)):
        raise ValueError('Worker, center and finalist counts must be positive integers')
    if type(LOCAL_RADIUS) is not int or LOCAL_RADIUS<0:raise ValueError('LOCAL_RADIUS must be a nonnegative integer')
    if type(SCREEN_T) is not int or type(cfg.T) is not int or not 2<=SCREEN_T<=cfg.T:
        raise ValueError('Observation windows must be integers with 2<=SCREEN_T<=real_config.T')
    if any(type(x) is not int or x<0 or x>=2**32 for x in SEARCH_SEEDS+VALIDATION_SEEDS):
        raise ValueError('Seeds must be integers in [0, 2**32-1]')
    if not COARSE_VALUES or any(type(x) is not int or x<0 for x in COARSE_VALUES):
        raise ValueError('COARSE_VALUES must be a nonempty list of nonnegative integers')
    if not all(np.isfinite(w) and w>=0 for w in (NODE_WEIGHT,EDGE_WEIGHT)) or NODE_WEIGHT+EDGE_WEIGHT<=0:
        raise ValueError('Score weights must be nonnegative with a positive sum')
    if not 0<INITIAL_SURVIVAL_TARGET<1:raise ValueError('INITIAL_SURVIVAL_TARGET must be in (0,1)')
    core_hash=hashlib.sha256(b''.join((cfg.ROOT/f).read_bytes() for f in
        ('simulate_real.py','ibad_functions.py','real_utils.py'))).hexdigest()
    folder=SEARCH_DIR/'reports'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True,exist_ok=True)
    selections={}
    for name in SEARCH_NETWORKS:
        real_path=cfg.REAL_DIR/(name+'.json')
        if not real_path.exists():raise FileNotFoundError('Run prepare_real.py first: '+str(real_path))
        real=json.loads(real_path.read_text(encoding='utf-8'))
        if any(real['preprocessing'].get(k)!=v for k,v in cfg.DATASETS[name].items()):
            raise ValueError('Preprocessing changed; rerun prepare_real.py: '+name)
        base=sim.parameters_for(real,use_selected=False)
        if base['T_pre']<0 or base['lambda_n']<=0 or base['lambda_e']<=0 or base['mu_n']<=0 or base['mu_e']<=0:
            raise ValueError('Invalid burn-in or birth/death parameters')


        floor=math.ceil(2*(base['lambda_n']+base['lambda_e'])*max(1/base['mu_n'],1/base['mu_e'])*(-math.log(INITIAL_SURVIVAL_TARGET)))
        if AUTO_WARMUP_FLOOR:base['T_pre']=max(base['T_pre'],floor)
        axes=[sorted(set([x for x in COARSE_VALUES if bound[0]<=x<=bound[1]]+list(bound))) for bound in (M_N_BOUNDS,M_E_BOUNDS)]
        pairs={(mn,me) for mn in axes[0] for me in axes[1]}
        current=(base['m_n'],base['m_e'])
        if M_N_BOUNDS[0]<=current[0]<=M_N_BOUNDS[1] and M_E_BOUNDS[0]<=current[1]<=M_E_BOUNDS[1]:pairs.add(current)
        prefix=folder/name
        settings=dict(base=base,real_signature=sim.real_identity(real),core_hash=core_hash,
            m_n_bounds=M_N_BOUNDS,m_e_bounds=M_E_BOUNDS,coarse_values=COARSE_VALUES,
            local_centers=LOCAL_CENTERS,local_radius=LOCAL_RADIUS,finalists=FINALISTS,
            screen_T=SCREEN_T,search_seeds=SEARCH_SEEDS,validation_seeds=VALIDATION_SEEDS,
            node_weight=NODE_WEIGHT,edge_weight=EDGE_WEIGHT,run_validation=RUN_VALIDATION,
            auto_warmup_floor=AUTO_WARMUP_FLOOR,initial_survival_target=INITIAL_SURVIVAL_TARGET,
            score='mean of per-seed weighted JS/ln(2); event mean counts; k=0 included',
            validation='new seeds for selected pair only; not held-out real data')
        atomic_save(prefix.with_name(name+'_settings.json'),settings)
        print(f'\n{name}: {len(pairs)} coarse pairs; burn-in={base["T_pre"]}, observation={SCREEN_T} events.',flush=True)
        coarse,coarse_rows=evaluate(real,real_path,base,pairs,SEARCH_SEEDS[:1],SCREEN_T,False,core_hash)
        write_csv(prefix.with_name(name+'_coarse.csv'),coarse)
        refined=set(pairs)
        for center in coarse[:LOCAL_CENTERS]:
            for mn in range(max(M_N_BOUNDS[0],center['m_n']-LOCAL_RADIUS),min(M_N_BOUNDS[1],center['m_n']+LOCAL_RADIUS)+1):
                for me in range(max(M_E_BOUNDS[0],center['m_e']-LOCAL_RADIUS),min(M_E_BOUNDS[1],center['m_e']+LOCAL_RADIUS)+1):
                    refined.add((mn,me))
        fine,fine_rows=evaluate(real,real_path,base,refined,SEARCH_SEEDS[:1],SCREEN_T,False,core_hash)
        write_csv(prefix.with_name(name+'_screen_ranking.csv'),fine)
        write_csv(prefix.with_name(name+'_screen_runs.csv'),fine_rows)
        shortlist=[(row['m_n'],row['m_e']) for row in fine[:FINALISTS]]
        final,final_rows=evaluate(real,real_path,base,shortlist,SEARCH_SEEDS,cfg.T,False,core_hash)
        write_csv(prefix.with_name(name+'_final_ranking.csv'),final)
        write_csv(prefix.with_name(name+'_final_runs.csv'),final_rows)
        winner=final[0]

        chosen=dict(winner,real_signature=sim.real_identity(real),parameters=dict(base,m_n=winner['m_n'],m_e=winner['m_e']),
                    at_boundary=winner['m_n'] in M_N_BOUNDS or winner['m_e'] in M_E_BOUNDS,
                    interpretation='best tested pair under configured score, not guaranteed global optimum or stationarity')
        atomic_save(prefix.with_name(name+'_selected.json'),chosen)
        atomic_save(cfg.SELECTED_DIR/(name+'.json'),chosen)
        selections[name]=chosen
        print('Selected pair: ',winner,flush=True)
        if chosen['at_boundary']:print('The selected pair lies on a search boundary.',flush=True)
        best_rows=[r for r in final_rows if r['m_n']==winner['m_n'] and r['m_e']==winner['m_e']]
        if any(r['L_half_relative_change']>.2 or r['remaining_n_fraction']>.01 or r['remaining_e_fraction']>.01 for r in best_rows):
            print('Window diagnostics indicate substantial drift or surviving initial individuals.',flush=True)

        if RUN_VALIDATION:
            validation,validation_rows=evaluate(real,real_path,base,[(winner['m_n'],winner['m_e'])],VALIDATION_SEEDS,cfg.T,True,core_hash)
            write_csv(prefix.with_name(name+'_validation.csv'),validation)
            write_csv(prefix.with_name(name+'_validation_runs.csv'),validation_rows)
            print('Validation results: ',validation[0],flush=True)
        else:print('Set USE_SEARCH_PARAMETERS=True, then run simulate_real.py before plotting.',flush=True)
    print('\nSearch report: ',folder,flush=True)
    print('Set USE_SEARCH_PARAMETERS=True in real_config.py to plot or simulate the selected parameters.',flush=True)
    return dict(report_dir=str(folder),selections=selections)

if __name__=='__main__':
    freeze_support()
    search_results=main()
