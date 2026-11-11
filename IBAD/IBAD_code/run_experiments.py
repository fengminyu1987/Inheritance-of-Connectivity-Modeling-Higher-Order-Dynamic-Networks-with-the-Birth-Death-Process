"""Run parameter sweeps using the original procedural simulation script."""
from pathlib import Path
from itertools import product
import argparse
import json
import runpy

ROOT = Path(__file__).resolve().parent
EXPERIMENT = 'm'  # population, birth_rates, lifetimes, m, clustering or mechanisms
T_PRE = 100000
T = 20000
SAMPLE_EVERY = 400
SEEDS = list(range(281176436, 281176446))
RUN_COUNTS = dict(population=10, birth_rates=10, lifetimes=10, m=3, clustering=10, mechanisms=3)
OUTPUT_ROOT = ROOT / 'results_simple'
SKIP_EXISTING = True


def parameter_grid(experiment):
    common = dict(m_n=1, m_e=1, lambda_n=2, lambda_e=2, mu_n=1/80, mu_e=1/80,
                  a_n=1.0, a_e=1.0, min_history=100, similarity_rule='history_rbf_raw',
                  similarity_sigma=1.0, deletion_mode='ibad_single',
                  T_pre=T_PRE, T=T, sample_every=SAMPLE_EVERY)
    settings = []
    if experiment == 'population':
        settings = [dict(lambda_n=lam, lambda_e=lam, mu_n=1/tau, mu_e=1/tau)
                    for lam, tau in [(2,40), (4,40), (4,80)]]
    elif experiment == 'birth_rates':
        settings = [dict(lambda_n=lam, lambda_e=lam, mu_n=1/50, mu_e=1/50)
                    for lam in (5,10,15,20,25)]
    elif experiment == 'lifetimes':
        settings = [dict(lambda_n=5, lambda_e=5, mu_n=1/tau, mu_e=1/tau)
                    for tau in (50,100,150,200,250)]
    elif experiment == 'm':
        settings = [dict(m_n=mn, m_e=me) for mn, me in product((1,2,4,8), repeat=2)]
    elif experiment == 'clustering':
        settings = [dict(m_n=m, m_e=m, lambda_n=ln, lambda_e=le)
                    for m, ln, le in product((1,2), (4,6,8), (2,4,6,8,10))]
    elif experiment == 'mechanisms':
        models = [('direct','history_rbf_raw'), ('uniform_single','history_rbf_raw'),
                  ('ibad_single','history_rbf_raw'), ('ibad_single','shifted'),
                  ('ibad_single','positive'), ('ibad_single','history_rbf')]
        settings = [dict(m_n=m, m_e=m, deletion_mode=mode, similarity_rule=rule)
                    for m in (1,2) for mode, rule in models]
    else:
        raise ValueError('Unknown experiment: '+experiment)
    for setting in settings:
        p = dict(common, **setting)
        for seed in SEEDS[:RUN_COUNTS[experiment]]:
            yield dict(p, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=list(RUN_COUNTS), default=EXPERIMENT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if len(set(SEEDS)) != len(SEEDS) or len(SEEDS) < RUN_COUNTS[args.experiment]:
        raise ValueError('Provide enough distinct seeds for the requested run count')
    existing = []
    if SKIP_EXISTING:
        for file in OUTPUT_ROOT.rglob('results.json'):
            data = json.loads(file.read_text(encoding='utf-8'))
            if data.get('implementation_revision') == 'ibad_v1':
                existing.append(data['parameters'])
    for index, p in enumerate(parameter_grid(args.experiment), 1):
        effective = dict(p)
        if p['deletion_mode'] not in ('ibad_single','ibad_feasible') or not p['similarity_rule'].startswith('history_rbf'):
            effective['similarity_sigma'] = None
        if any(all(q.get(k)==v for k,v in effective.items()) for q in existing):
            print('Already saved:', p)
            continue
        print(index, p, flush=True)
        if not args.dry_run:
            runpy.run_path(str(ROOT/'hyper_test1_simple_history.py'), run_name='__main__',
                           init_globals={'run_parameters':dict(p, output_root=OUTPUT_ROOT)})


if __name__ == '__main__':
    main()
