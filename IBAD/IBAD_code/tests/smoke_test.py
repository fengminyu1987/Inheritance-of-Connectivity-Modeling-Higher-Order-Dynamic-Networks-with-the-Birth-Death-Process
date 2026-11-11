"""Exercise short simulations, empirical preparation, calibration and all plot readers.
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]


def execute(path, settings):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets)==1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in settings:
                node.value = ast.parse(repr(settings[target.id]), mode='eval').body
    ast.fix_missing_locations(tree)
    namespace = {'__name__':'__main__', '__file__':str(path)}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def main():
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='ibad_smoke_') as directory:
        work = Path(directory)
        for path in ROOT.glob('*.py'):
            shutil.copy2(path, work/path.name)
        shutil.copytree(ROOT/'data', work/'data')
        sys.path.insert(0, str(work))
        import real_config as cfg
        import simulate_real as sim
        import prepare_real
        import plot_real
        import search_m
        from real_utils import read_network, topology, make_graph
        cfg.T_PRE=80;cfg.T=60;cfg.SAMPLE_EVERY=20;cfg.TRACE_EVERY=80
        cfg.MIN_HISTORY=3;cfg.SEEDS=[11,12]
        cfg.USE_SEARCH_PARAMETERS=False
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_real.prepare()
        for name, expected in [('PPN',(666,131)),('FW',(244,248)),('EN',(374,470))]:
            real=json.loads((cfg.REAL_DIR/f'{name}.json').read_text())
            assert (real['metrics']['N'],real['metrics']['E'])==expected
            p=sim.parameters_for(real)
            for seed in cfg.SEEDS:
                with contextlib.redirect_stdout(io.StringIO()):
                    result=sim.run_one(real,p,seed)
                G,E=make_graph(result['final_nodes'],result['final_hyperedges'])
                assert topology(G,E)==result['final_metrics']
        assert abs(json.loads((cfg.REAL_DIR/'FW.json').read_text())['metrics']['C']-.2563)<.00006
        plot_real.SHOW_FIGURES=False;plot_real.SAVE_FORMATS=['png'];plot_real.EXPECTED_RUNS=2
        with contextlib.redirect_stdout(io.StringIO()):
            real_plots=plot_real.main()
        assert len(real_plots['degree_n'])==3
        assert (real_plots['plot_dir']/'real_comparison_grid.png').exists()
        for name, shape in cfg.INITIAL_SIZES.items():
            real=json.loads((cfg.REAL_DIR/f'{name}.json').read_text())
            sim.initialize(real,sim.parameters_for(real),11)
            assert (len(sim.model.G),len(sim.model.E))==shape
        print('Empirical conversion, initialization, metrics and plotting: OK')
        # Short calibration and application of its saved selection.
        search_m.SEARCH_NETWORKS=['FW'];search_m.M_N_BOUNDS=(1,2);search_m.M_E_BOUNDS=(1,2)
        search_m.COARSE_VALUES=[1,2];search_m.LOCAL_CENTERS=1;search_m.LOCAL_RADIUS=0
        search_m.FINALISTS=1;search_m.SCREEN_T=30;search_m.AUTO_WARMUP_FLOOR=False
        search_m.SEARCH_SEEDS=[21];search_m.VALIDATION_SEEDS=cfg.SEEDS
        with contextlib.redirect_stdout(io.StringIO()):
            search_m.main()
        cfg.USE_SEARCH_PARAMETERS=True;cfg.NETWORKS=['FW']
        with contextlib.redirect_stdout(io.StringIO()):
            selected_plots=plot_real.main()
        assert selected_plots['group_parameters'][0]['m_n'] in (1,2)
        cfg.USE_SEARCH_PARAMETERS=False
        print('Calibration, saved selection and validation reader: OK')
        common=dict(lambda_n=2,lambda_e=2,mu_n=1/80,mu_e=1/80,a_n=1.,a_e=1.,
                    min_history=3,similarity_sigma=1.,T_pre=200,T=120,sample_every=20,
                    output_root=work/'results_simple')
        variants=[('direct','history_rbf_raw'),('uniform_single','history_rbf_raw'),
                  ('ibad_single','history_rbf_raw'),('ibad_single','shifted'),
                  ('ibad_single','positive'),('ibad_single','history_rbf')]
        def simulate(options):
            with contextlib.redirect_stdout(io.StringIO()):
                ns=runpy.run_path(str(work/'hyper_test1_simple_history.py'),run_name='__main__',
                                 init_globals={'run_parameters':dict(common,**options)})
            r=ns['results']
            assert len(r['d1'])==121 and len(r['cluser_'])==6
            assert np.allclose(r['d1'][-1],np.mean(r['d1'][:-1],axis=0))
            return r
        for mn,me in [(1,1),(1,2),(2,1),(2,2)]:
            choices=variants if mn==me else [('ibad_single','history_rbf_raw')]
            for mode,rule in choices:
                for seed in (11,12):
                    simulate(dict(m_n=mn,m_e=me,seed=seed,deletion_mode=mode,similarity_rule=rule))
        filters=dict(m_n=[1,2],m_e=[1,2],lambda_n=[2],lambda_e=[2],mu_n=[1/80],mu_e=[1/80],
                     a_n=1.,a_e=1.,min_history=3,similarity_sigma=[1.],
                     similarity_rule='history_rbf_raw',deletion_mode='ibad_single')
        with contextlib.redirect_stdout(io.StringIO()):
            degree=execute(work/'plot_degree.py',dict(filters=filters,expected_runs=2,show_figures=False))
            structure=execute(work/'plot_structure.py',dict(filters=filters,expected_runs=2,show_figures=False))
        assert len(degree['degree_n'])==4 and all(v is None for v in degree['gamma_n_list'])
        for key,runs in structure['groups'].items():
            i=list(structure['groups']).index(key)
            x=np.concatenate([r['cluser_'] for r in runs])
            assert np.isclose(structure['cluser_mean_list'][i],x.mean())
            assert np.isclose(structure['cluser_std_list'][i],x.std(ddof=1))
        controls={k:v for k,v in filters.items() if k not in ('similarity_sigma','similarity_rule','deletion_mode')}
        controls.update(T_pre=[200],T=[120],sample_every=[20])
        with contextlib.redirect_stdout(io.StringIO()):
            compare=execute(work/'plot_mechanism_comparison.py',dict(filters=controls,expected_runs=2,
                            show_figures=False,save_formats=['png'],compare_mode='both'))
        assert len(compare['degree_n'])==12
        for seed in (11,12):
            simulate(dict(m_n=1,m_e=1,seed=seed,deletion_mode='ibad_single',similarity_rule='history_rbf_raw',
                          lambda_n=4,lambda_e=4))
        with contextlib.redirect_stdout(io.StringIO()):
            execute(work/'plot_population.py',dict(cases=[(2,80),(4,80)],expected_runs=2,show_window=False,
                                                  filters=dict(min_history=3,m_n=1,m_e=1,
                                                               deletion_mode='ibad_single',similarity_rule='history_rbf_raw')))
        print('Synthetic runs, grouped PMFs, pooled SD, mechanism and population plots: OK')
        assert list((work/'results_simple').rglob('*.pdf'))
    print('ALL SMOKE CHECKS PASSED')


if __name__=='__main__':
    main()
