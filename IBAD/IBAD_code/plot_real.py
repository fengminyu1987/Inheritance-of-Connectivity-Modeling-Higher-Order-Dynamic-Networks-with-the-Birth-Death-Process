"""Compare empirical networks with independent simulated runs."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import real_config as cfg


MATCH_CURRENT_CONFIG = True
EXTRA_FILTERS = {}
EXPECTED_RUNS = len(cfg.SEEDS)
ALLOW_INCOMPLETE = False
DISTRIBUTION_SOURCE = 'event_counts'
# Empirical tables report the mean and sample SD of independent-run means.
STRUCTURE_AVERAGE = 'seed_mean'
SHOW_FIGURES = True
SAVE_FORMATS = ['png', 'pdf']
DPI = 200
AXIS_FONTSIZE = 15
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
POINT_ALPHA = 0.9
LEGEND_LOC = 'best'
LEGEND_NCOL = 1
PMF_XLIM = None
PMF_YLIM = (1e-3,1)
CCDF_YLIM = (1e-4,1)
KL_EPSILON = 1e-10
FIT_ENABLED = False
DRAW_FIT_LINES = False

FIT_RANGES = {name:{side:{'real':(2,None),'simulation':(2,None)} for side in ('n','e')}
              for name in ('PPN','FW','EN')}

CONNECT_PMF_POINTS = True
LINE_WIDTH = 1.0
POINT_SIZE = 6.0
POINT_EDGE_WIDTH = 1.1
SINGLE_FIGSIZE = (4.8, 3.8)
AUTO_VISIBLE_XLIM = True
MAKE_COMPARISON_GRID = True
GRID_NETWORKS = ('PPN', 'FW', 'EN')
GRID_FIGSIZE = (7.4, 2.75)
GRID_AXIS_FONTSIZE = 10
GRID_TICK_FONTSIZE = 8.5
GRID_LEGEND_FONTSIZE = 10
GRID_POINT_SIZE = 3.8
GRID_LINE_WIDTH = 0.8


def draw_degree_panel(ax, observed, simulated, side, mode='PMF', compact=False):

    ylim = PMF_YLIM if mode == 'PMF' else CCDF_YLIM
    visible_degrees = []
    for distribution, color, marker, label, style in (
            (observed, 'royalblue', 'o', 'Real', '--'),
            (simulated, 'crimson', '^', 'Simulation', '-')):
        y = np.asarray(distribution, dtype=float)
        if y.ndim != 1 or np.any(~np.isfinite(y)) or np.any(y < 0):
            raise ValueError('Degree distribution must be a finite nonnegative one-dimensional list')
        if mode == 'CCDF':
            y = np.cumsum(y[::-1])[::-1]
        k = np.arange(len(y))

        y_plot = np.where((k > 0) & (y > 0), y, np.nan)
        linestyle = style if mode == 'CCDF' or CONNECT_PMF_POINTS else 'None'
        ax.loglog(k[1:], y_plot[1:], linestyle=linestyle, marker=marker,
                  color=color, markerfacecolor='white' if label == 'Real' else color,
                  markersize=GRID_POINT_SIZE if compact else POINT_SIZE,
                  markeredgewidth=POINT_EDGE_WIDTH,
                  linewidth=GRID_LINE_WIDTH if compact else LINE_WIDTH,
                  alpha=POINT_ALPHA, label=label, zorder=3 if label == 'Real' else 2)
        visible = (k > 0) & (y > 0)
        if ylim is not None:
            visible &= (y >= ylim[0]) & (y <= ylim[1])
        visible_degrees.extend(k[visible].tolist())
    if PMF_XLIM is not None:
        ax.set_xlim(*PMF_XLIM)
    elif AUTO_VISIBLE_XLIM and visible_degrees:
        ax.set_xlim(0.85, max(2.0, max(visible_degrees) * 1.18))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(rf'$hk_{side}$', fontsize=GRID_AXIS_FONTSIZE if compact else AXIS_FONTSIZE)
    ax.set_ylabel(rf'$p(hk_{side})$' if mode == 'PMF' else r'$\Pr(K\geq k)$',
                  fontsize=GRID_AXIS_FONTSIZE if compact else AXIS_FONTSIZE)
    ax.tick_params(axis='both', which='major', direction='in',
                   labelsize=GRID_TICK_FONTSIZE if compact else TICK_FONTSIZE,
                   length=3.0 if compact else 4.0, width=0.7)
    ax.tick_params(axis='both', which='minor', direction='in', length=1.8, width=0.5)
    ax.grid(axis='y', which='major', linewidth=0.5, alpha=0.18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def plot_comparison_grid(plot_lists, plot_dir):

    data = plot_lists
    parameters = data['group_parameters']
    networks = [name for name in GRID_NETWORKS if any(p['dataset']==name for p in parameters)]
    if not networks:return None
    indices=[]
    for name in networks:
        candidates=[i for i,p in enumerate(parameters) if p['dataset']==name]
        if len(candidates)!=1:
            print('Skipping combined plot: ',name,'has multiple parameter groups; restrict the filters.')
            return None
        indices.append(candidates[0])
    ncols=len(networks);plot_dir=Path(plot_dir);plot_dir.mkdir(parents=True,exist_ok=True)
    for side, filename in [('n','real_comparison_grid'),('e','real_hyperedge_grid')]:
        fig,axes=plt.subplots(1,ncols,squeeze=False,
                             figsize=(GRID_FIGSIZE[0]*ncols/3,GRID_FIGSIZE[1]))
        for col,(name,index) in enumerate(zip(networks,indices)):
            ax=axes[0,col]
            draw_degree_panel(ax,data['real_degree_'+side][index],data['degree_'+side][index],side,compact=True)
            ax.text(.5,-.31,'('+chr(97+col)+')',ha='center',va='top',transform=ax.transAxes,fontsize=10)
            if col>0:ax.set_ylabel('')
            ax.set_title(name,fontsize=11,pad=6)
            legend=ax.legend(loc=LEGEND_LOC,frameon=False,fontsize=GRID_LEGEND_FONTSIZE,
                             handlelength=1.8,markerscale=1.2)
            for line in legend.get_lines():line.set_alpha(1)
        fig.subplots_adjust(left=.085 if ncols==3 else .15,right=.985,bottom=.29,top=.86,wspace=.24)
        for fmt in SAVE_FORMATS:
            fig.savefig((plot_dir/filename).with_suffix('.'+fmt),dpi=DPI,bbox_inches='tight',pad_inches=.04)
        if not SHOW_FIGURES:plt.close(fig)
    (plot_dir/'plot_lists_display.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    return plot_dir/'real_comparison_grid'


def finish(fig,path):
    fig.tight_layout()
    for fmt in SAVE_FORMATS:
        fig.savefig(path.with_suffix('.'+fmt),dpi=DPI,bbox_inches='tight')
    if not SHOW_FIGURES:
        plt.close(fig)


def summarize(rows):
    run_means=np.array([np.mean(row) for row in rows])
    samples=np.concatenate(rows) if STRUCTURE_AVERAGE=='pooled' else run_means
    return dict(mean=float(np.mean(samples)),std=float(np.std(samples,ddof=1)) if len(samples)>1 else None,
                run_mean_std=float(np.std(run_means,ddof=1)) if len(run_means)>1 else None,
                samples=sum(map(len,rows)),runs=len(rows))


def main():
    from simulate_real import parameters_for, real_identity
    from real_utils import mean_lists, probability, distribution_distances, descriptive_fit, save_json, write_csv
    if DISTRIBUTION_SOURCE not in ('event_counts','time') or STRUCTURE_AVERAGE not in ('pooled','seed_mean'):
        raise ValueError('Invalid averaging configuration')
    plot_dir=cfg.RESULT_DIR/'plots'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    plot_dir.mkdir(parents=True,exist_ok=True)
    real_networks={name:json.loads((cfg.REAL_DIR/f'{name}.json').read_text(encoding='utf-8')) for name in cfg.NETWORKS}
    groups,seen,coverage,sources={},{},[],[]
    for path in sorted(cfg.RESULT_DIR.rglob('results.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version')!='real_simulation_v1' or data.get('dataset') not in real_networks:
            continue
        name=data['dataset'];p=data['parameters'];real=real_networks[name]
        if data['real_signature']!=real_identity(real):
            print('Skipping a different empirical preprocessing version: ',path)
            continue
        if MATCH_CURRENT_CONFIG:
            wanted=parameters_for(real)
            if any(p.get(k)!=v for k,v in wanted.items()) or p.get('seed') not in cfg.SEEDS:
                continue
        if any(p.get(k) not in (v if isinstance(v,(tuple,list)) else [v]) for k,v in EXTRA_FILTERS.items()):
            continue
        control={k:v for k,v in p.items() if k!='seed'}
        key=json.dumps(dict(dataset=name,real_signature=data['real_signature'],parameters=control,
                            implementation_revision=data.get('implementation_revision')),sort_keys=True)
        identity=(key,p['seed'])
        fingerprint=hashlib.sha256(json.dumps({k:data[k] for k in (
            'node_counts_mean','edge_counts_mean','P_node_time','P_edge_time','topology_samples')},sort_keys=True).encode()).hexdigest()
        if identity in seen:
            if seen[identity]!=fingerprint:
                raise ValueError('Conflicting results for the same configuration and seed: '+str(path))
            print('Skipping duplicate: ',path)
            continue
        seen[identity]=fingerprint
        groups.setdefault(key,[]).append(data)
        sources.append(str(path))
    for name in cfg.NETWORKS:
        if not any(json.loads(k)['dataset']==name for k in groups):
            coverage.append(dict(dataset=name,status='missing_network',runs=0))
    for key,runs in groups.items():
        status='ok' if EXPECTED_RUNS is None or len(runs)==EXPECTED_RUNS else 'run_count_mismatch'
        coverage.append(dict(dataset=runs[0]['dataset'],group=hashlib.sha256(key.encode()).hexdigest()[:10],
                             status=status,runs=len(runs)))
    write_csv(plot_dir/'coverage.csv',coverage)
    bad=[row for row in coverage if row['status']!='ok']
    if bad:
        print('Incomplete coverage: ',bad)
        if not ALLOW_INCOMPLETE:
            raise SystemExit('Incomplete results; see coverage.csv or set ALLOW_INCOMPLETE=True for a preview: '+str(plot_dir))
    if not groups:
        raise SystemExit('No matching results; check real_config, preprocessing and results_real')
    degree_n,degree_e,real_degree_n,real_degree_e=[],[],[],[]
    group_parameters,group_labels,topology_rows,divergence_rows,fit_rows,run_rows,diagnostic_rows=[],[],[],[],[],[],[]
    colors=['royalblue','darkorange','seagreen','purple','brown']

    ordered_groups=sorted(groups.items(),key=lambda item:(cfg.NETWORKS.index(json.loads(item[0])['dataset']),item[0]))
    for group_no,(key,runs) in enumerate(ordered_groups,start=1):
        runs.sort(key=lambda r:r['parameters']['seed'])
        name=runs[0]['dataset'];p={k:v for k,v in runs[0]['parameters'].items() if k!='seed'}
        real=real_networks[name]
        p['nodes_axis']=real['preprocessing'].get('nodes_axis')
        label=p['deletion_mode']+' / '+p['similarity_rule']
        group_parameters.append(dict(dataset=name,**p));group_labels.append(label)
        prefix=f'{name}_group{group_no:02d}'
        complete=EXPECTED_RUNS is None or len(runs)==EXPECTED_RUNS
        averages={}
        for side,field,output_list,real_list in [('n','node',degree_n,real_degree_n),('e','edge',degree_e,real_degree_e)]:
            if DISTRIBUTION_SOURCE=='event_counts':
                simulated=probability(mean_lists([r[field+'_counts_mean'] for r in runs]))
            else:
                probabilities=[r['P_'+field+'_time'] for r in runs]
                if any(not np.isclose(sum(x),1) for x in probabilities):
                    raise ValueError('Cannot average empty or unnormalized PMFs')
                simulated=mean_lists(probabilities)
            observed=real['P_'+field]
            output_list.append(simulated);real_list.append(observed)
            averages[side]=simulated
            distances=distribution_distances(observed,simulated,KL_EPSILON)
            divergence_rows.append(dict(dataset=name,group=group_no,basis='distance_to_averaged_distribution',
                                        side=side,runs=len(runs),**distances))

            seed_distances=[]
            for r in runs:
                q=probability(r[field+'_counts_mean']) if DISTRIBUTION_SOURCE=='event_counts' else r['P_'+field+'_time']
                max_length=max(len(q),len(observed))
                pr=np.pad(observed,(0,max_length-len(observed))).tolist()
                qr=np.pad(q,(0,max_length-len(q))).tolist()
                d=distribution_distances(pr,qr,KL_EPSILON)
                seed_distances.append(d)
                divergence_rows.append(dict(dataset=name,group=group_no,basis='individual_run',side=side,
                                            seed=r['parameters']['seed'],**d))
            for metric in ('JS','KL_smoothed'):
                vals=[x[metric] for x in seed_distances if x[metric] is not None]
                divergence_rows.append(dict(dataset=name,group=group_no,basis='mean_of_run_distances',side=side,
                    metric=metric,mean=float(np.mean(vals)) if vals else None,
                    std=float(np.std(vals,ddof=1)) if len(vals)>1 else None,log_base='e',smoothing_epsilon=KL_EPSILON))
            fits={}
            for kind,distribution in [('real',observed),('simulation',simulated)]:
                fit=descriptive_fit(distribution,FIT_RANGES[name][side][kind]) if FIT_ENABLED else {'gamma':None}
                fits[kind]=fit
                fit_rows.append(dict(dataset=name,group=group_no,side=side,kind=kind,
                                     method='descriptive_loglog_OLS' if FIT_ENABLED else 'disabled',**fit))
            for mode in ('PMF','CCDF'):
                fig,ax=plt.subplots(figsize=SINGLE_FIGSIZE)
                draw_degree_panel(ax,observed,simulated,side,mode)
                for kind,color in [('real','royalblue'),('simulation','crimson')]:
                    fit=fits[kind]
                    if mode=='PMF' and DRAW_FIT_LINES and fit.get('gamma') is not None:
                        x=np.geomspace(fit['k_min'],fit['k_max'],100)
                        ax.loglog(x,np.exp(fit['c'])*x**(-fit['gamma']),color=color,linestyle=':')
                legend=ax.legend(fontsize=LEGEND_FONTSIZE,loc=LEGEND_LOC,ncol=LEGEND_NCOL,
                                 markerscale=1.2,frameon=False,handlelength=2.0)
                for line in legend.get_lines():line.set_alpha(1)
                ax.set_title(name+('' if complete else ' [INCOMPLETE]'),fontsize=AXIS_FONTSIZE)
                finish(fig,plot_dir/(prefix+'_'+mode+'_'+side))

        for metric,observed in real['metrics'].items():
            samples=[]
            for r in runs:
                values=[sample[metric] for sample in r['topology_samples'] if sample.get(metric) is not None]
                if values:
                    samples.append(np.asarray(values,dtype=float))
            stat=summarize(samples) if len(samples)==len(runs) else dict(mean=None,std=None,run_mean_std=None,samples=0,runs=len(samples))
            topology_rows.append(dict(dataset=name,group=group_no,metric=metric,real=observed,
                                      structure_average=STRUCTURE_AVERAGE,**stat))
        for r in runs:
            for metric in ('N','E','L','C','PLCC','APL_LCC','N_triadic','density'):
                values=[s[metric] for s in r['topology_samples'] if s.get(metric) is not None]
                run_rows.append(dict(dataset=name,group=group_no,seed=r['parameters']['seed'],metric=metric,
                                     mean=float(np.mean(values)) if values else None,samples=len(values)))
            diagnostic_rows.append(dict(dataset=name,group=group_no,seed=r['parameters']['seed'],
                warmup_time=r['warmup_time'],observation_duration=r['observation_duration'],
                N_target=r['stationary_population_targets']['N'],E_target=r['stationary_population_targets']['E'],
                N_time_mean=r['physical_time_means']['N'],E_time_mean=r['physical_time_means']['E'],
                **r['initialization_survivors_after_warmup'],
                **{metric+'_'+half:value for metric,halves in r['half_window_diagnostic'].items() for half,value in halves.items()}))
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        for ax,metric in zip(axes.flat,('C','PLCC','APL_LCC','N_triadic','N','E','L','density')):
            row=next(x for x in topology_rows if x['group']==group_no and x['metric']==metric)
            for x,(kind,value) in enumerate([('Real',row['real']),('Simulation',row['mean'])]):
                if value is not None:
                    ax.plot(x,value,'o',color='royalblue' if x==0 else 'crimson',markersize=7)
                else:
                    ax.text(x,.1,'not computed',ha='center',transform=ax.get_xaxis_transform())
            if row['mean'] is not None and row['std'] is not None:
                ax.errorbar(1,row['mean'],yerr=row['std'],fmt='none',ecolor='crimson',capsize=4)
            ax.set_xticks([0,1],['Real','Simulation']);ax.set_xlim(-.5,1.5)
            ax.set_title(metric);ax.grid(axis='y',alpha=.3)
        fig.suptitle(name+('' if complete else ' [INCOMPLETE]'))
        finish(fig,plot_dir/(prefix+'_topology'))

        fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        for run_index,r in enumerate(runs):
            color=colors[run_index%len(colors)]
            for ax,metric,key_series in zip(axes,('N','E','L'),('number_n','number_e','L_series')):
                trace=r['warmup_trace'];series=r[key_series];step=max(1,len(series)//500)
                x=[v['event'] for v in trace]+[r['parameters']['T_pre']+i+1 for i in range(0,len(series),step)]
                y=[v[metric] for v in trace]+[series[i] for i in range(0,len(series),step)]
                ax.plot(x,y,color=color,linewidth=1,alpha=.7,label=f"seed={r['parameters']['seed']}")
                ax.set_ylabel(metric);ax.grid(alpha=.25)
        for ax,metric in zip(axes,('N','E','L')):
            ax.axvline(p['T_pre'],color='black',linestyle='--',label='Observation starts')
            if metric in ('N','E'):
                ax.axhline(runs[0]['stationary_population_targets'][metric],color='grey',linestyle=':')
        axes[0].legend(fontsize=9,ncol=2);axes[-1].set_xlabel('Event index')
        fig.suptitle(name+' diagnostics (not a stationarity certificate)')
        finish(fig,plot_dir/(prefix+'_diagnostics'))
    write_csv(plot_dir/'topology_comparison.csv',topology_rows)
    write_csv(plot_dir/'distribution_distances.csv',divergence_rows)
    write_csv(plot_dir/'descriptive_gamma.csv',fit_rows)
    write_csv(plot_dir/'run_topology_means.csv',run_rows)
    write_csv(plot_dir/'stationarity_diagnostics.csv',diagnostic_rows)
    save_json(plot_dir/'plot_lists.json',dict(group_parameters=group_parameters,group_labels=group_labels,
        degree_n=degree_n,degree_e=degree_e,real_degree_n=real_degree_n,real_degree_e=real_degree_e))
    if MAKE_COMPARISON_GRID:
        plot_comparison_grid(dict(group_parameters=group_parameters,degree_n=degree_n,degree_e=degree_e,
            real_degree_n=real_degree_n,real_degree_e=real_degree_e),plot_dir)
    save_json(plot_dir/'plot_settings.json',dict(source_files=sources,distribution_source=DISTRIBUTION_SOURCE,
        structure_average=STRUCTURE_AVERAGE,fit_enabled=FIT_ENABLED,fit_ranges=FIT_RANGES,
        PMF_xlim=PMF_XLIM,PMF_ylim=PMF_YLIM,connect_PMF_points=CONNECT_PMF_POINTS,
        auto_visible_xlim=AUTO_VISIBLE_XLIM,comparison_grid=MAKE_COMPARISON_GRID,
        KL_epsilon=KL_EPSILON,KL_direction='real || simulated',divergence_log_base='e',
        JS_definition='divergence, not square-root distance',sample_dependence='SD describes spread, not CI'))
    print('Comparison plots and lists: ',plot_dir)
    if SHOW_FIGURES:plt.show()
    return dict(plot_dir=plot_dir,degree_n=degree_n,degree_e=degree_e,
                real_degree_n=real_degree_n,real_degree_e=real_degree_e,
                group_parameters=group_parameters,topology_rows=topology_rows)

if __name__=='__main__':

    plot_results=main()
    degree_n=plot_results['degree_n'];degree_e=plot_results['degree_e']
    real_degree_n=plot_results['real_degree_n'];real_degree_e=plot_results['real_degree_e']
    group_parameters=plot_results['group_parameters']
