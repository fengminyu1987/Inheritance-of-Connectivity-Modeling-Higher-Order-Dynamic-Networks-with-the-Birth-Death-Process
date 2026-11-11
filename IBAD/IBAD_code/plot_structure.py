"""Plot clustering and largest-component statistics across parameter sweeps."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from results_common import load_results, parameter_label


output_root=Path(__file__).resolve().parent/'results_simple'
filters={
    'deletion_mode':'ibad_single',
    'm_n':[1,2,4,8], 'm_e':[1,2,4,8],
    'lambda_n':[2], 'lambda_e':[2], 'mu_n':[1/80], 'mu_e':[1/80],
    'a_n':1.0, 'a_e':1.0, 'min_history':100,
    'similarity_rule':'history_rbf_raw', 'similarity_sigma':[1],
}
paired_parameters=[]
expected_runs=3


x_parameter='m_e'
line_parameter='m_n'
experiment='m'  # m or birth_rates
if experiment=='birth_rates':
    filters.update(m_n=[1,2],m_e=[1,2],lambda_n=[4,6,8],lambda_e=[2,4,6,8,10])
    expected_runs=10
    paired_parameters=[('m_n','m_e')]
    x_parameter='lambda_e'
    line_parameter='lambda_n'
elif experiment!='m':
    raise ValueError('experiment must be m or birth_rates')

mu_display='lifetime'
error_type='std'
# pooled: all recorded snapshots; seed_mean: independent-run averages. SD is descriptive.
structure_average='pooled'
axis_fontsize=22
tick_fontsize=16
legend_fontsize=16
legend_loc='best'
legend_ncol=2
ylim_C=None
ylim_PLCC=(0,1.05)
show_titles=False
shade_plcc=False
show_figures=True

if x_parameter==line_parameter:
    raise ValueError('x_parameter and line_parameter must differ')
if error_type not in ('std','sem','none'):
    raise ValueError('error_type must be std, sem or none')

if structure_average not in ('pooled','seed_mean'):
    raise ValueError('structure_average must be pooled or seed_mean')
if structure_average=='pooled' and error_type=='sem':
    raise ValueError('SEM requires independent run means; use structure_average=seed_mean')

all_results,file_list,groups=load_results(output_root,filters,paired_parameters,expected_runs)
group_parameters=[dict(group) for group in groups]
if any(x_parameter not in p or line_parameter not in p for p in group_parameters):
    raise ValueError('Missing x_parameter or line_parameter in saved parameters')
varying_keys=[key for key in group_parameters[0]
              if len({p.get(key) for p in group_parameters})>1]


cluser_run_mean_list=[]
plcc_run_mean_list=[]
cluser_mean_list=[]
plcc_mean_list=[]
cluser_std_list=[]
plcc_std_list=[]
cluser_sem_list=[]
plcc_sem_list=[]
run_count_list=[]
summary_rows=[]
run_rows=[]

for group,runs in groups.items():
    p=dict(group)
    C_runs=[]
    plcc_runs=[]
    sample_counts=[]
    C_samples=[]
    plcc_samples=[]
    for data in runs:
        C=np.asarray(data.get('cluser_',[]),dtype=float)
        plcc=np.asarray(data.get('plcc',[]),dtype=float)
        seed=data['parameters']['seed']
        if C.ndim!=1 or plcc.ndim!=1 or len(C)==0 or len(C)!=len(plcc):
            raise ValueError(f'seed={seed}: C/PLCC samples are empty, non-vector or unequal in length')
        if not np.all(np.isfinite(C)) or not np.all(np.isfinite(plcc)):
            raise ValueError(f'seed={seed}: Nonfinite C/PLCC samples')
        if np.any(C<0) or np.any(C>1+1e-12) or np.any(plcc<0) or np.any(plcc>1+1e-12):
            raise ValueError(f'seed={seed}: C/PLCC values must be fractions in [0,1]')
        C_samples.append(C)
        plcc_samples.append(plcc)
        C_runs.append(float(C.mean()))
        plcc_runs.append(float(plcc.mean()))
        sample_counts.append(len(C))
        run_rows.append(dict(p,seed=seed,n_snapshots=len(C),C_run_mean=C_runs[-1],
                             PLCC_run_mean=plcc_runs[-1],average_basis='sampled_post_event_snapshots'))
    count=len(runs)
    cluser_run_mean_list.append(C_runs)
    plcc_run_mean_list.append(plcc_runs)
    run_count_list.append(count)

    C_values=np.concatenate(C_samples) if structure_average=='pooled' else np.asarray(C_runs)
    plcc_values=np.concatenate(plcc_samples) if structure_average=='pooled' else np.asarray(plcc_runs)
    cluser_mean_list.append(float(np.mean(C_values)))
    plcc_mean_list.append(float(np.mean(plcc_values)))
    cluser_std_list.append(float(np.std(C_values,ddof=1)) if len(C_values)>1 else float('nan'))
    plcc_std_list.append(float(np.std(plcc_values,ddof=1)) if len(plcc_values)>1 else float('nan'))
    cluser_sem_list.append(float(np.std(C_runs,ddof=1)/np.sqrt(count)) if count>1 else float('nan'))
    plcc_sem_list.append(float(np.std(plcc_runs,ddof=1)/np.sqrt(count)) if count>1 else float('nan'))
    summary_rows.append(dict(p,runs=count,C_mean=cluser_mean_list[-1],PLCC_mean=plcc_mean_list[-1],
                            C_std=cluser_std_list[-1],PLCC_std=plcc_std_list[-1],
                            C_sem=cluser_sem_list[-1],PLCC_sem=plcc_sem_list[-1],
                            min_snapshots=min(sample_counts),max_snapshots=max(sample_counts),
                            std_ddof=1,average_basis=structure_average,
                            C_run_mean_std=float(np.std(C_runs,ddof=1)) if count>1 else None,
                            PLCC_run_mean_std=float(np.std(plcc_runs,ddof=1)) if count>1 else None))
    print(', '.join(parameter_label(key,p[key],mu_display) for key in varying_keys)
          +f': C={cluser_mean_list[-1]:.6g}, PLCC={plcc_mean_list[-1]:.6g}, runs={count}')
    if count==1 and structure_average=='seed_mean':
        print('  One seed: no between-run SD or SEM is available.')

plot_dir=Path(output_root)/'plots_structure'/f'x_{x_parameter}_line_{line_parameter}'
plot_dir.mkdir(parents=True,exist_ok=True)
for name,rows in [('structure_summary.csv',summary_rows),('structure_run_means.csv',run_rows)]:

    fieldnames=list(dict.fromkeys(key for row in rows for key in row))
    with (plot_dir/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:('' if isinstance(value,float) and not np.isfinite(value) else value)
                             for key,value in row.items()})


panels={}

moving_parameters={x_parameter,line_parameter}
for _ in range(len(paired_parameters)+1):
    for left,right in paired_parameters:
        if left in moving_parameters or right in moving_parameters:
            moving_parameters.update((left,right))
for i,p in enumerate(group_parameters):
    fixed=tuple(sorted((key,value) for key,value in p.items()
                       if key not in moving_parameters))
    panels.setdefault(fixed,[]).append(i)

color_=['blue','green','darkorange','red','purple']
marker_=['^','o','s','d','x']
line_values=list(dict.fromkeys(p[line_parameter] for p in group_parameters))
if line_parameter in filters:
    values=filters[line_parameter]
    line_values=list(values) if isinstance(values,(list,tuple)) else [values]

if x_parameter in filters:
    values=filters[x_parameter]
    x_values=list(values) if isinstance(values,(list,tuple)) else [values]
else:
    x_values=sorted({p[x_parameter] for p in group_parameters})
if not all(isinstance(value,(int,float)) and np.isfinite(value) for value in x_values):
    raise ValueError('The x axis requires numeric parameters')
if x_parameter in ('mu_n','mu_e') and mu_display=='lifetime':
    if any(value<=0 for value in x_values):
        raise ValueError('Lifetime axes require positive death rates')
    x=[1/value for value in x_values]
    x_label=rf'$\tau_{x_parameter[-1]}$'
else:
    x=list(x_values)
    symbols={'m_n':r'$m_n$','m_e':r'$m_e$','lambda_n':r'$\lambda_n$',
             'lambda_e':r'$\lambda_e$','mu_n':r'$\mu_n$','mu_e':r'$\mu_e$'}
    x_label=symbols.get(x_parameter,x_parameter)

plot_data=[]
for panel_no,(fixed,indices) in enumerate(panels.items(),start=1):
    fixed=dict(fixed)
    title=', '.join(parameter_label(key,fixed[key],mu_display) for key in varying_keys if key in fixed)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    fig_C,ax_C=plt.subplots(figsize=(8,8),dpi=200)
    fig_plcc,ax_plcc=plt.subplots(figsize=(8,8),dpi=200)
    for line_value in line_values:
        selected=[i for i in indices if group_parameters[i][line_parameter]==line_value]
        if not selected:
            continue
        by_x={group_parameters[i][x_parameter]:i for i in selected}
        if len(by_x)!=len(selected):
            raise ValueError('Duplicate x values within one curve; check grouping parameters')
        missing=[value for value in x_values if value not in by_x]
        if missing:
            print(f'Panel {panel_no}, {line_parameter}={line_value}: missing {x_parameter}={missing}.')
        style=line_values.index(line_value)
        label=parameter_label(line_parameter,line_value,mu_display)
        for kind,means,stds,sems,targets in [
                ('C',cluser_mean_list,cluser_std_list,cluser_sem_list,[axes[0],ax_C]),
                ('PLCC',plcc_mean_list,plcc_std_list,plcc_sem_list,[axes[1],ax_plcc])]:
            y=[means[by_x[value]] if value in by_x else float('nan') for value in x_values]
            errors=(stds if error_type=='std' else sems) if error_type!='none' else None
            error=[errors[by_x[value]] if value in by_x else float('nan') for value in x_values] if errors is not None else None
            plot_data.append({'panel':panel_no,'fixed_parameters':fixed,'metric':kind,
                              'line_value':line_value,'label':label,'x':list(x),'y':list(y),
                              'error':None if error is None else list(error),
                              'group_indices':[by_x.get(value) for value in x_values]})
            for ax in targets:
                ax.plot(x,y,marker=marker_[style%len(marker_)],linewidth=2,markersize=9,
                        markerfacecolor='white',markeredgewidth=1.8,color=color_[style%len(color_)],label=label)
                if error is not None:
                    valid=np.isfinite(y)&np.isfinite(error)
                    if np.any(valid):
                        ax.errorbar(np.asarray(x)[valid],np.asarray(y)[valid],yerr=np.asarray(error)[valid],
                                    fmt='none',ecolor=color_[style%len(color_)],capsize=4)
                    if kind=='PLCC' and shade_plcc:
                        ax.fill_between(x,np.asarray(y)-error,np.asarray(y)+error,
                                        color=color_[style%len(color_)],alpha=0.12)
    for kind,targets in [('C',[axes[0],ax_C]),('PLCC',[axes[1],ax_plcc])]:
        for ax in targets:
            ax.set_xlabel(x_label,fontsize=axis_fontsize)
            ax.set_ylabel('$C$' if kind=='C' else r'$R_{\mathrm{LCC}}$',fontsize=axis_fontsize)
            ax.set_xticks(x)
            if x_parameter in ('mu_n','mu_e') and mu_display=='fraction':
                labels=[parameter_label(x_parameter,value,'fraction').replace(
                    rf'\mu_{x_parameter[-1]}','').replace('=','') for value in x_values]
                ax.set_xticklabels(labels)
            ax.tick_params(labelsize=tick_fontsize)
            ax.grid(True,linewidth=0.7,alpha=0.6,axis='y')
            ax.legend(fontsize=legend_fontsize,loc=legend_loc,ncol=legend_ncol)
            if show_titles:ax.set_title(title)

            limits=ylim_C if kind=='C' else ylim_PLCC
            if limits is not None:ax.set_ylim(limits)
    for name,figure in [('C',fig_C),('PLCC',fig_plcc),('C_PLCC',fig)]:
        figure.tight_layout()
        figure.savefig(plot_dir/f'{name}_panel{panel_no:02d}.png',dpi=200)
        figure.savefig(plot_dir/f'{name}_panel{panel_no:02d}.pdf')
        if not show_figures:
            plt.close(figure)
    print(f'Saved C/PLCC plots for panel {panel_no}: {plot_dir}')

if show_figures:
    plt.show()
