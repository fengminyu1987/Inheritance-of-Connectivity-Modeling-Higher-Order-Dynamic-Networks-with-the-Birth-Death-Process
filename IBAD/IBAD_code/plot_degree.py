"""Average independent-run degree counts and plot node and hyperedge distributions."""
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from results_common import load_results, result_order, parameter_label


output_root=Path(__file__).resolve().parent/'results_simple'
filters={
    'deletion_mode':'ibad_single',
    'm_n':[1,2,4,8], 'm_e':[1,2,4,8],


    'lambda_n':[2], 'lambda_e':[2], 'mu_n':[1/80], 'mu_e':[1/80],
    'a_n':1.0, 'a_e':1.0, 'min_history':100,
    'similarity_rule':'history_rbf_raw',
    'similarity_sigma':[1],
}


experiment='m'  # m, birth_rates or lifetimes
paired_parameters=[('lambda_n','lambda_e')]
if experiment=='birth_rates':
    filters.update(m_n=[1],m_e=[1],lambda_n=[5,10,15,20,25],lambda_e=[5,10,15,20,25],
                   mu_n=[1/50],mu_e=[1/50])
elif experiment=='lifetimes':
    filters.update(m_n=[1],m_e=[1],lambda_n=[5],lambda_e=[5],
                   mu_n=[1/t for t in (50,100,150,200,250)],
                   mu_e=[1/t for t in (50,100,150,200,250)])
    paired_parameters.append(('mu_n','mu_e'))
elif experiment!='m':
    raise ValueError('experiment must be m, birth_rates or lifetimes')
expected_runs=3 if experiment=='m' else 10
mu_display='lifetime'


plot_mode='auto'


distribution_source='event_counts'
show_overview=False
show_figures=True
fit_k_min=1
fit_k_max=100


fit_ranges_n={}
fit_ranges_e={}


fit_node_mn=[]
fit_edge_me=[]


fit_enabled_n={}
fit_enabled_e={}
show_titles=False
axis_fontsize=22
tick_fontsize=16
legend_fontsize=16
legend_loc='best'
legend_ncol=1
legend_markerscale=1.6
point_alpha=0.9
xlim=(1,100)
ylim=(1e-5,1)
# Descriptive log-log fitting is enabled for birth-rate/lifetime sweeps only.
draw_fit_lines=True


all_results,file_list,groups=load_results(output_root,filters,paired_parameters,expected_runs)


parameters_list=[data['parameters'] for data in all_results]
d1_list=[data['d1'] for data in all_results]
d2_list=[data['d2'] for data in all_results]
P_node_list=[data['P_node_time'] for data in all_results]
P_edge_list=[data['P_edge_time'] for data in all_results]
number_n_list=[data['number_n'] for data in all_results]
number_e_list=[data['number_e'] for data in all_results]
cluser_list=[data['cluser_'] for data in all_results]
apl_list=[data['apl_'] for data in all_results]
plcc_list=[data['plcc'] for data in all_results]

print(f'Loaded {len(all_results)} independent-run results.')

varying_keys=[key for key in sorted(set().union(*(p.keys() for p in parameters_list))) if key!='seed'
              and len({p.get(key) for p in parameters_list})>1]

group_parameters=[]
group_labels=[]
d1_mean_list=[]
d2_mean_list=[]
run_count_list=[]
P_node_mean_list=[]
P_edge_mean_list=[]
if show_overview:
    fig,axes=plt.subplots(2,2,figsize=(11,8))
for group,runs in groups.items():
    p=dict(group)
    group_parameters.append(p)
    run_count_list.append(len(runs))

    for key,mean_list in [('d1',d1_mean_list),('d2',d2_mean_list)]:
        rows=[data[key][-1] for data in runs]
        if any(not row for row in rows):
            raise ValueError(f'{key} has an empty mean-count row')
        total=np.zeros(max(map(len,rows)),dtype=float)
        for row in rows:
            if not np.all(np.isfinite(row)) or np.any(np.asarray(row)<0):
                raise ValueError(f'{key} has invalid values in the mean-count row')
            total[:len(row)]+=row
        mean_list.append((total/len(rows)).tolist())
    label_parts=[]
    for key in varying_keys:
        if key in ('lambda_e','mu_e') and p.get(key)==p.get(key[:-1]+'n'):
            continue
        label=parameter_label(key,p.get(key),mu_display)
        if key in ('lambda_n','mu_n') and p.get(key)==p.get(key[:-1]+'e'):
            label=label.replace('_n','')
        label_parts.append(label)
    label=', '.join(label_parts) or p['deletion_mode']
    group_labels.append(label)
    for col,(kind,mean_list) in enumerate(zip(('node','edge'),(P_node_mean_list,P_edge_mean_list))):
        rows=[data[f'P_{kind}_time'] for data in runs if data[f'{kind}_nonempty_time']>0]
        if not rows:
            mean_list.append([])
            print(f'{label}: {kind} is empty throughout the observation window.')
            continue
        length=max(map(len,rows))
        P=np.zeros(length)
        for row in rows:
            if not np.isclose(sum(row),1) or not np.all(np.isfinite(row)) or np.any(np.asarray(row)<0):
                raise ValueError(f'{label}: {kind} invalid probability array')
            P[:len(row)]+=row
        P/=len(rows)
        mean_list.append(P.tolist())
        k=np.arange(length)
        if show_overview:
            ax=axes[0,col]
            mask=(k>0)&(P>0)
            ax.loglog(k[mask],P[mask],'o',markersize=3,label=f'{label} (runs={len(rows)})')
            ax.set_xlabel('Node hyperdegree k' if kind=='node' else 'Hyperedge size k')
            ax.set_ylabel('P(k)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2,which='both')
            ax.set_ylim(top=1.1)
            S=np.cumsum(P[::-1])[::-1]
            mask=(k>0)&(S>0)
            axes[1,col].loglog(k[mask],S[mask],'-',label=f'{label} (runs={len(rows)})')
            axes[1,col].set_xlabel(ax.get_xlabel())
            axes[1,col].set_ylabel('Pr(K >= k)')
            axes[1,col].set_ylim(top=1.1)
            axes[1,col].grid(alpha=0.2,which='both')
        mean=float(k@P)
        cv=float(np.sqrt(((k-mean)**2)@P)/mean) if mean>0 else float('nan')
        q95=int(np.searchsorted(np.cumsum(P),0.95))
        print(f'{label}, {kind}: P(0)={P[0]:.6g}, P(1)={P[1] if len(P)>1 else 0:.6g}, mean={mean:.6g}, CV={cv:.6g}, q95={q95}, valid_runs={len(rows)}')

if plot_mode not in ('auto','m','general'):
    raise ValueError('plot_mode must be auto, m or general')
m_mode=(plot_mode=='m' or (plot_mode=='auto' and any(k in varying_keys for k in ('m_n','m_e'))))
scan_name='m_scan' if m_mode else 'scan_'+'_'.join(varying_keys or ['single'])
plot_dir=Path(output_root)/'plots_simple_v2'/scan_name
plot_dir.mkdir(parents=True,exist_ok=True)
if show_overview:
    fig.tight_layout()
    fig.savefig(plot_dir/'degree_distribution_pmf_ccdf.png',dpi=200)


if distribution_source=='event_counts':
    degree_n=[list(row) for row in d1_mean_list]
    degree_e=[list(row) for row in d2_mean_list]
elif distribution_source=='time':
    degree_n=[list(row) for row in P_node_mean_list]
    degree_e=[list(row) for row in P_edge_mean_list]
else:
    raise ValueError('distribution_source must be event_counts or time')
s=[]
for i in range(len(group_parameters)):
    s.append(sum(degree_n[i]))
    for degree in (degree_n,degree_e):
        total=sum(degree[i])
        if total<=0:
            raise ValueError(f'Group {i} has no valid distribution to normalize')
        degree[i]=[value/total for value in degree[i]]


def func(lnx,c,gamma):
    return c-gamma*lnx


def fit_distribution(P,lower,upper,enabled=True):


    if not enabled:
        return {'gamma':None,'c':None,'r2_log':None,'n_points':0,
                'requested_k_min':None,'requested_k_max':None,
                'actual_k_min':None,'actual_k_max':None,
                'probability_mass':None,'status':'not_selected'}
    if lower<1 or (upper is not None and upper<lower):
        raise ValueError('Fit bounds must satisfy lower>=1 and upper>=lower')
    P=np.asarray(P,dtype=float)
    k=np.arange(len(P))
    mask=(k>=lower)&(P>0)
    if upper is not None:
        mask &= k<=upper
    x=k[mask]
    y=P[mask]
    result={'gamma':None,'c':None,'r2_log':None,'n_points':len(x),
            'requested_k_min':lower,'requested_k_max':upper,
            'actual_k_min':int(x[0]) if len(x) else None,
            'actual_k_max':int(x[-1]) if len(x) else None,
            'probability_mass':float(y.sum()),'status':'too_few_points'}
    if len(x)<3:
        return result
    logx=np.log(x)
    logy=np.log(y)

    c,gamma=np.linalg.lstsq(np.column_stack((np.ones(len(x)),-logx)),logy,rcond=None)[0]
    residual=logy-func(logx,c,gamma)
    total=float(np.sum((logy-logy.mean())**2))
    r2=1-float(residual@residual)/total if total>1e-24 else None
    result.update(gamma=float(gamma),c=float(c),r2_log=r2,
                  status='fitted' if gamma>0 else 'nondecreasing_fit')
    return result


gamma_n_list=[]
gamma_e_list=[]
fit_n_list=[]
fit_e_list=[]
gamma_table=[]
for i,p in enumerate(group_parameters):
    row=dict(p)
    row.update(runs=run_count_list[i],distribution_source=distribution_source,
               fit_method='unweighted_loglog_least_squares_on_group_mean',
               tau_n=1/p['mu_n'] if p['mu_n']>0 else float('inf'),
               tau_e=1/p['mu_e'] if p['mu_e']>0 else float('inf'))
    for kind,degree,overrides,enabled_overrides,fit_list,gamma_list in [
            ('n',degree_n,fit_ranges_n,fit_enabled_n,fit_n_list,gamma_n_list),
            ('e',degree_e,fit_ranges_e,fit_enabled_e,fit_e_list,gamma_e_list)]:
        lower,upper=overrides.get((p['m_n'],p['m_e']),(fit_k_min,fit_k_max))
        default_enabled=(not m_mode or
                         (p['m_n'] in fit_node_mn if kind=='n' else p['m_e'] in fit_edge_me))
        enabled=enabled_overrides.get((p['m_n'],p['m_e']),default_enabled)
        fitted=fit_distribution(degree[i],lower,upper,enabled=enabled)
        row[f'fit_selected_{kind}']=bool(enabled)
        fit_list.append(fitted)
        gamma_list.append(fitted['gamma'])
        for key,value in fitted.items():
            row[f'{key}_{kind}']=value
        gamma_text='NA' if fitted['gamma'] is None else f"{fitted['gamma']:.6f}"
        print(f"{group_labels[i]}, {kind}: gamma={gamma_text}, "
              f"k=[{fitted['actual_k_min']},{fitted['actual_k_max']}], "
              f"points={fitted['n_points']}, R2_log={fitted['r2_log']}, status={fitted['status']}")
    gamma_table.append(row)

with (plot_dir/'gamma_fit_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(key for row in gamma_table for key in row)))
    writer.writeheader()
    writer.writerows(gamma_table)
print('Gamma summary: ',plot_dir/'gamma_fit_results.csv')


plot_groups={}
for i,p in enumerate(group_parameters):
    panel=tuple(sorted((key,value) for key,value in p.items() if key!='m_n')) if m_mode else ()
    plot_groups.setdefault(panel,[]).append(i)

plot_groups=list(plot_groups.values())
if m_mode:
    plot_groups.sort(key=lambda indices: result_order({'parameters':dict(group_parameters[indices[0]],m_n=group_parameters[0]['m_n'],seed=0)},filters))


mn_values=list(dict.fromkeys(p['m_n'] for p in group_parameters))
color_=['red','blue','darkviolet','green'] if m_mode else ['red','darkorange','green','blue','purple']
marker_=['1','2','3','4','x']
if m_mode:
    fig5,axes5=plt.subplots(2,len(plot_groups),figsize=(4.2*len(plot_groups),7),squeeze=False)

for panel_number,indices in enumerate(plot_groups):
    p=group_parameters[indices[0]]
    fixed_label=', '.join(parameter_label(key,p.get(key),mu_display) for key in varying_keys if key!='m_n')
    if m_mode and 'm_e' not in varying_keys:
        fixed_label=', '.join(filter(None,[parameter_label('m_e',p['m_e'],mu_display),fixed_label]))
    for row_index,(kind,degree,fit_list) in enumerate([
            ('n',degree_n,fit_n_list),('e',degree_e,fit_e_list)]):
        fig,ax=plt.subplots(figsize=(8,8),dpi=200)
        targets=[ax,axes5[row_index,panel_number]] if m_mode else [ax]
        for j in indices:
            p_j=group_parameters[j]
            style=mn_values.index(p_j['m_n']) if m_mode else j
            P=np.asarray(degree[j])
            k=np.arange(len(P))

            mask=(k>0)&(P>0)
            label=parameter_label('m_n',p_j['m_n'],mu_display) if m_mode else group_labels[j]
            fitted=fit_list[j]
            for target in targets:
                target.loglog(k[1:],np.where(P[1:]>0,P[1:],np.nan),marker=marker_[style%len(marker_)],
                              linestyle='--' if m_mode else 'None',
                              color=color_[style%len(color_)],linewidth=1.2,
                              markerfacecolor='none',markeredgewidth=1.5,
                              markersize=7,alpha=point_alpha,label=label)
                if draw_fit_lines and fitted['gamma'] is not None:

                    xfit=np.geomspace(fitted['actual_k_min'],fitted['actual_k_max'],100)
                    target.loglog(xfit,np.exp(func(np.log(xfit),fitted['c'],fitted['gamma'])),
                                  color=color_[style%len(color_)],linewidth=1.6)
        for target in targets:
            target.set_xlabel(rf'$hk_{kind}$')
            target.set_ylabel(rf'$p(hk_{kind})$')
            target.grid(True,linewidth=0.7,alpha=0.5,axis='y')
            target.set_ylim(top=1.1)
            legend=target.legend(fontsize=legend_fontsize,loc=legend_loc,
                                 ncol=legend_ncol,markerscale=legend_markerscale)
            for line in legend.get_lines():line.set_alpha(1)
            if xlim is not None:target.set_xlim(xlim)
            if ylim is not None:target.set_ylim(ylim)
            if m_mode and show_titles:
                target.set_title(fixed_label)
        ax.tick_params(labelsize=tick_fontsize)
        ax.xaxis.label.set_size(axis_fontsize)
        ax.yaxis.label.set_size(axis_fontsize)
        fig.tight_layout()
        name=(f'degree_distribution_me{p["m_e"]}_panel{panel_number+1:02d}_{kind}'
              if m_mode else f'degree_distribution_{kind}')
        fig.savefig(plot_dir/f'{name}.png',dpi=200)
        fig.savefig(plot_dir/f'{name}.pdf')
        print('Saved: ',plot_dir/f'{name}.png')
        if not show_figures:
            plt.close(fig)
if m_mode:
    fig5.tight_layout()
    fig5.savefig(plot_dir/'figure4_m_comparison.png',dpi=200)
    fig5.savefig(plot_dir/'figure4_m_comparison.pdf')
    print('Saved: ',plot_dir/'figure4_m_comparison.png')
    if not show_figures:
        plt.close(fig5)


if show_figures:
    plt.show()
