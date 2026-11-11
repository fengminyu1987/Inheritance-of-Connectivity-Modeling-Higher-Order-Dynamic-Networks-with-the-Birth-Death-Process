"""Compare deletion mechanisms and similarity measures using saved simulations."""
from pathlib import Path
from datetime import datetime
from itertools import product
from collections import Counter
import hashlib
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


output_root = Path(__file__).resolve().parent / 'results_simple'

file_pattern = 'results.json'
compare_mode = 'both'
include_feasible = False
raw_sigma = 1.0
log_sigma = 1.0


filters = {
    'm_n': [1, 2], 'm_e': [1, 2],
    'lambda_n': [2], 'lambda_e': [2],
    'mu_n': [1/80], 'mu_e': [1/80],
    'a_n': [1.0], 'a_e': [1.0], 'min_history': [100],
    'T_pre': [100000], 'T': [20000], 'sample_every': [400],
}
paired_parameters = [('m_n', 'm_e')]
implementation_revision = 'ibad_v1'
expected_runs = 3
allow_incomplete = False


mechanism_models = [
    dict(id='direct', label='Direct deletion', mode='direct', rule=None, sigma=None),
    dict(id='uniform', label='Uniform inheritance', mode='uniform_single', rule=None, sigma=None),
    dict(id='raw', label='IBAD (raw distance)', mode='ibad_single', rule='history_rbf_raw', sigma=raw_sigma),
]
similarity_models = [
    dict(id='raw', label='Raw distance', mode='ibad_single', rule='history_rbf_raw', sigma=raw_sigma),
    dict(id='pearson_shifted', label='Pearson (r+1)', mode='ibad_single', rule='shifted', sigma=None),
    dict(id='pearson_positive', label='Positive Pearson', mode='ibad_single', rule='positive', sigma=None),
    dict(id='log', label='Log distance', mode='ibad_single', rule='history_rbf', sigma=log_sigma),
]


distribution_source = 'event_counts'

structure_average = 'pooled'
error_type = 'std'
show_figures = True
save_formats = ['png', 'pdf']
dpi = 200
figsize = (8, 6)
axis_fontsize = 23
tick_fontsize = 18
legend_fontsize = 15
legend_ncol = 1
legend_loc = 'best'
legend_anchor = None
legend_markerscale = 1.1
marker_size = 8
point_alpha = 1.0

connect_points = True
sparse_markers = True
marker_count = 26
marker_edgewidth = 1.4
marker_facecolor = 'white'
curve_linewidth = 1.6
curve_alpha = 0.85
curve_linestyles = ['-', '--', '-.', ':', '--', '-.', ':']

model_style_index = {'direct': 0, 'uniform': 1, 'raw': 2, 'log': 3,
                     'pearson_shifted': 4, 'pearson_positive': 5, 'feasible': 6}
show_titles = False
pmf_xlim = {'n': (1, 100), 'e': (1, 100)}
pmf_ylim = {'n': (1e-5, 1), 'e': (1e-5, 1)}
ccdf_xlim = {'n': None, 'e': None}
ccdf_ylim = {'n': (1e-4, 1), 'e': (1e-4, 1)}
metric_ylim = {'C': None, 'PLCC': (0, 1.05), 'L': None, 'density': None}

colors = ['royalblue', 'darkorange', 'crimson', 'seagreen', 'purple', 'saddlebrown', 'black']
markers = ['^', 'o', 's', 'D', 'v', 'P', 'x']


def matches(parameters, selections):
    return all(parameters.get(key) in (value if isinstance(value, (list, tuple)) else [value])
               for key, value in selections.items())


def as_array(values, name, bounded=False):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError(f'{name}: must be a nonempty finite nonnegative one-dimensional list')
    if bounded and np.any(x > 1 + 1e-12):
        raise ValueError(f'{name}: values must be in [0,1]')
    return x


def average_lists(rows):

    total = np.zeros(max(len(row) for row in rows))
    for row in rows:
        total[:len(row)] += row
    return (total / len(rows)).tolist()


def summarize_samples(rows):

    run_means = np.array([np.mean(x) for x in rows])
    x = np.concatenate(rows) if structure_average == 'pooled' else run_means
    return dict(mean=float(x.mean()),
                std=float(x.std(ddof=1)) if len(x) > 1 else None,
                seed_mean=float(run_means.mean()),
                seed_std=float(run_means.std(ddof=1)) if len(run_means) > 1 else None,
                samples=sum(len(row) for row in rows))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finish_figure(fig, ax, name, xlabel, ylabel, limits=None, legend=True):
    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_axisbelow(True)
    ax.grid(axis='y', which='major', alpha=0.22, linewidth=0.7)
    if limits is not None:
        ax.set_ylim(*limits)
    if legend:
        lg = ax.legend(fontsize=legend_fontsize, ncol=legend_ncol, loc=legend_loc,
                       bbox_to_anchor=legend_anchor, markerscale=legend_markerscale,
                       handlelength=2.6, handletextpad=0.65, labelspacing=0.45,
                       framealpha=0.95, edgecolor='0.85', numpoints=1)
        for line in lg.get_lines():
            line.set_alpha(1)
    if show_titles or not panel_complete:
        ax.set_title(panel_title + ('' if panel_complete else ' [INCOMPLETE]'))
    fig.tight_layout()
    for extension in save_formats:
        fig.savefig(plot_dir / f'{name}.{extension}', dpi=dpi, bbox_inches='tight')
    if not show_figures:
        plt.close(fig)

def draw_distribution(ax, k, y, model_id, label, fallback_style=0):

    style = model_style_index.get(model_id, fallback_style)
    color = colors[style % len(colors)]
    marker = markers[style % len(markers)]
    linestyle = curve_linestyles[style % len(curve_linestyles)] if connect_points else 'None'
    valid = (k > 0) & (y > 0) & np.isfinite(y)
    positions = np.flatnonzero(valid)
    if not len(positions):
        return

    positive_k = k > 0
    line_y = np.where(valid, y, np.nan)
    if connect_points:
        ax.loglog(k[positive_k], line_y[positive_k], color=color, linestyle=linestyle,
                  linewidth=curve_linewidth, alpha=curve_alpha, zorder=2)
    marker_positions = positions
    if connect_points and sparse_markers and len(positions) > marker_count:
        targets = np.geomspace(k[positions[0]], k[positions[-1]], max(2, marker_count))
        chosen = positions[np.minimum(np.searchsorted(k[positions], targets), len(positions)-1)]

        breaks = np.flatnonzero(np.diff(positions) > 1)
        marker_positions = np.unique(np.concatenate((
            chosen, positions[:6], positions[breaks], positions[breaks+1], positions[-1:])))
    face = color if marker_facecolor == 'auto' else marker_facecolor
    ax.loglog(k[marker_positions], y[marker_positions], linestyle='None', marker=marker,
              color=color, markersize=marker_size, markerfacecolor=face,
              markeredgewidth=marker_edgewidth, alpha=point_alpha, zorder=3)

    ax.add_line(Line2D([], [], color=color, linestyle=linestyle, linewidth=curve_linewidth,
                       marker=marker, markersize=marker_size, markerfacecolor=face,
                       markeredgewidth=marker_edgewidth, alpha=1, label=label))


if compare_mode not in ('mechanism', 'similarity', 'both'):
    raise ValueError('compare_mode must be mechanism, similarity or both')
if structure_average not in ('pooled', 'seed_mean') or error_type not in ('std', 'none'):
    raise ValueError('Check structure_average and error_type')
if distribution_source not in ('event_counts', 'time'):
    raise ValueError('distribution_source must be event_counts or time')
if not output_root.is_dir():
    raise SystemExit(f'Result directory does not exist: {output_root}')
for key in ('deletion_mode', 'similarity_rule', 'similarity_sigma', 'seed'):
    if key in filters:
        raise ValueError(f'{key} belongs in the model configuration, not in common filters')
models = list(mechanism_models if compare_mode != 'similarity' else similarity_models)
if compare_mode == 'both':
    models += [m for m in similarity_models if m['id'] not in {x['id'] for x in models}]
if include_feasible and compare_mode != 'similarity':
    models.append(dict(id='feasible', label='Feasible inheritance', mode='ibad_feasible',
                       rule='history_rbf_raw', sigma=raw_sigma))
if len({m['id'] for m in models}) != len(models):
    raise ValueError('Model IDs must be unique')
for model in models:
    if model['rule'] in ('history_rbf', 'history_rbf_raw') and not (
            isinstance(model['sigma'], (int, float)) and np.isfinite(model['sigma']) and model['sigma'] > 0):
        raise ValueError('RBF sigma  must be finite and positive')

plot_dir = output_root / 'plots_comparison' / (
    compare_mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
plot_dir.mkdir(parents=True)
groups = {}
seen = {}
source_files = []
inventory = Counter()
counts = Counter()
required = {'m_n', 'm_e', 'lambda_n', 'lambda_e', 'mu_n', 'mu_e', 'a_n', 'a_e',
            'min_history', 'T_pre', 'T', 'sample_every', 'seed', 'deletion_mode'}
for path in sorted(output_root.rglob(file_pattern)):
    counts['files'] += 1
    with path.open(encoding='utf-8-sig') as f:
        data = json.load(f)
    if data.get('schema_version') != 'simple_v2':
        counts['old_schema'] += 1
        continue
    p = dict(data['parameters'])
    revision = data.get('implementation_revision', 'unknown')
    inventory[(p.get('deletion_mode'), p.get('similarity_rule'), p.get('similarity_sigma'), revision)] += 1
    if not matches(p, filters):
        continue
    if any(p.get(left) != p.get(right) for left, right in paired_parameters):
        continue
    if implementation_revision is not None and revision != implementation_revision:
        counts['revision_mismatch'] += 1
        continue
    selected = []
    for model in models:
        if p.get('deletion_mode') != model['mode']:
            continue

        if model['rule'] is not None and p.get('similarity_rule') != model['rule']:
            continue
        if model['rule'] in ('history_rbf', 'history_rbf_raw') and p.get('similarity_sigma') != model['sigma']:
            continue
        selected.append(model)
    if not selected:
        continue
    if len(selected) != 1:
        raise ValueError(f'Multiple model definitions match the same file: {path}')
    if not required.issubset(p) or p.get('seed') is None:
        raise ValueError(f'Missing control parameters or random seed: {path}')
    model = selected[0]

    control = {k: v for k, v in p.items()
               if k not in ('seed', 'deletion_mode', 'similarity_rule', 'similarity_sigma')}
    control['implementation_revision'] = revision
    panel_key = json.dumps(control, sort_keys=True, ensure_ascii=False)
    run = dict(parameters=p, model_id=model['id'], path=str(path), series={})
    for key in ('d1', 'd2'):
        run[key] = as_array(data[key][-1], f'{path}: {key}[-1]').tolist()
    for side in ('node', 'edge'):
        key = f'P_{side}_time'
        if distribution_source == 'time':
            run[key] = as_array(data[key], key, bounded=True).tolist()
            if not np.isclose(sum(run[key]), 1):
                raise ValueError(f'{path}: {key} is not normalized')
    for metric, key in [('C', 'cluser_'), ('PLCC', 'plcc'), ('L', 'L_series'), ('density', 'density_series')]:
        if key not in data or not len(data[key]):
            print(f'Missing {key}; corresponding statistics will be blank: {path}')
            continue
        run['series'][metric] = as_array(data[key], key, metric != 'L')
    if 'C' in run['series'] and 'PLCC' in run['series']:
        if len(run['series']['C']) != len(run['series']['PLCC']):
            raise ValueError(f'{path}: C and PLCC snapshot counts disagree')
    run['stats'] = data.get('inheritance_stats_sampling', {})

    fingerprint = hashlib.sha256(json.dumps(dict(
        d1=run['d1'], d2=run['d2'],
        P_node_time=data.get('P_node_time'), P_edge_time=data.get('P_edge_time'),
        series={k: v.tolist() for k, v in run['series'].items()}, stats=run['stats']),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    identity = (panel_key, model['id'], p['seed'])
    if identity in seen:
        if seen[identity] != fingerprint:
            raise ValueError(f'Conflicting results for the same effective parameters, model and seed: {path}')
        counts['duplicates'] += 1
        continue
    seen[identity] = fingerprint
    groups.setdefault(panel_key, {}).setdefault(model['id'], []).append(run)
    source_files.append(dict(path=str(path), parameters=p, implementation_revision=revision, model_id=model['id']))
    counts['matched'] += 1
    del data

print('Loading summary: ', dict(counts))
print('Available mechanisms, similarities, bandwidths, implementations and file counts:')
for key, value in inventory.items():
    print(key, value)


coverage = []
filter_keys = list(filters)
choices = [v if isinstance(v, (list, tuple)) else [v] for v in filters.values()]
requested = [dict(zip(filter_keys, values)) for values in product(*choices)]
for p in requested:
    if any(p.get(left) != p.get(right) for left, right in paired_parameters):
        continue
    if not any(matches(json.loads(panel), {k: [v] for k, v in p.items()}) for panel in groups):
        coverage.append(dict(p, model_id='ALL', runs=0, status='missing_parameter_group'))

def panel_order(panel):
    p = json.loads(panel)
    return tuple(list(v if isinstance(v, (list, tuple)) else [v]).index(p[k])
                 for k, v in filters.items()) + (panel,)

for panel in sorted(groups, key=panel_order):
    for model in models:
        runs = groups[panel].get(model['id'], [])
        runs.sort(key=lambda r: r['parameters']['seed'])
        status = ('missing_model' if not runs else 'ok' if expected_runs is None or len(runs) == expected_runs
                  else 'run_count_mismatch')
        coverage.append(dict(json.loads(panel), model_id=model['id'], runs=len(runs), status=status))
write_csv(plot_dir / 'coverage.csv', coverage)
problems = [row for row in coverage if row['status'] != 'ok']
for row in problems:
    print('Incomplete experiment: ', row)
if not groups:
    raise SystemExit(f'No matching results; check model settings, filters and implementation. Coverage: {plot_dir}')
if problems and not allow_incomplete:
    raise SystemExit(f'Incomplete experiments; see {plot_dir}/coverage.csv. Set allow_incomplete=True to preview available results.')


group_parameters, group_labels, run_count_list = [], [], []
d1_mean_list, d2_mean_list, degree_n, degree_e = [], [], [], []
cluser_mean_list, plcc_mean_list, L_mean_list, density_mean_list = [], [], [], []
cluser_std_list, plcc_std_list = [], []
run_summary, group_summary, plot_data = [], [], []
panel_indices = {}
summary_lookup = {}
for panel in sorted(groups, key=panel_order):
    control = json.loads(panel)
    panel_indices[panel] = []
    for model in models:
        runs = groups[panel].get(model['id'], [])
        if not runs:
            continue
        i = len(group_parameters)
        panel_indices[panel].append(i)
        p_group = dict(control, model_id=model['id'], deletion_mode=model['mode'],
                       similarity_rule=model['rule'], similarity_sigma=model['sigma'])
        group_parameters.append(p_group)
        group_labels.append(model['label'])
        run_count_list.append(len(runs))
        row = dict(p_group, label=model['label'], runs=len(runs),
                   structure_average=structure_average, distribution_source=distribution_source)
        for key, raw_list, probability_list, side in [('d1', d1_mean_list, degree_n, 'node'),
                                                      ('d2', d2_mean_list, degree_e, 'edge')]:
            mean_counts = average_lists([r[key] for r in runs])
            raw_list.append(mean_counts)
            if distribution_source == 'event_counts':
                if sum(mean_counts) <= 0:
                    raise ValueError(f'{p_group}: All mean counts are zero; no degree distribution is defined')
                probability = (np.array(mean_counts) / sum(mean_counts)).tolist()
            else:
                probability = average_lists([r[f'P_{side}_time'] for r in runs])
            probability_list.append(probability)
            k = np.arange(len(probability))
            mean_degree = float(k @ probability)
            row[f'{side}_mean_degree'] = mean_degree
            row[f'{side}_P0'] = probability[0]
            row[f'{side}_CV'] = (float(np.sqrt(((k - mean_degree)**2) @ probability) / mean_degree)
                                if mean_degree > 0 else None)
        for metric, mean_list in [('C', cluser_mean_list), ('PLCC', plcc_mean_list),
                                  ('L', L_mean_list), ('density', density_mean_list)]:
            samples = [r['series'][metric] for r in runs if metric in r['series']]
            if len(samples) != len(runs):
                stat = dict(mean=None, std=None, seed_mean=None, seed_std=None,
                            samples=sum(len(x) for x in samples))
            else:
                stat = summarize_samples(samples)
            row.update({f'{metric}_{key}': value for key, value in stat.items()})
            row[f'{metric}_runs_with_data'] = len(samples)
            mean_list.append(stat['mean'])
            if metric == 'C':
                cluser_std_list.append(stat['std'])
            elif metric == 'PLCC':
                plcc_std_list.append(stat['std'])

        for side in ('node', 'edge'):
            for item in ('success', 'duplicate', 'no_candidate'):
                values, total_num, total_den = [], 0, 0
                for r in runs:
                    stat = r['stats']
                    den, num = stat.get(side + '_lost'), stat.get(side + '_' + item)
                    if model['mode'] == 'direct' or den is None or num is None or den <= 0:
                        continue
                    values.append(num / den)
                    total_num += num
                    total_den += den
                prefix = f'{side}_{item}_rate'
                row[prefix] = float(np.mean(values)) if values else None
                row[prefix + '_seed_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                row[prefix + '_runs'] = len(values)
                row[prefix + '_pooled'] = total_num / total_den if total_den else None
        group_summary.append(row)
        summary_lookup[i] = row
        for r in runs:
            run_row = dict(p_group, seed=r['parameters']['seed'], source_file=r['path'])
            for metric, values in r['series'].items():
                run_row[metric + '_mean'] = float(values.mean())
                run_row[metric + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else None
                run_row[metric + '_samples'] = len(values)
            run_row.update({'count_' + key: value for key, value in r['stats'].items()})
            run_summary.append(run_row)
        print(model['label'], 'mn=', control['m_n'], 'me=', control['m_e'],
              'runs=', len(runs), 'C=', row['C_mean'], 'PLCC=', row['PLCC_mean'])
write_csv(plot_dir / 'run_summary.csv', run_summary)
write_csv(plot_dir / 'group_summary.csv', group_summary)


for panel_number, (panel, indices) in enumerate(panel_indices.items(), start=1):
    control = json.loads(panel)
    prefix = f'panel{panel_number:02d}_mn{control["m_n"]}_me{control["m_e"]}'
    panel_complete = all(row['status'] == 'ok' for row in coverage
                         if row.get('implementation_revision') == control['implementation_revision']
                         and all(row.get(k) == v for k, v in control.items()))
    panel_title = rf'$m_n={control["m_n"]},\ m_e={control["m_e"]}$'
    for side, distributions in [('n', degree_n), ('e', degree_e)]:
        for kind in ('PMF', 'CCDF'):
            fig, ax = plt.subplots(figsize=figsize)
            for i in indices:
                p = np.asarray(distributions[i])
                k = np.arange(len(p))
                y = p if kind == 'PMF' else np.cumsum(p[::-1])[::-1]
                style = next(j for j, m in enumerate(models) if m['id'] == group_parameters[i]['model_id'])
                draw_distribution(ax, k, y, group_parameters[i]['model_id'], group_labels[i], style)
                plot_data.append(dict(panel=panel_number, metric=kind + '_' + side,
                                      parameters=group_parameters[i], label=group_labels[i],
                                      x=k.tolist(), y=y.tolist()))
            xlim = (pmf_xlim if kind == 'PMF' else ccdf_xlim)[side]
            if xlim is not None:
                if min(xlim) <= 0:
                    raise ValueError('Logarithmic axis bounds must be positive')
                ax.set_xlim(*xlim)
            ylim = (pmf_ylim if kind == 'PMF' else ccdf_ylim)[side]
            if ylim is not None and min(ylim) <= 0:
                raise ValueError('Logarithmic axis bounds must be positive')
            finish_figure(fig, ax, prefix + '_' + kind + '_' + side, rf'$hk_{side}$',
                          rf'$p(hk_{side})$' if kind == 'PMF' else rf'$\Pr(K_{side}\geq k)$', ylim)

    for metric in ('C', 'PLCC', 'L', 'density'):
        fig, ax = plt.subplots(figsize=figsize)
        for style, model in enumerate(models):
            found = [i for i in indices if group_parameters[i]['model_id'] == model['id']]
            row = summary_lookup[found[0]] if found else {}
            mean, std = row.get(metric + '_mean'), row.get(metric + '_std')
            if mean is None:
                ax.text(style, 0.04, 'missing', ha='center', transform=ax.get_xaxis_transform())
                continue
            ax.plot(style, mean, marker=markers[style % len(markers)], markersize=marker_size + 2,
                    color=colors[style % len(colors)], markerfacecolor='white', linestyle='None')
            if error_type == 'std' and std is not None:
                ax.errorbar(style, mean, yerr=std, fmt='none', capsize=5,
                            ecolor=colors[style % len(colors)])
            plot_data.append(dict(panel=panel_number, metric=metric, parameters=group_parameters[found[0]],
                                  label=model['label'], mean=mean, std=std,
                                  error_shown=std if error_type == 'std' else None))
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m['label'] for m in models], rotation=20, ha='right')
        ax.set_xlim(-0.5, len(models) - 0.5)
        ylabel = {'C': '$C$', 'PLCC': r'$R_{\mathrm{LCC}}$', 'L': '$L$',
                  'density': r'$L/(|N||E|)$'}[metric]
        finish_figure(fig, ax, prefix + '_' + metric, '', ylabel, metric_ylim[metric], legend=False)

    for side in ('node', 'edge'):
        fig, ax = plt.subplots(figsize=figsize)
        active = [m for m in models if m['mode'] != 'direct']
        for j, item in enumerate(('success', 'duplicate', 'no_candidate')):
            label_added = False
            for x, model in enumerate(active):
                found = [i for i in indices if group_parameters[i]['model_id'] == model['id']]
                row = summary_lookup[found[0]] if found else {}
                rate = row.get(f'{side}_{item}_rate')
                if rate is not None:
                    ax.bar(x + (j-1)*0.24, rate, width=0.24, color=colors[j],
                           label=item.replace('_', ' ') if not label_added else None)
                    label_added = True
        ax.set_xticks(range(len(active)))
        ax.set_xticklabels([m['label'] for m in active], rotation=20, ha='right')
        ax.set_xlim(-0.5, max(0.5, len(active)-0.5))
        finish_figure(fig, ax, prefix + '_inheritance_' + side, '', side + ' inheritance fraction', (0, 1.05),
                      legend=bool(ax.get_legend_handles_labels()[0]))

with (plot_dir / 'plot_data.json').open('w', encoding='utf-8') as f:
    json.dump(dict(group_parameters=group_parameters, group_labels=group_labels,
                   run_count_list=run_count_list, degree_n=degree_n, degree_e=degree_e,
                   d1_mean_list=d1_mean_list, d2_mean_list=d2_mean_list,
                   cluser_mean_list=cluser_mean_list, plcc_mean_list=plcc_mean_list,
                   L_mean_list=L_mean_list, density_mean_list=density_mean_list,
                   plot_data=plot_data), f, ensure_ascii=False, allow_nan=False)
with (plot_dir / 'config_and_sources.json').open('w', encoding='utf-8') as f:
    json.dump(dict(compare_mode=compare_mode, models=models, filters=filters,
                   paired_parameters=paired_parameters, expected_runs=expected_runs,
                   implementation_revision=implementation_revision,
                   allow_incomplete=allow_incomplete, structure_average=structure_average,
                   distribution_source=distribution_source, error_type=error_type,
                   uncertainty='descriptive SD; pooled snapshots are not independent repetitions',
                   inheritance_rate_average='equal run rates; direct and zero denominator are undefined',
                   inheritance_candidate_note='positive Pearson also changes eligibility; shifted uses undefined r=0',
                   history_note='min_history is eligibility threshold; full common event history is used',
                   plot_style=dict(connect_points=connect_points, sparse_markers=sparse_markers,
                                   marker_count=marker_count, marker_size=marker_size,
                                   marker_edgewidth=marker_edgewidth, marker_facecolor=marker_facecolor,
                                   point_alpha=point_alpha, curve_linewidth=curve_linewidth,
                                   curve_alpha=curve_alpha, curve_linestyles=curve_linestyles,
                                   model_style_index=model_style_index, colors=colors, markers=markers,
                                   note='Full original positive data connected; zero bins break lines; no smoothing.'),
                   source_files=source_files), f, ensure_ascii=False, indent=2, allow_nan=False)
print('Plots and lists saved to: ', plot_dir)
if show_figures:
    plt.show()
