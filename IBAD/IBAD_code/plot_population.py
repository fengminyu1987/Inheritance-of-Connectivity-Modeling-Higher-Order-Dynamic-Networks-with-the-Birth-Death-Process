"""Plot population trajectories and time-weighted stationary population distributions."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


ROOT = Path(__file__).resolve().parent
output_root = ROOT / 'results_simple'
save_dir = ROOT / 'fig2_population'


cases = [(2, 40), (4, 40), (4, 80)]
filters = {
    'deletion_mode': 'ibad_single',
    'similarity_rule': 'history_rbf_raw',
    'similarity_sigma': 1.0,
    'm_n': 1,
    'm_e': 1,


}
expected_runs = 10
axis_mode = 'event'
plot_points = 1500
colors = ['orange', 'green', 'blue', 'darkviolet', 'red']
figsize = (10, 10)
label_size = 19
tick_size = 15
legend_size = 14
panel_size = 21
line_width = 1.4
marker_size = 3.6
legend_marker_scale = 1.5
band_alpha = 0.25
fill_alpha = 0.25
band_sd = 2
xlim_time = None
ylim_population = None
xlim_population = None
ylim_probability = None
show_window = True


def equal(x, y):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return bool(np.isclose(x, y, rtol=1e-10, atol=1e-12))
    return x == y


def population_probability(data, side):

    name = f'number_{side}_time_distribution'
    saved = data.get(name)
    if saved:
        indices = np.array([int(k) for k in saved])
        if (indices < 0).any():
            raise ValueError(f'{name} contains negative counts.')
        p = np.zeros(int(indices.max()) + 1)
        for k, v in saved.items():
            p[int(k)] = float(v)
    else:


        times = np.asarray(data['t_population'], dtype=float)
        counts = np.asarray(data[f'number_{side}'], dtype=int)
        pre = int(data['parameters']['T_pre'])
        steps = int(data['parameters']['T'])
        if pre < 1 or len(times) != pre + steps or len(counts) != len(times):
            raise ValueError(f'Missing {name}; the saved trajectory is insufficient for reconstruction.')
        if not np.isclose(times[pre-1], data['time_start']):
            raise ValueError('Burn-in endpoint does not match time_start.')
        dt = np.diff(times[pre-1:])
        if (dt <= 0).any() or not np.isclose(dt.sum(), data['duration']):
            raise ValueError('Inconsistent observation duration or event times.')
        p = np.bincount(counts[pre-1:-1], weights=dt) / dt.sum()
    if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError(f'{name} is not a valid population probability distribution.')
    return p / p.sum()


def load_population():
    if not output_root.is_dir():
        raise FileNotFoundError(f'Cannot find {output_root}. Place the script next to results_simple.')
    groups = [[] for _ in cases]
    signatures = [set() for _ in cases]
    seen = set()
    available = set()
    for path in sorted(output_root.rglob('results.json')):
        with path.open(encoding='utf-8') as f:
            data = json.load(f)
        p = data.get('parameters', {})
        if not all(any(equal(p.get(k), choice) for choice in
                       (v if isinstance(v, (list, tuple)) else [v])) for k, v in filters.items()):
            continue
        if not all(k in p for k in ('lambda_n', 'lambda_e', 'mu_n', 'mu_e')):
            continue
        if p['mu_n'] <= 0 or p['mu_e'] <= 0:
            continue
        if not equal(p['lambda_n'], p['lambda_e']) or not equal(p['mu_n'], p['mu_e']):
            continue
        available.add((p['lambda_n'], round(1/p['mu_n'], 10)))
        for i, (lam, tau) in enumerate(cases):
            if not equal(p['lambda_n'], lam) or not equal(p['mu_n'], 1/tau):
                continue
            if p.get('seed') is None:
                raise ValueError(f'Missing random seed: {path}')
            identity = (i, p['seed'])
            if identity in seen:
                raise ValueError(f'Duplicate seed for the same parameters: seed={p["seed"]}: {path}\n'
                                 'Remove duplicate files or restrict filters to one experiment.')
            seen.add(identity)

            signature = dict(p)
            signature.pop('seed')
            signature['implementation_revision'] = data.get('implementation_revision')
            signatures[i].add(json.dumps(signature, sort_keys=True))
            t = np.asarray(data['t_population'], dtype=float)
            n = np.asarray(data['number_n'], dtype=float)
            e = np.asarray(data['number_e'], dtype=float)
            if (t.ndim != 1 or len(t) < 2 or n.shape != t.shape or e.shape != t.shape
                    or not np.isfinite(t).all() or not np.isfinite(n).all()
                    or not np.isfinite(e).all() or (np.diff(t) <= 0).any()
                    or (t <= 0).any() or (n < 0).any() or (e < 0).any()
                    or (n != np.floor(n)).any() or (e != np.floor(e)).any()):
                raise ValueError(f'Invalid population trajectory or physical time: {path}')
            if len(t) != int(p['T_pre']) + int(p['T']):
                raise ValueError(f'Population records do not cover burn-in and observation: {path}')
            groups[i].append({'time': t, 'number': [n, e],
                              'probability': [population_probability(data, s) for s in ('n', 'e')],
                              'path': str(path), 'parameters': p,
                              'time_start': data['time_start'], 'duration': data['duration']})
    for i, group in enumerate(groups):
        if not group:
            raise ValueError(f'Missing results for (lambda,tau)={cases[i]}.\n'
                             f'Available cases under these filters: {sorted(available)}\n'
                             'Check cases and filters against the available results.')
        if len(signatures[i]) > 1:
            raise ValueError(f'{cases[i]} includes different settings or implementations; restrict filters.')
        if expected_runs is not None and len(group) != expected_runs:
            raise ValueError(f'{cases[i]}: found {len(group)} runs; expected {expected_runs}.')
        print(f'(lambda,tau)={cases[i]}: {len(group)} runs')
        if len(group) == 1:
            print('  One run: plotting its trajectory without a between-run SD band.')
    return groups


def main():
    if axis_mode not in ('physical', 'event'):
        raise ValueError("axis_mode must be physical or event")
    if not cases or len(set(cases)) != len(cases) or any(l <= 0 or t <= 0 for l, t in cases):
        raise ValueError('cases must be nonempty and distinct, with positive lambda and tau.')
    groups = load_population()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                         'font.size': tick_size, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(left=.10, right=.985, bottom=.10, top=.985, wspace=.25, hspace=.48)
    rows, plot_data = [], []
    for c, ((lam, tau), group) in enumerate(zip(cases, groups)):
        label = rf'$\lambda={lam:g},\ \tau={tau:g}$'
        color = colors[c % len(colors)]
        if axis_mode == 'physical':

            start = max(run['time'][0] for run in group)
            stop = min(run['time'][-1] for run in group)
            if start >= stop:
                raise ValueError('Runs have no common physical-time interval.')
            x = np.geomspace(start, stop, plot_points)
            indices = [np.clip(np.searchsorted(run['time'], x, side='right')-1,
                               0, len(run['time'])-1) for run in group]
        else:
            stop = min(len(run['time']) for run in group)
            x = np.unique(np.geomspace(1, stop, plot_points).astype(int))
            indices = [x-1 for run in group]
        entry = {'lambda': lam, 'tau': tau, 'axis_mode': axis_mode, 'x': x.tolist(),
                 'source_files': [run['path'] for run in group],
                 'parameters': [run['parameters'] for run in group],
                 'observation_windows': [{'time_start': r['time_start'], 'duration': r['duration']}
                                         for r in group]}
        for a, symbol in enumerate(('N', 'M')):
            values = np.asarray([r['number'][a][idx] for r, idx in zip(group, indices)])
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1) if len(group) > 1 else np.zeros_like(mean)
            ax = axs[0, a]
            ax.plot(x, mean, color=color, lw=line_width, label=label)
            if len(group) > 1:
                ax.fill_between(x, mean-band_sd*sd, mean+band_sd*sd,
                                color=color, alpha=band_alpha, linewidth=0)
            length = max(len(r['probability'][a]) for r in group)
            p = sum(np.pad(r['probability'][a], (0, length-len(r['probability'][a])))
                    for r in group) / len(group)
            k = np.arange(len(p))

            k_theory = np.arange(max(length, int(poisson.ppf(1-1e-8, lam*tau))+1))
            ax = axs[1, a]
            ax.fill_between(k_theory, poisson.pmf(k_theory, lam*tau),
                            color=color, alpha=fill_alpha, linewidth=0)
            keep = p > 0
            ax.plot(k[keep], p[keep], 'o', color=color, ms=marker_size,
                    markeredgewidth=0, label=label)
            observed_mean = np.dot(k, p)
            rows.append({'lambda': lam, 'tau': tau, 'mu': 1/tau,
                         'population': ['node', 'hyperedge'][a],
                         'theoretical_mean': lam*tau, 'observed_mean': observed_mean,
                         'theoretical_variance': lam*tau,
                         'observed_variance': np.dot((k-observed_mean)**2, p)})
            entry[symbol] = {'mean': mean.tolist(), 'std': sd.tolist(), 'probability': p.tolist()}
        plot_data.append(entry)

    for a, symbol in enumerate(('N', 'M')):
        ax = axs[0, a]
        ax.set_xscale('log')
        ax.set_xlabel(r'$t$' if axis_mode == 'physical' else r'$T$', fontsize=label_size)
        ax.set_ylabel(rf'${symbol}$', fontsize=label_size)
        ax.grid(True, axis='y', alpha=.55, linewidth=.8)
        ax.legend(fontsize=legend_size, loc='upper left')
        if xlim_time is not None: ax.set_xlim(xlim_time)
        if ylim_population is not None: ax.set_ylim(ylim_population)
        ax = axs[1, a]
        ax.set_xlabel(rf'${symbol}$', fontsize=label_size)
        ax.set_ylabel(rf'$P({symbol})$', fontsize=label_size)
        ax.grid(True, axis='x', alpha=.55, linewidth=.8)
        ax.legend(fontsize=legend_size, loc='upper right', markerscale=legend_marker_scale)
        if xlim_population is not None: ax.set_xlim(xlim_population)
        if ylim_probability is not None: ax.set_ylim(ylim_probability)
    for i, ax in enumerate(axs.flat):
        ax.tick_params(labelsize=tick_size)
        ax.set_axisbelow(True)
        ax.text(.5, -.25, f'({chr(97+i)})', transform=ax.transAxes, ha='center', va='top',
                fontsize=panel_size, fontweight='bold')
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / 'F2_new.pdf', bbox_inches='tight')
    fig.savefig(save_dir / 'F2_new.png', dpi=300, bbox_inches='tight')
    with (save_dir / 'population_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (save_dir / 'population_plot_data.json').open('w', encoding='utf-8') as f:
        json.dump(plot_data, f, ensure_ascii=False, allow_nan=False)
    print(f'Saved: {save_dir}')
    print(f'Top row: {axis_mode} axis, run mean +/- {band_sd} SD.')
    print('Bottom row: equal-run mean of time-weighted population PMFs; shading shows the Poisson prediction.')
    if show_window: plt.show()
    plt.close(fig)
    return plot_data


if __name__ == '__main__':
    main()
