"""Load, filter and group independent simulation results."""
import json
from pathlib import Path
from fractions import Fraction
import numpy as np

def parameter_label(key,value,mu_display='lifetime'):

    if key in ('m_n','m_e'):
        return rf'$m_{key[-1]}={value}$'
    if key in ('lambda_n','lambda_e'):
        return rf'$\lambda_{key[-1]}={value:g}$'
    if key not in ('mu_n','mu_e') or value is None:
        return f'{key}={value}'
    side=key[-1]
    if mu_display=='lifetime':
        lifetime=r'\infty' if value==0 else f'{1/value:.10g}'
        return rf'$\tau_{side}={lifetime}$'
    if mu_display=='fraction':
        fraction=Fraction(str(value)).limit_denominator(1000000)

        relation='=' if np.isclose(float(fraction),value,rtol=1e-12,atol=0) else r'\approx '
        number=(str(fraction.numerator) if fraction.denominator==1 else
                f'{fraction.numerator}/{fraction.denominator}')
        return rf'$\mu_{side}{relation}{number}$'
    if mu_display=='rate':
        return rf'$\mu_{side}={value:.6g}$'
    raise ValueError("mu_display must be lifetime, fraction or rate")


def result_order(data,filters):
    p=data['parameters']
    positions=[]
    for key,value in filters.items():
        choices=value if isinstance(value,(list,tuple)) else [value]
        positions.append(choices.index(p.get(key)))
    other=json.dumps({k:v for k,v in p.items() if k not in filters and k!='seed'},
                     sort_keys=True,ensure_ascii=False)
    return tuple(positions)+(other,p['seed'])


def load_results(output_root,filters,paired_parameters=(),expected_runs=3):


    all_results=[]
    file_list=[]
    seen=set()
    legacy_files=0
    if not Path(output_root).is_dir():
        raise SystemExit(f'Result directory does not exist: {output_root}')
    for path in sorted(Path(output_root).rglob('results.json')):
        with path.open(encoding='utf-8') as f:
            data=json.load(f)
        if data.get('schema_version')!='simple_v2':
            legacy_files+=1
            continue
        p=data['parameters']

        p=dict(p,implementation_revision=data.get('implementation_revision','unknown'))
        p.setdefault('similarity_sigma',None)
        data['parameters']=p
        matched=True
        for key,value in filters.items():
            choices=value if isinstance(value,(list,tuple)) else [value]
            if p.get(key) not in choices:
                matched=False
                break
        if not matched:
            continue
        if any(p[left]!=p[right] for left,right in paired_parameters):
            continue
        if p.get('seed') is None:
            raise ValueError(f'Missing random seed: {path}')

        identity=tuple(sorted(p.items()))
        if identity in seen:
            print('Skipping duplicate parameter/seed file: ',path)
            continue
        seen.add(identity)
        all_results.append(data)
        file_list.append(str(path))
    if not all_results:
        raise SystemExit(f'No matching simple_v2 results; skipped {legacy_files} incompatible files. Check output_root and filters.')
    print(f'Skipped {legacy_files} incompatible result files.')


    order=sorted(range(len(all_results)),key=lambda i:result_order(all_results[i],filters))
    all_results=[all_results[i] for i in order]
    file_list=[file_list[i] for i in order]
    groups={}
    for data in all_results:
        group=tuple(sorted((key,value) for key,value in data['parameters'].items() if key!='seed'))
        groups.setdefault(group,[]).append(data)
    for group,runs in groups.items():
        if expected_runs is not None and len(runs)!=expected_runs:
            raise ValueError(f'Parameters {dict(group)}: {len(runs)} distinct seeds; expected {expected_runs}')
    print(f'Loaded {len(all_results)} runs in {len(groups)} parameter groups.')
    return all_results,file_list,groups
