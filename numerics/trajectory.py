"""Training-level check on short trajectories (the statistical gate).

A trajectory is `python -m numerics record --losses-only` of one case per seed: per-step losses on every rank.
The spec names endpoints, each the mean of one loss over a step window (e.g. the last 50 steps of a phase
segment), averaged over ranks. The reference's seed-to-seed standard deviation sigma of every endpoint, measured
on its own seeds before any candidate is seen, is frozen; the bound is delta = k x sigma (default k = 2).

A candidate runs the same seeds; per endpoint the paired differences d_i = candidate - reference give mean m and a
paired t-test p-value. Per endpoint:
  deviates  |m| > delta and p < alpha       (a real, practically relevant shift)
  within    |m| <= delta
  unclear   |m| > delta but not significant (too few seeds to tell)
Overall: 'different' if any endpoint deviates, 'equivalent' if all are within, otherwise 'inconclusive' (run more
seeds). With final=True an unclear endpoint counts as deviating: the point estimate is beyond delta.

Paired runs of a chaotic GAN training decorrelate within a few hundred steps, so an equivalence test (TOST,
numerics.stats) with delta = 1 sigma rejects even legitimate rounding variants; this rule only rejects shifts
that are both significant and larger than the reference's own seed spread allows.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from .stats import t_cdf


def load_runs(root):
    """{seed: {step: {loss: mean over ranks}}} from trajectory manifests."""
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(Path(root).glob('*.r*.json')):
        m = json.loads(path.read_text())
        if m.get('schema') != 'numerics-trajectory@1':
            continue
        seed = m['case']['seed']
        for entry in m['steps']:
            for k, v in entry['losses'].items():
                per[seed][entry['step']][k].append(v)
    return {seed: {s: {k: statistics.fmean(v) for k, v in losses.items()} for s, losses in steps.items()}
            for seed, steps in per.items()}


def endpoint_values(run, spec):
    out = {}
    for e in spec['endpoints']:
        a, b = e['steps']
        vals = [run[s][e['loss']] for s in range(a, b) if s in run and e['loss'] in run[s]]
        if len(vals) != b - a:
            raise ValueError(f"endpoint {e['name']}: {len(vals)} of {b - a} steps recorded")
        out[e['name']] = statistics.fmean(vals)
    return out


def freeze(ref_root, spec, k=2.0):
    """delta = k x the reference's own seed-to-seed standard deviation, per endpoint."""
    runs = load_runs(ref_root)
    if len(runs) < 3:
        raise ValueError('need at least three reference seeds to freeze delta')
    values = {seed: endpoint_values(run, spec) for seed, run in sorted(runs.items())}
    endpoints = {}
    for e in spec['endpoints']:
        xs = [v[e['name']] for v in values.values()]
        sigma = statistics.stdev(xs)
        endpoints[e['name']] = {'reference_mean': statistics.fmean(xs), 'sigma_seed': sigma, 'delta': k * sigma}
    return {'schema': 'numerics-trajectory-frozen@2', 'spec': spec, 'k': k, 'reference_seeds': sorted(runs),
            'reference_values': {str(s): v for s, v in values.items()}, 'endpoints': endpoints}


def _paired(diffs):
    n = len(diffs)
    mean, sd = statistics.fmean(diffs), statistics.stdev(diffs)
    if sd == 0:
        return mean, sd, (0.0 if mean != 0 else 1.0)
    t = abs(mean) / (sd / math.sqrt(n))
    return mean, sd, 2 * (1 - t_cdf(t, n - 1))


def compare(frozen, cand_root, alpha=0.05, final=False):
    spec = frozen['spec']
    ref_values = {int(s): v for s, v in frozen['reference_values'].items()}
    cand_runs = load_runs(cand_root)
    seeds = sorted(set(cand_runs) & set(ref_values))
    report = {'seeds': seeds, 'alpha': alpha, 'final': final, 'endpoints': {}, 'verdict': 'inconclusive'}
    if len(seeds) < 2:
        report['verdict'] = 'error'
        report['error'] = f'need at least two paired seeds, have {seeds}'
        return report
    cand_values = {seed: endpoint_values(cand_runs[seed], spec) for seed in seeds}
    states = []
    for e in spec['endpoints']:
        n = e['name']
        delta = frozen['endpoints'][n]['delta']
        sigma = frozen['endpoints'][n]['sigma_seed']
        diffs = [cand_values[s][n] - ref_values[s][n] for s in seeds]
        mean, sd, p = _paired(diffs)
        if abs(mean) <= delta:
            state = 'within'
        elif p < alpha or final:
            state = 'deviates'
        else:
            state = 'unclear'
        report['endpoints'][n] = {'state': state, 'mean': mean, 'sd': sd, 'p_paired_t': p, 'delta': delta,
                                  'mean_in_sigma': mean / sigma if sigma else None, 'diffs': diffs}
        states.append(state)
    if any(x == 'deviates' for x in states):
        report['verdict'] = 'different'
    elif all(x == 'within' for x in states):
        report['verdict'] = 'equivalent'
    return report


def bound(frozen, cand_root, k=3.0):
    """Gate 3 of every check (the agent's, `check --n1`, the final check's): the candidate's trajectory on the first
    frozen reference seed. Per endpoint the paired difference d = candidate - reference (the mean over the seeds run,
    by default one) against the bias bound k x sigma_seed, sigma_seed the reference's own seed-to-seed standard
    deviation frozen before any candidate (default k = 3): within |d| <= k x sigma, beyond otherwise. All within:
    'equivalent' (PASS, N1); any beyond: 'different' (INCORRECT). No test and no further seeds."""
    spec = frozen['spec']
    ref_values = {int(s): v for s, v in frozen['reference_values'].items()}
    cand_runs = load_runs(cand_root)
    seeds = sorted(set(cand_runs) & set(ref_values))
    report = {'seeds': seeds, 'sigma_k': k, 'endpoints': {}, 'verdict': 'error'}
    if not seeds:
        report['error'] = 'no candidate trajectory on a reference seed'
        return report
    cand_values = {seed: endpoint_values(cand_runs[seed], spec) for seed in seeds}
    for e in spec['endpoints']:
        n = e['name']
        sigma = frozen['endpoints'][n]['sigma_seed']
        limit = k * sigma
        diffs = [cand_values[s][n] - ref_values[s][n] for s in seeds]
        mean = statistics.fmean(diffs)
        report['endpoints'][n] = {'state': 'within' if abs(mean) <= limit else 'beyond', 'mean': mean,
                                  'sigma_seed': sigma, 'bound': limit,
                                  'mean_in_sigma': mean / sigma if sigma else None, 'diffs': diffs}
    report['verdict'] = ('equivalent' if all(e['state'] == 'within' for e in report['endpoints'].values())
                         else 'different')
    return report
