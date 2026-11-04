"""python -m numerics {record,calibrate,compare,selftest,tost} -- see docs/numerics.md.

compare exits 0 PASS / 2 INCORRECT / 3 ERROR; selftest exits 0 only if every mutation is caught.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

EXIT = {'PASS': 0, 'INCORRECT': 2, 'ERROR': 3}


def load_adapter(spec, kwargs):
    module, _, cls = spec.partition(':')
    return getattr(importlib.import_module(module), cls)(**kwargs)


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m numerics', description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record', help='record cases with a task adapter (run under torchrun for multi-rank)')
    r.add_argument('--adapter', required=True, help='module:Class')
    r.add_argument('--adapter-kwargs', default='{}', help='JSON kwargs for the adapter')
    r.add_argument('--cases', required=True, help='JSON file: list of case dicts')
    r.add_argument('--out', required=True)
    r.add_argument('--golden', action='store_true', help='record the fp64 reference')
    r.add_argument('--dtype-trace', action='store_true', help='trace op output dtypes in step 0 (eager only)')
    r.add_argument('--losses-only', action='store_true', help='record a training trajectory (per-step losses only)')
    r.add_argument('--deterministic', action='store_true',
                   help='deterministic kernels, so that a bitwise comparison between two runs is meaningful')
    r.add_argument('--math-mode', choices=('default', 'tf32'), default='default',
                   help='process-wide math mode set before the adapter builds (recorded per step)')
    r.add_argument('--sketch-dim', type=int, default=0,
                   help='store tensors larger than this as CountSketches (e.g. 4096 for large models); 0 = full')
    c = sub.add_parser('calibrate', help='freeze gates from legitimate variants of the reference')
    c.add_argument('--ref', required=True)
    c.add_argument('--variants', nargs='+', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--k-fwd', type=float, default=2.0)
    c.add_argument('--k-bwd', type=float, default=3.0)
    c.add_argument('--floor-fwd', type=float, default=1e-6)
    c.add_argument('--floor-bwd', type=float, default=1e-5)
    c.add_argument('--golden', help='fp64 records: pools the reference\'s own distance to fp64 for arbitration')
    tf = sub.add_parser('traj-freeze', help='freeze per-endpoint delta from the reference trajectories')
    tf.add_argument('--ref', required=True)
    tf.add_argument('--spec', required=True, help='JSON: {"endpoints": [{"name", "loss", "steps": [a, b]}]}')
    tf.add_argument('--k', type=float, default=2.0, help='delta = k x reference seed-to-seed sigma')
    tf.add_argument('--out', required=True)
    tc = sub.add_parser('traj-compare', help='paired-seed TOST of a candidate\'s trajectories')
    tc.add_argument('--frozen', required=True)
    tc.add_argument('--cand', required=True)
    tc.add_argument('--final', action='store_true', help='last seed batch: an endpoint beyond delta counts as deviating')
    tc.add_argument('--sigma-k', type=float,
                    help='gate 3: per endpoint |paired difference| <= k x reference sigma (bound), no test')
    tc.add_argument('--out', required=True)
    o = sub.add_parser('loo', help='leave-one-out: each legitimate variant must PASS gates calibrated on the others')
    o.add_argument('--ref', required=True)
    o.add_argument('--variants', nargs='+', required=True)
    o.add_argument('--golden')
    o.add_argument('--out', required=True)
    m = sub.add_parser('compare', help='judge a candidate against the reference')
    m.add_argument('--ref', required=True)
    m.add_argument('--cand', required=True)
    m.add_argument('--thresholds', required=True)
    m.add_argument('--golden')
    m.add_argument('--exclude', action='append', default=[], help='REGEX=REASON (reason is mandatory)')
    m.add_argument('--out', required=True)
    s = sub.add_parser('selftest', help='mutation-test the judge on this task\'s records')
    s.add_argument('--ref', required=True)
    s.add_argument('--variant', required=True, help='a legitimate variant that must PASS')
    s.add_argument('--thresholds', required=True)
    s.add_argument('--out', required=True)
    t = sub.add_parser('tost', help='paired equivalence test')
    t.add_argument('--diffs', required=True, help='comma-separated paired differences (candidate - reference)')
    t.add_argument('--delta', type=float, required=True, help='frozen equivalence margin')
    t.add_argument('--higher-is-better', choices=('yes', 'no'))
    a = p.parse_args(argv)

    if a.cmd == 'record':
        from .record import Recorder, set_deterministic, set_math_mode
        if a.deterministic:
            set_deterministic()
        set_math_mode(a.math_mode)
        adapter = load_adapter(a.adapter, json.loads(a.adapter_kwargs))
        recorder = Recorder(adapter, a.out, golden=a.golden, dtype_trace=a.dtype_trace, sketch_dim=a.sketch_dim,
                            losses_only=a.losses_only)
        for case in json.loads(Path(a.cases).read_text()):
            recorder.run(case)
        return 0
    if a.cmd == 'calibrate':
        from .judge import calibrate, thresholds_sha
        th = calibrate(a.ref, a.variants, a.k_fwd, a.k_bwd, a.floor_fwd, a.floor_bwd, golden=a.golden)
        Path(a.out).write_text(json.dumps(th, indent=1, sort_keys=True) + '\n')
        print(json.dumps({'gates': len(th['gates']), 'sha256': thresholds_sha(th)}))
        return 0
    if a.cmd == 'traj-freeze':
        from .trajectory import freeze
        fr = freeze(a.ref, json.loads(Path(a.spec).read_text()), k=a.k)
        Path(a.out).write_text(json.dumps(fr, indent=1) + '\n')
        print(json.dumps({n: round(e['delta'], 6) for n, e in fr['endpoints'].items()}))
        return 0
    if a.cmd == 'traj-compare':
        from .trajectory import bound as traj_bound, compare as traj_compare
        frozen = json.loads(Path(a.frozen).read_text())
        rep = traj_bound(frozen, a.cand, k=a.sigma_k) if a.sigma_k else traj_compare(frozen, a.cand, final=a.final)
        Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')
        print(json.dumps({'verdict': rep['verdict'], 'seeds': rep['seeds'],
                          'endpoints': {n: e['state'] for n, e in rep['endpoints'].items()}}))
        return {'equivalent': 0, 'different': 2, 'inconclusive': 4}.get(rep['verdict'], 3)
    if a.cmd == 'loo':
        from .judge import leave_one_out
        res = leave_one_out(a.ref, a.variants, golden=a.golden)
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
        print(json.dumps({'ok': res['ok'], 'results': [(r['variant'], r['status'], r['violations'], r['unresolved'])
                                                        for r in res['results']]}))
        return 0 if res['ok'] else 1
    if a.cmd == 'compare':
        from .judge import compare
        exclude = []
        for item in a.exclude:
            pattern, sep, reason = item.partition('=')
            if not sep or not reason.strip():
                p.error(f'--exclude needs REGEX=REASON: {item}')
            exclude.append((pattern, reason))
        rep = compare(a.ref, a.cand, json.loads(Path(a.thresholds).read_text()), golden=a.golden, exclude=exclude)
        Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')
        print(json.dumps({k: rep[k] for k in ('status', 'measured_tier', 'checked', 'bitwise_identical')}
                         | {'violations': len(rep['violations']), 'unresolved': len(rep['unresolved'])}))
        return EXIT[rep['status']]
    if a.cmd == 'selftest':
        from .selftest import run
        ev = run(a.ref, a.variant, json.loads(Path(a.thresholds).read_text()), out=a.out)
        print(json.dumps({'ok': ev['ok'], 'failed': [r['mutation'] for r in ev['results'] if not r['ok']]}))
        return 0 if ev['ok'] else 1
    if a.cmd == 'tost':
        from .stats import tost
        hib = None if a.higher_is_better is None else a.higher_is_better == 'yes'
        print(json.dumps(tost([float(x) for x in a.diffs.split(',')], a.delta, higher_is_better=hib)))
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
