"""Compare RDTURBO_PARTICLE_TRACE files (particles_<worker>.pkl): the USD-side fluid particles / cloth points that the
fluid and garment scorers read, per (layout, chunk, step, instance).

Fluid and cloth physics are not bit-deterministic run to run, so compare a candidate against a reference *and* against a
second reference run (the noise floor). The decisive staleness signal is the "frozen" share: the fraction of
consecutive steps whose points are byte-identical. With per-substep write-back, points move every step (frozen ~0.00);
when a patch stops writing them to USD they stay constant for the whole episode (frozen 1.00).

Usage: python -m robodojo_turbo.tools.particle_compare REF/particles_w0.pkl CAND/particles_w0.pkl [MORE ...]
Exit status 1 if any candidate has a (layout, instance) series that is frozen while the reference's is not.
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict

import numpy as np


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def series(trace: dict) -> dict:
    out = defaultdict(list)
    for k in sorted(trace):
        out[(k[0], k[3])].append(trace[k])
    return out


def frozen_stats(trace: dict) -> dict:
    """(layout, instance) -> (frozen share, mean first->last displacement in metres, steps)."""
    res = {}
    for sk, arrs in series(trace).items():
        same = sum(np.array_equal(arrs[i], arrs[i + 1]) for i in range(len(arrs) - 1))
        disp = float(np.linalg.norm(arrs[-1].astype(np.float64) - arrs[0].astype(np.float64), axis=-1).mean()) if len(arrs) > 1 else 0.0
        res[sk] = (same / max(1, len(arrs) - 1), disp, len(arrs))
    return res


def diff_stats(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    same, maxd, sumd, n = 0, 0.0, 0.0, 0
    for k in keys:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            continue
        if np.array_equal(x, y):
            same += 1
            continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        maxd, sumd, n = max(maxd, float(d.max())), sumd + float(d.mean()), n + 1
    return {"common": len(keys), "identical": same, "max_abs_diff": maxd, "mean_abs_diff": sumd / max(1, n)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__)
        return 2
    ref = load(argv[0])
    fr_ref = frozen_stats(ref)
    bad = False
    for p in argv[1:]:
        cand = load(p)
        d = diff_stats(ref, cand)
        fr = frozen_stats(cand)
        print(f"== {argv[0]}  vs  {p}")
        print(f"   common keys {d['common']}; identical {d['identical']}; max |diff| {d['max_abs_diff']:.4g} m; mean |diff| of differing keys {d['mean_abs_diff']:.4g} m")
        for sk in sorted(fr):
            r = fr_ref.get(sk, (float("nan"), float("nan"), 0))
            c = fr[sk]
            flag = "  <-- FROZEN" if c[0] > 0.99 and r[0] < 0.5 else ""
            bad |= bool(flag)
            print(f"   layout {sk[0]} {sk[1]}: frozen ref {r[0]:.2f} / cand {c[0]:.2f}; displacement ref {r[1]*100:.2f} cm / cand {c[1]*100:.2f} cm ({c[2]} steps){flag}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
