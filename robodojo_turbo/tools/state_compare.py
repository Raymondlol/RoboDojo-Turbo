"""Compare two RDTURBO_STATE_TRACE files (state_<worker>.pkl) key by key.

Physics entries are (layout, chunk, action_step) -> float64 vector of PhysX state: robot joint positions and link
poses, scene rigid-object poses, articulated-object joint positions. USD entries are (layout, chunk, action_step, "usd")
-> the USD world transforms of articulated-object links (what USD-reading scorers see; they depend on PhysX -> USD
write-back, which RDTURBO_USD_LAST changes by design). When both runs replay the same recorded actions
(RDTURBO_REPLAY_ACTIONS), a patch that leaves physics untouched must reproduce every physics key bit for bit on
rigid-body scenes (PhysX guarantees same-machine determinism for rigid bodies and articulations, not for particles or
cloth). USD keys are compared and reported separately and do not affect the exit status.

Usage: python -m robodojo_turbo.tools.state_compare A/state_w0.pkl B/state_w0.pkl [C/state_w0.pkl ...]
Exit status 0 when every B matches A bit for bit on all physics keys and both have the same physics keys; 1 otherwise.
Only load trace files you produced: they are pickles.
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict

import numpy as np


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def compare(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    same, maxd = 0, 0.0
    first, per_layout = {}, defaultdict(float)
    for k in keys:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            first.setdefault(k[0], (k, "shape"))
            continue
        if np.array_equal(x, y):
            same += 1
            continue
        d = float(np.max(np.abs(x - y)))
        maxd = max(maxd, d)
        per_layout[k[0]] = max(per_layout[k[0]], d)
        if k[0] not in first or k < first[k[0]][0]:
            first[k[0]] = (k, d)
    return {"common": len(keys), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)), "identical": same,
            "max_abs_diff": maxd, "first_divergence": first, "per_layout_max": dict(per_layout)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__)
        return 2
    a = load(argv[0])
    ok = True
    for p in argv[1:]:
        b = load(p)
        print(f"== {argv[0]}  vs  {p}")
        for kind, sel in (("physics", lambda k: len(k) == 3), ("USD link transforms (informational)", lambda k: len(k) == 4)):
            aa = {k: v for k, v in a.items() if sel(k)}
            bb = {k: v for k, v in b.items() if sel(k)}
            if not aa and not bb:
                continue
            r = compare(aa, bb)
            print(f"   {kind}: common keys {r['common']} (only A {r['only_a']}, only B {r['only_b']}); "
                  f"bit-identical {r['identical']}/{r['common']}; max |diff| {r['max_abs_diff']:.3g}")
            for lid in sorted(r["first_divergence"]):
                k, d = r["first_divergence"][lid]
                dd = d if isinstance(d, str) else f"{d:.3g}"
                print(f"     layout {lid}: first divergence at chunk {k[1]} step {k[2]} ({dd}); layout max {r['per_layout_max'].get(lid, 0):.3g}")
            if kind == "physics":
                ok &= r["identical"] == r["common"] and r["only_a"] == 0 and r["only_b"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
