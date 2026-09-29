"""Paired per-layout comparison of two evaluation results (RoboDojo _result.json or merged_result.json).

Reports wins/losses/ties, the mean score difference with a paired bootstrap 95% CI, the exact McNemar test on
successes, and the minimum detectable effect at 80% power (MDE80 = 2.80 x sd(diff) / sqrt(n)) for both score and
success. "No significant difference" is only informative when the CI sits inside a margin you declared beforehand.

Usage: python -m robodojo_turbo.tools.paired_compare A.json B.json [--name-a base --name-b patched] [--margin 8.9]
       python -m robodojo_turbo.tools.paired_compare A1.json B1.json A2.json B2.json ...   (pooled over pairs, e.g. tasks:
       each pair is reported, then all pairs together with a bootstrap stratified by pair)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys


def load(path: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    det = d.get("details", {})
    return {int(v.get("layout_id", k)): (float(v["score"]), bool(v.get("success", v["score"] >= 1.0))) for k, v in det.items()}


def mcnemar_exact(b01: int, b10: int) -> float:
    m = b01 + b10
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, k) for k in range(0, min(b01, b10) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def compare(A: dict, B: dict, boot: int = 20000, seed: int = 0) -> dict:
    """Keys are layout ids, or (group, layout id) tuples for pooled runs; the bootstrap then resamples within each group."""
    ids = sorted(set(A) & set(B))
    n = len(ids)
    if n == 0:
        raise ValueError("no common layouts")
    da = [A[i][0] for i in ids]
    db = [B[i][0] for i in ids]
    diff = [y - x for x, y in zip(da, db)]
    rng = random.Random(seed)
    strata = {}
    for k, i in enumerate(ids):
        strata.setdefault(i[0] if isinstance(i, tuple) else None, []).append(k)
    groups = list(strata.values())
    boots = sorted(sum(diff[g[rng.randrange(len(g))]] for g in groups for _ in g) / n * 100 for _ in range(boot))
    sd = (sum((x - sum(diff) / n) ** 2 for x in diff) / max(1, n - 1)) ** 0.5
    sa = [A[i][1] for i in ids]
    sb = [B[i][1] for i in ids]
    b01 = sum((not x) and y for x, y in zip(sa, sb))
    b10 = sum(x and (not y) for x, y in zip(sa, sb))
    sdiff = [int(y) - int(x) for x, y in zip(sa, sb)]
    sd_s = (sum((x - sum(sdiff) / n) ** 2 for x in sdiff) / max(1, n - 1)) ** 0.5
    return {
        "n": n, "score_a": sum(da) / n * 100, "score_b": sum(db) / n * 100, "success_a": sum(sa), "success_b": sum(sb),
        "wins": sum(d > 1e-9 for d in diff), "losses": sum(d < -1e-9 for d in diff),
        "ties": sum(abs(d) <= 1e-9 for d in diff), "mean_diff": sum(diff) / n * 100,
        "ci95": (boots[int(0.025 * boot)], boots[int(0.975 * boot)]),
        "mcnemar": {"only_b": b01, "only_a": b10, "p": mcnemar_exact(b01, b10)},
        "mde80_score": 2.80 * sd * 100 / math.sqrt(n), "mde80_success_pp": 2.80 * sd_s * 100 / math.sqrt(n),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="A B", help="two results, or pairs A1 B1 A2 B2 ... to pool")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--margin", type=float, default=None, help="pre-declared equivalence margin in score points")
    ap.add_argument("--strict", action="store_true", help="exit 2 when a layout is present in only one of the two runs")
    a = ap.parse_args(argv)
    if len(a.runs) < 2 or len(a.runs) % 2:
        ap.error("give two results, or an even number of results as pairs A1 B1 A2 B2 ...")
    pairs = [(a.runs[k], a.runs[k + 1]) for k in range(0, len(a.runs), 2)]
    A, B, missing = {}, {}, False
    for g, (pa, pb) in enumerate(pairs):
        Ag, Bg = load(pa), load(pb)
        only_a, only_b = sorted(set(Ag) - set(Bg)), sorted(set(Bg) - set(Ag))
        if only_a or only_b:
            missing = True
            print(f"{pa} vs {pb}: layouts only in {a.name_a}: {only_a}; only in {a.name_b}: {only_b} (compared on the common layouts)")
        if len(pairs) > 1:
            print(f"-- pair {g + 1}: {pa} vs {pb}")
            report(compare(Ag, Bg, a.boot), a)
        A.update({(g, i): v for i, v in Ag.items()} if len(pairs) > 1 else Ag)
        B.update({(g, i): v for i, v in Bg.items()} if len(pairs) > 1 else Bg)
    if len(pairs) > 1:
        print(f"-- pooled over {len(pairs)} pairs (bootstrap stratified by pair)")
    r = compare(A, B, a.boot)
    report(r, a)
    return 2 if (a.strict and missing) else 0


def report(r: dict, a) -> None:
    lo, hi = r["ci95"]
    print(f"n={r['n']} layouts | {a.name_a}: score {r['score_a']:.1f}, success {r['success_a']}/{r['n']} | "
          f"{a.name_b}: score {r['score_b']:.1f}, success {r['success_b']}/{r['n']}")
    print(f"paired: {a.name_b} wins {r['wins']} / loses {r['losses']} / ties {r['ties']}; mean diff {r['mean_diff']:+.1f} pts, "
          f"95% bootstrap CI [{lo:+.1f}, {hi:+.1f}]")
    mc = r["mcnemar"]
    print(f"success McNemar: only-{a.name_b} {mc['only_b']}, only-{a.name_a} {mc['only_a']}, exact p={mc['p']:.3f}")
    print(f"MDE80: {r['mde80_score']:.1f} score points, {r['mde80_success_pp']:.1f} pp success")
    if a.margin is not None:
        inside = -a.margin < lo and hi < a.margin
        print(f"equivalence within +/-{a.margin:g} points: {'yes' if inside else 'no'} (CI [{lo:+.1f}, {hi:+.1f}])")


if __name__ == "__main__":
    sys.exit(main())
