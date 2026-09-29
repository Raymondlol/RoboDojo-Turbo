"""Per-batch timing from evaluation traces (RDTURBO_TRACE files): reset + episode seconds, action steps, mean
per-step sim/obs/render time, env-steps/s and time spent waiting for inference; full-batch time extrapolated to N
layouts. Comparing full batches is fairer than comparing total wall clock, where a half-empty tail batch and process
start-up dominate short runs.

Usage: python -m robodojo_turbo.tools.batch_times runs/<tag>/trace_w0.jsonl [...] [--layouts 125]
"""
from __future__ import annotations

import argparse
import json
import math

from robodojo_turbo.tools.shard import parse_id_spec


def load(path: str):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    inst = next((r for r in rows if r.get("kind") == "install"), {})
    k = int(inst.get("num_envs", 0))
    ids = str(inst.get("layout_ids", "") or "")
    n_layouts = len(parse_id_spec(ids)) if ids else 0
    top = sorted((r for r in rows if r.get("depth") == 0 and r.get("kind") in ("reset", "episode")), key=lambda r: r["t0"])
    batches, cur = [], None
    for r in top:
        if r["kind"] == "reset":
            cur = {"reset": r["dt"], "t0": r["t0"]}
        elif cur is not None:
            cur.update(episode=r["dt"], t1=r["t1"])
            batches.append(cur)
            cur = None
    steps = [r for r in rows if r.get("kind") in ("sim_step", "obs") and r.get("depth") == 1]
    rend = [r for r in rows if r.get("kind") == "render" and r.get("depth") == 2]
    infer = [r for r in rows if r.get("fn") == "get_action_batch"]
    left = n_layouts or k * len(batches)
    for b in batches:
        def inb(r, b=b):
            return b["t0"] <= r["t0"] <= b["t1"]
        ss = [r for r in steps if r["kind"] == "sim_step" and inb(r)]
        ob = [r for r in steps if r["kind"] == "obs" and inb(r)]
        rr = [r for r in rend if inb(r)]
        b["real"] = min(k, left)
        left -= b["real"]
        b["steps"] = len(ss)
        b["env_steps"] = sum(int(r.get("n", k)) for r in ss)
        b["sim_ms"] = 1000 * sum(r["dt"] for r in ss) / max(1, len(ss))
        b["obs_ms"] = 1000 * sum(r["dt"] for r in ob) / max(1, len(ob))
        b["render_ms"] = 1000 * sum(r["dt"] for r in rr) / max(1, len(rr))
        b["infer_s"] = sum(r["dt"] for r in infer if inb(r))
        b["total"] = b["reset"] + b["episode"]
    return k, n_layouts, batches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--layouts", type=int, default=125)
    a = ap.parse_args(argv)
    for p in a.traces:
        k, n, bs = load(p)
        print(f"\n== {p}\n   num_envs={k}, {n} layouts, {len(bs)} batch(es)")
        print("   batch | layouts | reset s | episode s | total s | steps | per step sim / obs(incl. render) / render ms | env-steps/s | inference s")
        for i, b in enumerate(bs, 1):
            eps = b["env_steps"] / b["episode"] if b["episode"] else 0
            print(f"   {i:5d} | {b['real']:2d}/{k} | {b['reset']:7.1f} | {b['episode']:9.1f} | {b['total']:7.1f} | {b['steps']:5d} | "
                  f"{b['sim_ms']:6.1f} / {b['obs_ms']:6.1f} / {b['render_ms']:6.1f} | {eps:11.1f} | {b['infer_s']:11.1f}")
        full = [b for b in bs if b["real"] == k]
        if full and k:
            tf = sum(b["total"] for b in full) / len(full)
            nb = math.ceil(a.layouts / k)
            tail = a.layouts - (nb - 1) * k
            # a tail batch occupies all k env slots too, so it is approximated by the full-batch time
            print(f"   mean full batch T({k}) = {tf:.1f} s over {len(full)} full batch(es) -> {tf / k:.1f} s per layout; "
                  f"{a.layouts} layouts = {nb} batches (tail {tail}/{k}) ~ {nb * tf / 60:.1f} min (excluding process start-up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
