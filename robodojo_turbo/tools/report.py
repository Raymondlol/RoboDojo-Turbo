"""Bubble report: per-phase exclusive time, GPU utilisation and cross-worker overlap from client/server traces (jsonl)
and nvidia-smi samples (csv).

Usage:
  python -m robodojo_turbo.tools.report --out <dir> --client trace_w0.jsonl [trace_w1.jsonl ...] \
      [--server server_trace.jsonl] [--gpu-sim gpu.csv] [--gpu-policy gpu_policy.csv] [--baseline-wall SEC] [--title ...]
Writes <out>/bubble_report.json and <out>/bubble_report.md.
Definitions:
  wall             = earliest t0 to latest t1 over the top-level spans of all workers
  exclusive(kind)  = span.dt minus the dt of its direct children (nesting is not double counted)
  policy_wait      = exclusive time of ws(get_action_batch): the simulator waiting for policy inference
  transport        = exclusive time of ws(update_obs_batch): observation encode/transfer/decode round trips
  overlap          = with several workers, summed / union length of ws(get_action_batch) intervals (>1: requests overlap)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _kind_key(r: dict) -> str:
    k = r.get("kind", "?")
    if k == "ws" and r.get("fn"):
        k = f"ws:{r['fn']}"
    return k


def exclusive_times(spans: List[dict]) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Exclusive time and count per kind."""
    child_sum: Dict[int, float] = defaultdict(float)
    for s in spans:
        p = s.get("parent")
        if p is not None:
            child_sum[p] += s["dt"]
    excl: Dict[str, float] = defaultdict(float)
    cnt: Dict[str, int] = defaultdict(int)
    for s in spans:
        k = _kind_key(s)
        excl[k] += max(0.0, s["dt"] - child_sum.get(s["id"], 0.0))
        cnt[k] += 1
    return dict(excl), dict(cnt)


def union_length(intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    total, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s)


def gpu_stats(path: str) -> dict:
    """nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1"""
    utils, mems = [], []
    with open(path) as fh:
        rd = csv.reader(fh)
        for row in rd:
            if len(row) < 3 or "timestamp" in row[0]:
                continue
            try:
                mems.append(float(row[1].strip().split()[0]))
                utils.append(float(row[2].strip().split()[0]))
            except (ValueError, IndexError):
                continue
    if not utils:
        return {"samples": 0}
    utils_sorted = sorted(utils)
    return {
        "samples": len(utils),
        "util_mean": sum(utils) / len(utils),
        "util_p50": utils_sorted[len(utils) // 2],
        "util_p90": utils_sorted[int(len(utils) * 0.9)],
        "util_busy_frac_gt20": sum(1 for u in utils if u > 20) / len(utils),
        "mem_peak_mib": max(mems) if mems else None,
    }


def analyze_client(traces: Dict[str, List[dict]]) -> dict:
    per_worker = {}
    all_top: List[Tuple[float, float]] = []
    infer_iv: List[Tuple[float, float]] = []
    for w, rows in traces.items():
        spans = [r for r in rows if "dt" in r]
        events = [r for r in rows if r.get("ev")]
        excl, cnt = exclusive_times(spans)
        top = [(s["t0"], s["t1"]) for s in spans if s.get("parent") is None]
        all_top += top
        w_t0 = min((s["t0"] for s in spans), default=0.0)
        w_t1 = max((s["t1"] for s in spans), default=0.0)
        wall = w_t1 - w_t0
        covered = union_length(top)
        ga = [(s["t0"], s["t1"]) for s in spans if _kind_key(s) == "ws:get_action_batch"]
        infer_iv += ga
        steps = sum(s.get("n", 0) for s in spans if s.get("kind") == "sim_step")
        up_bytes = sum(s.get("bytes", 0) for s in spans if _kind_key(s) == "ws:update_obs_batch")
        install = next((e for e in events if e.get("kind") == "install"), {})
        per_worker[w] = {
            "wall_s": wall,
            "t0": w_t0,
            "t1": w_t1,
            "covered_s": covered,
            "uncovered_s": max(0.0, wall - covered),
            "exclusive_s": dict(sorted(excl.items(), key=lambda kv: -kv[1])),
            "counts": cnt,
            "env_steps": steps,
            "episodes": sum(s.get("n", 0) for s in spans if s.get("kind") == "reset"),
            "obs_upload_bytes": up_bytes,
            "policy_wait_s": excl.get("ws:get_action_batch", 0.0),
            "transport_s": excl.get("ws:update_obs_batch", 0.0),
            "config": {k: install.get(k) for k in ("num_envs", "layout_ids", "obs_every_step", "video_every")},
        }
    if not per_worker:
        return {}
    T0 = min(v["t0"] for v in per_worker.values())
    T1 = max(v["t1"] for v in per_worker.values())
    wall = T1 - T0
    infer_sum = sum(e - s for s, e in infer_iv)
    infer_union = union_length(infer_iv)
    agg_excl: Dict[str, float] = defaultdict(float)
    for v in per_worker.values():
        for k, t in v["exclusive_s"].items():
            agg_excl[k] += t
    return {
        "wall_s": wall,
        "workers": len(per_worker),
        "per_worker": per_worker,
        "exclusive_s_all_workers": dict(sorted(agg_excl.items(), key=lambda kv: -kv[1])),
        "env_steps": sum(v["env_steps"] for v in per_worker.values()),
        "episodes": sum(v["episodes"] for v in per_worker.values()),
        "policy_requests_s_sum": infer_sum,
        "policy_requests_s_union": infer_union,
        "policy_overlap_ratio": (infer_sum / infer_union) if infer_union > 0 else 0.0,
        "policy_busy_frac_of_wall": (infer_union / wall) if wall > 0 else 0.0,
        "sim_side_blocked_frac": (sum(v["policy_wait_s"] + v["transport_s"] for v in per_worker.values()) / (len(per_worker) * wall)) if wall > 0 else 0.0,
        "steps_per_s": (sum(v["env_steps"] for v in per_worker.values()) / wall) if wall > 0 else 0.0,
    }


def analyze_server(rows: List[dict], wall: Optional[float]) -> dict:
    spans = [r for r in rows if "dt" in r]
    infer = [s for s in spans if s.get("kind") == "srv_infer"]
    upd = [s for s in spans if s.get("kind") == "srv_update_obs"]
    infer_s = sum(s["dt"] for s in infer)
    model_ms = [s.get("model_ms") for s in infer if s.get("model_ms") is not None]
    batches = [s.get("batch", 1) for s in infer]
    out = {
        "infer_calls": len(infer),
        "infer_s": infer_s,
        "infer_mean_s": (infer_s / len(infer)) if infer else 0.0,
        "model_ms_mean": (sum(model_ms) / len(model_ms)) if model_ms else None,
        "batch_mean": (sum(batches) / len(batches)) if batches else None,
        "samples_inferred": sum(batches),
        "update_obs_calls": len(upd),
        "update_obs_s": sum(s["dt"] for s in upd),
        "pad_waste_frac": (1 - sum(batches) / sum(s.get("pad", s.get("batch", 1)) for s in infer)) if infer else None,
    }
    if wall:
        out["busy_frac_of_wall"] = (infer_s + out["update_obs_s"]) / wall
    return out


def render_md(rep: dict) -> str:
    L = []
    c = rep.get("client", {})
    L.append(f"# Bubble report: {rep.get('title','eval')}")
    L.append("")
    if c:
        L.append(f"- wall **{c['wall_s']/60:.1f} min** ({c['workers']} sim worker(s), {c['episodes']} episodes, {c['env_steps']} env-steps, {c['steps_per_s']:.1f} env-steps/s)")
        if rep.get("baseline_wall_s"):
            L.append(f"- vs baseline {rep['baseline_wall_s']/60:.1f} min: **{rep['baseline_wall_s']/c['wall_s']:.2f}x**")
        L.append(f"- policy requests (get_action_batch) cover {c['policy_busy_frac_of_wall']*100:.1f}% of wall; overlap ratio {c['policy_overlap_ratio']:.2f} (>1 = inference of several workers overlaps)")
        L.append(f"- simulator blocked (waiting for inference + observation transfer): {c['sim_side_blocked_frac']*100:.1f}% of each worker's wall")
        L.append("")
        L.append("## Exclusive time per phase (summed over workers)")
        L.append("")
        L.append("| phase | s | share of worker x wall |")
        L.append("|---|---:|---:|")
        denom = c["wall_s"] * c["workers"]
        for k, t in c["exclusive_s_all_workers"].items():
            L.append(f"| {k} | {t:.1f} | {t/denom*100:.1f}% |")
        unc = sum(v["uncovered_s"] for v in c["per_worker"].values())
        L.append(f"| (not covered by top-level spans: main loop, seeds, JSON) | {unc:.1f} | {unc/denom*100:.1f}% |")
        L.append("")
        L.append("## Per worker")
        L.append("")
        L.append("| worker | num_envs | layouts | episodes | wall min | policy wait s | obs transfer s | upload MB |")
        L.append("|---|---:|---|---:|---:|---:|---:|---:|")
        for w, v in c["per_worker"].items():
            cfg = v["config"]
            L.append(f"| {w} | {cfg.get('num_envs')} | {cfg.get('layout_ids') or 'all'} | {v['episodes']} | {v['wall_s']/60:.1f} | {v['policy_wait_s']:.0f} | {v['transport_s']:.0f} | {v['obs_upload_bytes']/1e6:.0f} |")
        L.append("")
    s = rep.get("server")
    if s:
        L.append("## Server (policy)")
        L.append("")
        mm = f"{s['model_ms_mean']:.0f} ms" if s.get('model_ms_mean') is not None else "n/a"
        bm = f"{s['batch_mean']:.1f}" if s.get('batch_mean') is not None else "n/a"
        L.append(f"- {s['infer_calls']} inference calls, {s['infer_s']:.0f} s total ({s['infer_mean_s']*1000:.0f} ms/call; model forward {mm}), mean batch {bm}, {s['samples_inferred']} samples")
        if s.get("busy_frac_of_wall") is not None:
            L.append(f"- server busy {s['busy_frac_of_wall']*100:.1f}% of wall (inference + update_obs)")
        if s.get("pad_waste_frac") is not None:
            L.append(f"- batch padding waste {s['pad_waste_frac']*100:.1f}%")
        L.append("")
    g = rep.get("gpu", {})
    if g:
        L.append("## GPU samples (nvidia-smi, 1 Hz)")
        L.append("")
        L.append("| side | samples | util mean | p50 | p90 | util>20% share | memory peak MiB |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for side, st in g.items():
            if st.get("samples"):
                L.append(f"| {side} | {st['samples']} | {st['util_mean']:.1f}% | {st['util_p50']:.0f}% | {st['util_p90']:.0f}% | {st['util_busy_frac_gt20']*100:.0f}% | {st['mem_peak_mib']:.0f} |")
        L.append("")
    if rep.get("result"):
        r = rep["result"]
        L.append("## Score (merged)")
        L.append("")
        L.append(f"- {r.get('successes')}/{r.get('eval_time')} successes = {r.get('success_rate',0)*100:.1f}%, score **{r.get('score',0):.1f}**; layouts {r.get('layouts')}; duplicates {r.get('duplicates')}, missing {r.get('missing')}")
        L.append("")
    return "\n".join(L)


def build_report(client_paths: List[str], server_path: Optional[str] = None, gpu: Optional[Dict[str, str]] = None,
                 baseline_wall_s: Optional[float] = None, result_path: Optional[str] = None, title: str = "eval") -> dict:
    traces = {}
    for p in client_paths:
        rows = load_jsonl(p)
        w = next((r.get("w") for r in rows if r.get("w")), os.path.basename(p))
        traces[w] = rows
    client = analyze_client(traces) if traces else {}
    rep = {"title": title, "client": client, "baseline_wall_s": baseline_wall_s}
    if server_path and os.path.exists(server_path):
        rep["server"] = analyze_server(load_jsonl(server_path), client.get("wall_s"))
    if gpu:
        rep["gpu"] = {side: gpu_stats(p) for side, p in gpu.items() if p and os.path.exists(p)}
    if result_path and os.path.exists(result_path):
        rep["result"] = json.load(open(result_path))
        rep["result"].pop("details", None)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--client", nargs="+", required=True)
    ap.add_argument("--server", default=None)
    ap.add_argument("--gpu-sim", default=None)
    ap.add_argument("--gpu-policy", default=None)
    ap.add_argument("--baseline-wall", type=float, default=None)
    ap.add_argument("--result", default=None, help="merged_result.json from the merge tool")
    ap.add_argument("--title", default="eval")
    args = ap.parse_args()
    rep = build_report(args.client, args.server, {"sim": args.gpu_sim, "policy": args.gpu_policy}, args.baseline_wall, args.result, args.title)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "bubble_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    md = render_md(rep)
    with open(os.path.join(args.out, "bubble_report.md"), "w") as fh:
        fh.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
