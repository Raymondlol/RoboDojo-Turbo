"""Merge several workers' `_result.json` files (RoboDojo's format) into one result.

Checks that every layout_id appears exactly once; missing and duplicate ids are written to the output and reported on
stderr (exit status 2 with --strict). Scoring matches RoboDojo's run_eval: success_rate = successes / episodes;
score = mean(per-episode score) x 100.
Usage: python -m robodojo_turbo.tools.merge --expected 0-24 runs/w0/_result.json runs/w1/_result.json > merged.json
Provenance: repeatable `--meta k=v` pairs are stored under "provenance" (switches, machine, commit, ...), plus merged_at (UTC).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional

from robodojo_turbo.tools.shard import format_id_spec, parse_id_spec


def merge_results(results: List[dict], expected_ids: Optional[Iterable[int]] = None, sources: Optional[List[str]] = None,
                  meta: Optional[Dict[str, str]] = None, unstable_count: int = 0) -> dict:
    """unstable_count: layouts RoboDojo itself dropped as physically unstable (its stability check after reset, or
    TaskEnv.mark_env_unstable during an episode). Upstream leaves them out of _result.json and only counts them, so
    when exactly that many expected ids are missing they are reported as `unstable_layouts`, not as `missing`."""
    details: Dict[int, dict] = {}
    duplicates: List[int] = []
    unstable = 0
    for k, r in enumerate(results):
        src = sources[k] if sources else str(k)
        unstable += int(r.get("unstable_nums", 0) or 0)
        for _, d in sorted(r.get("details", {}).items(), key=lambda kv: int(kv[0])):
            lid = int(d["layout_id"])
            if lid in details:
                duplicates.append(lid)
                continue
            details[lid] = {"layout_id": lid, "success": bool(d["success"]), "score": float(d["score"]), "source": src}
    n = len(details)
    successes = sum(1 for d in details.values() if d["success"])
    score = (sum(d["score"] for d in details.values()) / n * 100.0) if n else 0.0
    missing = sorted(set(expected_ids) - set(details)) if expected_ids is not None else []
    unstable = max(unstable, int(unstable_count or 0))
    unstable_layouts: List[int] = []
    if missing and len(missing) == unstable:
        unstable_layouts, missing = missing, []
    merged = {
        "success_rate": (successes / n) if n else 0.0,
        "score": score,
        "eval_time": n,
        "successes": successes,
        "unstable_nums": unstable,
        "layouts": format_id_spec(details.keys()),
        "duplicates": sorted(duplicates),
        "missing": missing,
        "unstable_layouts": unstable_layouts,
        "details": {str(i): details[lid] for i, lid in enumerate(sorted(details))},
        "sources": sources or [],
    }
    if meta is not None:
        import datetime as _dt
        prov = dict(meta)
        prov.setdefault("merged_at", _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        merged["provenance"] = prov
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--expected", default=None, help="layout ids the result must cover, e.g. 0-24")
    ap.add_argument("--meta", action="append", default=[], help="provenance field k=v (repeatable)")
    ap.add_argument("--strict", action="store_true", help="exit 2 when layouts are missing or duplicated")
    ap.add_argument("--unstable-count", type=int, default=0,
                    help="layouts RoboDojo dropped as unstable (sum of the workers' accounting); that many missing ids are not an error")
    args = ap.parse_args()
    results = [json.load(open(p)) for p in args.paths]
    expected = parse_id_spec(args.expected) if args.expected else None
    meta = dict(kv.split("=", 1) for kv in args.meta) if args.meta else None
    merged = merge_results(results, expected, sources=[os.path.basename(p) for p in args.paths], meta=meta,
                           unstable_count=args.unstable_count)
    json.dump(merged, sys.stdout, indent=1)
    print()
    if merged["unstable_layouts"]:
        print(f"[merge] note: layouts {merged['unstable_layouts']} were dropped by RoboDojo as unstable (not scored upstream)", file=sys.stderr)
    if merged["duplicates"] or merged["missing"]:
        print(f"[merge] WARNING duplicates={merged['duplicates']} missing={merged['missing']}", file=sys.stderr)
        if args.strict:
            sys.exit(2)


if __name__ == "__main__":
    main()
