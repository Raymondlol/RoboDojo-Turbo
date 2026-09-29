"""Layout sharding: split N evaluation layouts (layout_id = 0..N-1) across W Isaac client processes.

Goal: balance the number of batches per worker (ceil(n_i / num_envs)) and keep the last batch as full as possible.
Usage:  python -m robodojo_turbo.tools.shard --layouts 25 --workers 2 --num-envs 7
Output: one JSON line per worker: {"worker": i, "ids": "0-13", "n": 14, "batches": 2, "num_envs": 7}
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Iterable, List


def parse_id_spec(spec: str) -> List[int]:
    """'0-12,20,22-23' -> [0..12, 20, 22, 23] (deduplicated, ascending)."""
    ids: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            ids.update(range(a, b + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def format_id_spec(ids: Iterable[int]) -> str:
    """[0,1,2,5,7,8] -> '0-2,5,7-8'。"""
    ids = sorted(set(int(i) for i in ids))
    if not ids:
        return ""
    out = []
    start = prev = ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = i
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


def plan_shards(n_layouts: int, workers: int, num_envs: int) -> List[dict]:
    """Contiguous shards. Total batches B = ceil(N / E) are split evenly across workers (the remainder goes to the
    first workers), then converted back to layout counts (the last worker takes what is left)."""
    if n_layouts <= 0 or workers <= 0 or num_envs <= 0:
        raise ValueError("n_layouts, workers, num_envs must be positive")
    workers = min(workers, n_layouts)
    total_batches = math.ceil(n_layouts / num_envs)
    workers = min(workers, total_batches)
    base, extra = divmod(total_batches, workers)
    batches_per_worker = [base + (1 if i < extra else 0) for i in range(workers)]
    plan, cursor = [], 0
    for i, nb in enumerate(batches_per_worker):
        take = min(nb * num_envs, n_layouts - cursor)
        if i == workers - 1:
            take = n_layouts - cursor
        ids = list(range(cursor, cursor + take))
        cursor += take
        plan.append(
            {
                "worker": i,
                "ids": format_id_spec(ids),
                "n": len(ids),
                "batches": math.ceil(len(ids) / num_envs),
                "num_envs": num_envs,
            }
        )
    assert cursor == n_layouts, (cursor, n_layouts)
    return plan


def plan_shards_ids(ids: List[int], workers: int, num_envs: int) -> List[dict]:
    """Explicit layout list (e.g. one slice per node, then split across workers) with plan_shards' balancing rule."""
    ids = sorted(set(int(i) for i in ids))
    plan = plan_shards(len(ids), workers, num_envs)
    out, cursor = [], 0
    for row in plan:
        chunk = ids[cursor: cursor + row["n"]]
        cursor += row["n"]
        out.append({**row, "ids": format_id_spec(chunk)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", type=int, default=25)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=10)
    ap.add_argument("--layout-ids", default="", help="explicit layout ids, e.g. '9-16' (overrides --layouts)")
    args = ap.parse_args()
    rows = plan_shards_ids(parse_id_spec(args.layout_ids), args.workers, args.num_envs) if args.layout_ids else plan_shards(args.layouts, args.workers, args.num_envs)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
