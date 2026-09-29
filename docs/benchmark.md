# Benchmark numbers and their conditions

Report absolute times, the ratio, and every condition. Wall clock in this repository's launcher is `wall_total`: policy
server start to last client exit (comparable across start-up modes).

## Pre-release build, same machine (2026-09-27)

`scripts/reproduce_benchmark.sh --task stack_blocks --seed 0 --layouts 25 --reps 2` with the 2026-09-27 pre-release build on the
machine below; order upstream, harness, speedup, then reversed; `wall_total` = policy server start to last client exit;
no other process on the GPU during the runs (checked every 30 s).

| arm | rep 1 | rep 2 | mean | vs upstream | vs harness | successes | score | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream (tree reverted, three camera videos, per-step upload) | 638 s | 634 s | 636 s | | | 3, 6 /25 | 20.4, 32.4 | 24.4 GiB |
| harness preset | 548 s | 544 s | 546 s | -14.2% | | 3, 4 /25 | 22.2, 24.4 | 24.4 GiB |
| speedup preset | 301 s | 296 s | 299 s | **-53.1% (2.13x)** | -45.3% (1.83x) | 2, 3 /25 | 17.0, 19.2 | 19.0 GiB |

The upstream arm's policy server imports XPolicyLab's own openpi sources (its setup script puts them first on
`PYTHONPATH`). The speedup arm includes `OVERLAP_START` and a warm JAX compilation cache (both part of the preset); the
other arms start the server first and compile from scratch. Score comparisons at this n are noisy (README, "Measured so
far"; docs/validation.md).

## Predecessor measurements (2026-09-24/25, private predecessor code)

Machine: RTX 5090 32 GB + Ryzen 9 9950X3D, driver 580.178.04, Isaac Sim 5.1, Isaac Lab 2.3.2 (isaaclab extension
0.54.3), omni.physx 107.3.26. Task `stack_blocks`, arx_x5 joint actions, pi0.5 `RoboDojo-sim-arx_x5-joint-0` step 59999,
125 layouts (official 25 + 100 sampled with our own sampler built on RoboDojo's ClutteredGenerator, seeds
910000-910099, not shipped: readers can reproduce the 25-layout part only), 1 process x 10 envs, the predecessor's lossless fast path (`RDTURBO_FAST=1` here),
head-camera video, x264 `-threads 2` (hard-coded in the predecessor), common random numbers for the policy's sampling
noise (a server-side patch not included in v0.1). "6 switches" = `REVOKE_LISTENERS`, `RGB3`, `VIDEO_ONLY_OBS`,
`USD_LAST`, `CTRL_CACHE`, `GC`.

| run | switches | wall | full batch (mean of 12) | successes |
|---|---|---:|---:|---:|
| a125 | harness only | 2330 s (38.8 min) | 178.3 s (+-5.4) | 18/125 |
| exact125a / b | harness + 6 switches | 1280 / 1295 s | 97.0 / 97.2 s | 21 / 20 |
| final125 | harness + the 16 preset settings (README) | **1265 s (21.1 min)** | **95.4 s (+-1.4)** | 18/125 |

* -45.7% wall clock = **1.84x throughput**; env-steps/s 27.7 -> 50.7. Per full batch 1.87x.
* The wall clock above is the predecessor's definition (first client start to end, with a 15 s polling granularity; the
  baseline excluded ~10 s of server start, the final run included it). For final125 vs a125, orchestration wall clock
  (30 s polling) gives -44.8% and the trace span (first reset to last episode) -45.8%. The 6-switch run exact125b
  gives -43.5% under the orchestration definition.
* Replicates: the 2026-09-23 baseline had the same wall (2330 s) and 179.0 s per batch; ten 50-layout runs of the full
  set on 2026-09-25 ran at 10.0-10.3 s per layout (final125: 10.1 s).
* Whole-card VRAM peak 24.75 -> 19.81 GiB, almost entirely from the server allocator settings and `NO_PLANNER`.
* GPU board energy -18% (29.6 -> 24.4 Wh) was measured on the 6-switch subset over 25 layouts, n = 1, as a trapezoid
  integral of 1 Hz nvidia-smi board-power samples with dropped samples; not re-measured.

## What has not been measured

* Cloth and articulated-object tasks, other policies, other GPUs (a slow-single-core host should gain more: the savings
  are main-thread CPU work; GPU work is unchanged).
* More than two alternating repetitions per arm and task; other session orders; per-batch timing of this code
  (`tools.batch_times` on the traces).

## Historical: harness gain with a remote policy server

With G0.5 (not supported in v0.1) served from another machine over an SSH tunnel, the upstream loop spent 73% of worker
time uploading raw images every step. Chunk-start upload plus two workers with one server each: 5192 s -> 1075 s on 25
layouts (4.8x), 10/25 vs 10/25 successes, RTX 4090 client + RTX PRO 6000 server, RoboDojo ee67a14 (before upstream
changed `get_obs_batch` on 2026-09-12). With a local pi0.5 server, a different change (on-demand upload plus rendering
only every 5th step, 1 worker x 10 envs, RTX 4090, host load ~32 from other tenants) was worth ~1.05x; it was not a
controlled measurement of chunk-start upload, and 1/5 rendering is not allowed under the protocol.

## Where the remaining time goes (6-switch configuration, `stack_blocks`)

py-spy, main thread, 30 s steady state, 2026-09-24, the 6-switch configuration above (not the full 16-switch stack):
rendering 48%, PhysX step 16.5%, waiting for inference 9.6%, per-action USD flush 7.4%, scoring 5.4%. Further cuts need changes to rendering, the physics device or the protocol, which this project does
not make. A second evaluation process on the same GPU does not help (rendering time-slices: 45 -> 83 ms per step).

## Across workloads, two runs per arm, same JAX cache (2026-09-29)

`scripts/reproduce_benchmark.sh --task <task> --seed 0 --layouts 25 --reps 2 --legs upstream,speedup --same-jax-cache`
with 0.1.0a1 (commit eba171e) on the machine above, 2026-09-29 02:14-05:28 ET, the four tasks one after another in one
session. Order within each task: official, Turbo, Turbo, official. Both arms use the preset's JAX compilation cache
directory; no cache entry was written during the session, so both arms saw the same cache state. Before each task the
GPU had to pass 5 checks 60 s apart (about 4 minutes), each with no other compute process (`nvidia-smi
--query-compute-apps`), less than 3000 MiB in use and a host 1-minute load below 4; during the task other compute
processes were checked every 30 s. Every 10 s the host's 1-minute load average was recorded (median 1.77, 90th
percentile 3.66, maximum 8.52 of 32 threads over the 16 legs, evaluation included), together with non-root processes
whose lifetime-average CPU in `ps` exceeded 20% of a core: one hit (one core, an analysis script of ours) during the
conveyor task's first official run. That filter does not see a long-running process that turns busy, or root
processes. `*_random` tasks: `--num-envs 5`. Per-run data (wall clock, successes, per-episode completion times, host samples): `docs/data/benchmark.json`
(`workloads`; the README's stack_blocks figure uses `hero`, the same runs).

| task | envs per batch | official run 1 / run 2 | mean | RoboDojo-Turbo run 1 / run 2 | mean | ratio |
|---|---:|---:|---:|---:|---:|---:|
| `stack_blocks` | 10 | 641 / 638 s | 639.5 s | 295 / 300 s | 297.5 s | 2.15 |
| `make_kong` | 10 | 918 / 908 s | 913 s | 400 / 395 s | 397.5 s | 2.30 |
| `pick_from_conveyor_by_image` | 10 | 908 / 914 s | 911 s | 441 / 446 s | 443.5 s | 2.05 |
| `pour_liquid_into_cup_random` | 5 | 1018 / 999 s | 1008.5 s | 750 / 756 s | 753 s | 1.34 |

Two runs of the same arm in this session differ by at most 1.9%. Across sessions the times moved more: against the
single 2026-09-28 runs the official arm was 26 s, 32 s and 89.5 s faster on make_kong, the conveyor task and the fluid
task, and the Turbo arm +1.5 s, -6.5 s and -52 s (the fluid task is the only one where both arms moved, by 6-8%; on
2026-09-28 the Turbo arm ran a pre-release build); on stack_blocks both arms stayed within 4 s of 2026-09-27. The shared
JAX cache can account for at most about 6 s per run; the rest is not explained (one run per arm on 2026-09-28). From
unrounded times the ratios moved by +0.02 (stack_blocks), -0.07 (make_kong), -0.04 (conveyor) and -0.02 (fluid). Successes per run and the pooled score comparison: README, "Across workloads";
docs/validation.md, section 3.

## Across workloads, first runs (2026-09-28)

`scripts/reproduce_benchmark.sh --task <task> --seed 0 --layouts 25 --reps 1 --legs upstream,speedup` on the machine
above, one task after another in one session, the host otherwise idle (GPU checked for other compute processes before and
during each leg; host CPU sampled every 10 s). `*_random` tasks: `--num-envs 5` for the patched leg, which matches the
cap RoboDojo applies to the unpatched leg (`clutter_env_limit`, default 5). Per-leg data, success counts and notes:
`docs/data/benchmark.json` (`workloads_2026_09_28`); figures: `python scripts/make_figures.py` (drawn from the 2026-09-29
runs above).

| task | envs per batch | official `wall_total` | RoboDojo-Turbo `wall_total` | ratio |
|---|---:|---:|---:|---:|
| `stack_blocks` (2026-09-27, first table in this file, mean of 2) | 10 | 636 s | 299 s | 2.13 |
| `make_kong` | 10 | 939 s | 396 s | 2.37 |
| `pick_from_conveyor_by_image` | 10 | 943 s | 450 s | 2.10 |
| `pour_liquid_into_cup_random` | 5 | 1098 s | 805 s | 1.36 |

The official leg ran first in each task (one run per arm); the speedup leg used the preset's warm JAX cache. Notes:
during the conveyor task's official leg another user's processes used up to 5.6 CPU cores for about 3 minutes
(host load 3-4 of 32); the affected batch took 0.80 of the previous batch's time, the same ratio as stack_blocks'
official runs, so no slowdown is visible. The fluid task's first attempt was stopped when another GPU process appeared
and is not used.

## README race GIF

`docs/img/race.gif` shows the 2026-09-27 `scripts/verify_physics.sh` replay legs (one closed-loop recording of
`stack_blocks` official layouts 0-9, replayed with every switch off and with the speedup preset; PhysX state
bit-identical) for layout 8, head camera. Each frame appears at the wall-clock time it was rendered (end of the k-th
observation span after the episode started, from the span traces), played at 15x. Layout 8 finished after 104.8 s with
every switch off and 35.6 s with the preset (2.9x); the whole batch took 207.0 s vs 73.9 s. A replay runs no policy
inference and this counts the episode only, so the ratio is larger than the closed-loop ones above (1.3-2.3x). Frame times:
`docs/data/race.json`; generator: `scripts/make_race_gif.py`.

