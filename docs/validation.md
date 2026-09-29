# Validation

Three tiers, cheapest first. Undated results in sections 1-3, the 125-layout score comparison and the fluid/garment
section were produced with the private predecessor of this code (same patches under earlier names; not all check modes)
on one RTX 5090 + Ryzen 9 9950X3D (Isaac Sim 5.1, RoboDojo 726e9aa + XPolicyLab bb9a0b5), RoboDojo's released pi0.5
checkpoint, 1 process x 10 envs. This code re-covers them as follows: replay physics on `stack_blocks` (section 1,
2026-09-27), the dual-path checks on 11 more tasks (multi-task check run), and the fluid/cloth write-back (the same run:
`auto` chose `last` and the points never froze). Not re-run with this code: the fluid/garment score A/B and the
125-layout score comparison, which also used layouts that are not shipped. Everything dated 2026-09-27 was produced with
pre-release builds of this code: early builds for the multi-task check run and the laptop replay A/B, a later build for
the replay, frame and benchmark runs. The later build differs from the early ones in switch-value parsing (empty =
default), the replay-strict option and the split of USD-side values in the state trace; the current code (0.1.0a1)
differs from the later build in identifier names and in the edge-case fixes of 2026-09-29 (docs/STATUS.md). Switch and file names
below use the current spelling (`RDTURBO_*`).

## 1. Bit-exact physics under open-loop action replay

Closed-loop runs are not reproducible: RTX rendering differs by ~1.4/255 per pixel between identical runs, the policy
reacts to it, and trajectories diverge. So a closed-loop "same score" says little. Instead, `scripts/verify_physics.sh`
records the action chunks of one closed-loop run (switches off), replays them open loop with switches off and with the
speedup preset, and compares the float64 PhysX state after every action (robot joint positions and link poses, scene
rigid-object poses, articulated-object joint positions). The USD world transforms of articulated links (what USD-reading
scorers see) are traced under separate keys and reported separately, because `RDTURBO_USD_LAST` changes them by design.

* `stack_blocks`, 10 layouts: 5,164 states, **bit-identical** for every tested configuration: a reference with
  `REVOKE_LISTENERS`, `VIDEO_ONLY_OBS` and `RGB3` on, replayed twice; + USD write-back off; + control cache and GC freeze;
  15 of the 16 default switches (all but `OVERLAP_START`, which is launcher-side). 12 state files from different
  configurations and days are byte-identical (md5 e2c2db41...). A 2 x 5-env replay differs at every key (env grid
  positions change float32 rounding), which shows the comparison has power.
* The predecessor never replayed with every switch off; that gap is closed by the run below with this code.
* **2026-09-27 (`scripts/verify_physics.sh`, official layouts 0-9, seed 0):** one closed-loop recording with
  every switch off, replayed twice with every switch off (`off` preset) and once with the full `speedup` preset (all
  sixteen settings, incl. `REVOKE_LISTENERS`, `VIDEO_ONLY_OBS`, `RGB3`, `USD_LAST`): **5,208/5,208 PhysX states
  bit-identical** off vs off and off vs speedup; every replay took all 11 chunk batches from the recording (strict
  replay, no server call); scores identical (2/10, score 29.0, 10 ties). Replay wall clock 254-263 s (off) vs 105 s
  (speedup); a replay runs no policy inference, so this is not the benchmark ratio.
* Launcher and policy-server settings (`OVERLAP_START`, JAX/XLA) are not exercised by a replay.
* PhysX guarantees same-machine determinism for rigid bodies and articulations only. Cross-machine replay has not been
  tested.

Open-loop equality shows that the tested switches leave physics alone on the tested tasks. It does not show that
closed-loop behaviour is unchanged (arXiv 2606.04233, App. B.2, makes this point), which is what tier 3 is for.

* **`VIDEO_ONLY_OBS` and what the policy sees (2026-09-27).** The same recording replayed four times with the
  speedup preset and `RDTURBO_OBS_DUMP=1`, alternating `VIDEO_ONLY_OBS=0` / `1`; the chunk-start images uploaded to the
  policy (11 chunks x 10 layouts x 3 cameras) compared pixel by pixel. Mean absolute difference (0-255) / mean 99th
  percentile: same setting repeated (render noise) 0.466-0.531 / 2.57-2.87; `0` vs `1` (four cross pairs) 0.464-0.530 /
  2.53-2.88, for the head camera and both wrist cameras that `=1` does not read out between chunk starts. No difference
  beyond render noise; no image is byte-identical between any two runs, including repeats of the same setting.

## 2. Dual paths (`=check`)

Five check modes compute the fast and the upstream result on the same data and compare bytes (which result the run
continues with is listed in docs/switches.md): video frames
(539 calls / 5,059 frames), control resolution (105,540 float32 comparisons), parent transforms (5,164), control-dict copies
(52,186 pickles), joint-target tensors (5,174): **0 mismatches**, all on
`stack_blocks`. `RDTURBO_USD_LAST=check` compares the USD pose that rendering and USD-reading scorers see with the PhysX
pose after every action: 129,100 comparisons on `stack_blocks`, identical statistics with the switch on and off (max
position difference 3.6e-15 m; robot links and rigid objects). Articulated objects are included from this version on.
`REVOKE_LISTENERS` and `GC` have no check mode.

## 3. Paired closed-loop scores

* **2026-09-27** (the benchmark runs of docs/benchmark.md; official 25 layouts, two repetitions, no common random
  numbers): same code repeated (upstream rep 1 vs rep 2) +12.0 points [+2.4, +23.8]; speedup vs upstream -3.4
  [-12.0, +2.4] (rep 1) and -13.2 [-26.4, -2.4] (rep 2); harness vs upstream +1.8 [-8.4, +12.0] and -8.0 [-20.0, +1.2];
  exact McNemar p >= 0.25 everywhere; MDE80 11-18 points. At n = 25 per run the same-code repeat alone moves by 12
  points, so these runs cannot detect a change of that size; the speedup arm scored lowest in both repetitions. Physics
  is bit-identical under replay and the chunk-start images are within render noise (section 1), so the arms differ only
  by render noise and the policy's own sampling; see the pooled four-task comparison below.

* **2026-09-28, pooled over four tasks** (the two stack_blocks repetitions above plus one run per arm of `make_kong`,
  `pick_from_conveyor_by_image` and `pour_liquid_into_cup_random`; 125 paired official layouts): official 15/125, Turbo
  16/125 successes; score difference +0.7 points, 95% bootstrap CI [-4.2, +5.5] stratified by pair; exact McNemar
  p = 1.0; MDE80 7.4 points. Per pair: -3.4, -13.2, +12.0, 0.0 (no successes in either arm), +8.0 points. All available
  pairs are included; the pooling was chosen after the runs, not declared beforehand.

* **2026-09-29, two more runs per arm on each of the four tasks** (0.1.0a1, commit eba171e; `reproduce_benchmark.sh
  --reps 2 --legs upstream,speedup --same-jax-cache`, order official, Turbo, Turbo, official; docs/benchmark.md): 8 new
  pairs, 200 paired layouts, official 24/200 and Turbo 24/200 successes, score difference -0.1 points [-4.2, +4.0],
  McNemar p = 1.0, MDE80 6.0 points. Pooled with the five earlier pairs (pre-release builds, JAX cache warm in the
  Turbo arm only; the all-available-pairs rule of the 2026-09-28 comparison): 13 pairs, 325 paired layouts, official
  39/325, Turbo 40/325, **+0.2 points [-3.0, +3.4]**, exact McNemar p = 1.0 (15 of the 325 paired layouts succeeded
  only in the official run, 16 only with Turbo), MDE80 4.6 points. The conveyor task (no successes in 150 runs) adds 75
  ties only; without it: 250 paired layouts, +0.3 [-3.8, +4.3], MDE80 6.0 points. Per task over all pairs:
  `stack_blocks` 15 vs 15 of 100, -0.3 [-6.4, +5.6]; `make_kong` 17 vs 19 of 75, +2.7 [-6.7, +12.0];
  `pour_liquid_into_cup_random` 7 vs 6 of 75, -1.3 [-6.7, +4.0] (successes on 5 of 25 layouts, each of them in both
  arms at least once except layout 15, official only, 1 of 3, and layout 7, Turbo only, 1 of 3); `pick_from_conveyor_by_image`
  no successes in 75 layouts per arm. Read against the same-code repeat above (+12.0 points at n = 25): a change of about
  5 points averaged over all 325 layouts (6 points over the 250 layouts of the three tasks with successes), or 8-14
  points per task, would have been detected with 80% power; smaller changes may not be.

`python -m robodojo_turbo.tools.paired_compare` compares two runs layout by layout (wins/losses, mean score difference
with a paired bootstrap CI, exact McNemar, MDE80), or pools several pairs (`A1 B1 A2 B2 ...`) with a bootstrap stratified
by pair. Read it against a same-configuration repeat.

* `stack_blocks`, 125 layouts, harness only vs all switches: 18/125 vs 18/125, score difference -1.0 [-6.7, +4.6],
  McNemar p = 1.0; this comparison's own MDE80 is about 8.1 points. Two repeats of the 6-switch configuration differ by
  a similar amount (-1.5 [-6.6, +3.6]). The +/-8.9-point margin used in the predecessor's notes was chosen after the
  fact, from an n = 100 calibration on an RTX 4090 host; read the result as "no change larger than about 8 points".

## Fluid and garment tasks (2026-09-27)

Fluid and cloth scorers read USD, not PhysX: `FluidObject.get_particle_positions` reads the PointInstancer positions and,
with RoboDojo's `device=cpu`, `GarmentObject.sample_mesh_vertices` reads USD `points`. `update_transformations` (the
per-action flush of `flush` mode) writes rigid bodies and articulation links only. Test: `pour_liquid_into_cup` and
`fold_clothes`, official layouts 0-9, one recorded action file per task replayed in every arm (one replay per arm; the
other 15 default switches on except `OFFLINE_ASSETS`).

| task | arm | successes | USD points frozen (share of consecutive steps identical) | physics per step |
|---|---|---:|---:|---:|
| pour | USD write-back every substep (x2, noise floor) | 2/10, 1/10 | 0.00 | 460 ms |
| pour | `flush` (the 2026-09-24 behaviour) | **0/10** | **1.00** | 387 ms |
| pour | `keep-particles` | 2/10 | 0.00 | 381 ms |
| pour | `last` (via `auto`) | 2/10 | 0.00 | 379 ms |
| fold | USD write-back every substep (x2, noise floor) | 7/10, 4/10 | 0.00 | 246-251 ms |
| fold | closed-loop recording run (write-back every substep) | 4/10 | 0.00 | not recorded |
| fold | `flush` | **0/10** | **1.00** | 152 ms |
| fold | `keep-particles` | **0/10** | **1.00** | 155 ms |
| fold | `last` (via `auto`) | 4/10 | 0.00 | 188 ms |

* Fluid rendering is unaffected by `flush` (the isosurface does not come from the USD points); cloth rendering is: the
  garment stays flat for the whole episode while the grippers pass through it.
* Fluid and cloth physics are not bit-deterministic: two replays of the same actions already differ (pour: bottle and
  cup poses by ~1.5 mm at step 0), so these tasks are compared against a noise floor, and the decisive signal is whether
  the points freeze.
* With `auto`, `stack_blocks` takes the `flush` path and its replay stayed bit-identical to the 2026-09-24 reference.
* Not in this A/B: `pour_liquid_into_cup_random`, `pour_by_language`, `fold_clothes_random`; they are covered by the
  multi-task check run below (`auto` chose `last`, points never froze; scores not compared against a noise floor).

## What a static audit found (2026-09-27, all 54 task configurations)

* No other switch showed a task-specific assumption that fails elsewhere: every task shares one camera config, no scorer
  reads cameras, rigid parents are static category Xforms, joint-action evaluation never touches the planner, and the
  nine mirrored NVIDIA files belong to `camera_stand`, which every scene loads.
* One open item at the time: `store_laptop_and_headphones` (+ `_random`) scores a free-floating laptop articulation from
  its USD link pose; see the two sections below.

## Multi-task check run (2026-09-27, early pre-release builds)

Official layouts 0-9 per task, seed 0, the speedup preset plus every check mode (`--preset speedup,verify`), offline
assets on, one RTX 5090 + Ryzen 9 9950X3D, 1 process x 10 envs (5 for `_random` tasks), 11 tasks + 1 control leg,
closed loop. 101 of 110 layouts were evaluated: `imitate_sorting_sequence` (layouts 4, 8) and `fill_egg_holder` (5, 9)
lost two each to RoboDojo's own stability check (the merge tool now reports them as `unstable_layouts`), and
`store_laptop_and_headphones_random` finished only its first 5-env batch before Kit hung (see below).

* **Every byte/bit dual-path check that ran: 0 mismatches** (video frames 81,104; control resolution 1,319,354, incl.
  the franka support arm on three tasks; parent transforms 3,256,732; control-dict copies 55,480 on
  the tasks without a support robot, `PCI_COPY` turning itself off with one by design; joint-target tensors 208,000). No
  batch swallowed in the 11 legs that completed. The offline-asset mirror verified on every task. Fluid/cloth points never froze
  (`pour_by_language`, `pour_liquid_into_cup_random`, `fold_clothes_random`; `auto` chose `last`). Whole-card VRAM peak
  18.7-26.0 GiB (26.0 with the particle trace on the fluid task).
* **`USD_LAST` leaves bodies that fall asleep during an action stale in USD**, in `flush` and in `last` mode: max position
  difference 0.63 mm (`store_laptop_and_headphones`, headset), 0.90 mm (`match_and_pick_from_conveyor`), 1.2 mm and
  ~6 deg (`fill_egg_holder`, eggs), 0.49 mm (`pour_liquid_into_cup_random`, `last` mode), 0.32 mm
  (`store_laptop_and_headphones_random`); the rotation of the conveyor bottle differed by up to 1.4e-2 (quaternion
  component). Clean: `pick_from_conveyor_by_image`, `pour_by_language`, `fold_clothes_random` and the three franka
  tasks (there, rigid objects showed rotation-only differences up to 1.8e-5 and no position difference); `stack_blocks`
  was clean in the predecessor's 2026-09-24 check. Control leg (laptop task, `RDTURBO_USD_LAST=basecheck`, i.e.
  write-back every substep, same comparison, other switches on): rigid 0/8,000, so the switch causes it. Likely mechanism (inferred from PhysX's
  documentation, not verified): the active-actor list PhysX writes back contains only bodies that moved in the last step.
  Consumers: rendering on every task; scoring only where a scorer reads USD (the laptop; the other affected tasks score
  rigid objects from PhysX).
* The laptop's articulation links are stale in USD with `USD_LAST` off as well (control leg, other switches on:
  112/16,000, max 0.41 mm).
* CUDA error 700 in PhysX's GPU narrowphase (`Fetching GPU Narrowphase failed! 700`) appeared in both closed-loop laptop
  runs with `USD_LAST` on (all check modes on; the `_random` run then hung), and not in the control leg with it off (one
  run per arm, and the two arms ran different early builds, one without and one with `RDTURBO_FFMPEG_THREADS=2`), nor
  in the other nine tasks. It did not reproduce in six open-loop replays of the same actions with the switch off, on or
  in check mode (next section), but replays do not run policy inference on the GPU. It is neither attributed to the
  switch nor ruled out. The same error class is reported upstream (Isaac Lab issues
  [#1460](https://github.com/isaac-sim/IsaacLab/issues/1460), [#1737](https://github.com/isaac-sim/IsaacLab/issues/1737),
  [#1823](https://github.com/isaac-sim/IsaacLab/issues/1823), all open).
* `panda_link1` of the franka support arm shows rotation-only differences up to 1.7e-4 (quaternion component) with
  positions equal to 3e-8 m and every other franka link exact; not characterised (no franka control leg).

## Laptop task: open-loop replay A/B (2026-09-27, early pre-release build)

The actions recorded in the check run of `store_laptop_and_headphones` (layouts 0-9) replayed four times with the speedup
preset, alternating `RDTURBO_USD_LAST=0` / `1` (off, on, off, on):

* Replays are deterministic: both off-replays produced byte-identical state files (md5 d82d84cb...), and so did both
  on-replays (md5 a8157400...). Every chunk came from the recording (16/16 replay hits, no server call).
* **The switch does not change physics**: off vs on, the PhysX part of the state (robot joints and links, headset pose,
  laptop joint angles; the first 178 of 210 values) is bit-identical at all 7,702 recorded steps. Only the last 32 values
  differ: the USD world matrices of the two laptop links (max 5.2e-4 m), i.e. the stale USD poses described above.
* **Scores are identical**: 1/10 successes and score 30.0 in every replay; per layout 10 ties, McNemar p = 1.0. With
  nine layouts failing in every arm, this test has little power to show a flipped USD-read score.
* Two more replays with the check-run configuration (`USD_LAST=check` and the other check modes) gave the on-replays' state
  file (check mode does not change physics). The closed-loop check run's own state file is byte-identical to it as well
  (md5 a8157400...), and so are its staleness statistics (rigid 1,388/7,702, max 0.6347 mm; links 348/15,404): the
  check run's recorded physics was unaffected by its CUDA errors.
* No CUDA error in any of the six replays. Wall clock 245 s (off) vs 200 s (on) vs 230 s (check), other switches equal.
