# Status (work in progress)

Items dated 2026-09-27 and 2026-09-28 ran on pre-release builds of this code (docs/validation.md says how they differ).

## Done in this tree

- [x] Package with a patch registry, profiles (`harness`, `speedup`, `all`), `--only` / `--exclude` (typos rejected, dependents named), declared dependencies
- [x] Hash pins for every upstream file a patch may rewrite (RoboDojo 726e9aa, XPolicyLab bb9a0b5; d6332bf leaves them unchanged); refuse otherwise (`--force-unpinned`). Isaac Sim / Kit versions are not checked at apply time
- [x] Upstream API presence checks for patches that call upstream functions (e.g. `render_for_capture`)
- [x] Backups taken before any patch runs; byte-exact `revert` (tested for every profile and every single patch); an interrupted apply (incl. Ctrl-C) rolls back; revert refuses to discard post-apply edits without `--force` and never overwrites a file replaced after apply
- [x] A patched tree whose state directory was deleted is detected (`status`, `check-env`, `apply`, `revert`, launcher) with exact restore commands
- [x] `apply --dry-run` runs every patch on a scratch copy (pins, APIs, anchors, compile); CI runs it against upstream HEAD weekly
- [x] Refuse to patch a tree that is already patched (with or without its state directory)
- [x] "All switches off = upstream": ffmpeg thread cap and server memory fraction are their own `RDTURBO_*` switches; no observation tagging; no live dashboard
- [x] One switch grammar: unset or empty = upstream; `check-env` rejects values outside the documented set and lists the active switches
- [x] `RDTURBO_NO_PLANNER`: any planner use exits with status 3 instead of being swallowed
- [x] Accounting of batches swallowed by RoboDojo's generic exception handler; status file; `RDTURBO_STRICT=1` (exits after upstream's own cleanup)
- [x] Launcher: no hard-coded paths, never accepts NVIDIA's licenses for the user, all guards before anything is created, refuses half-applied trees and switches whose patch is missing, refuses a busy or duplicate server port, one server per shard, unique run id per invocation, stall timeout, cleanup (and a log line) on every exit path, non-zero exit on missing layouts, RoboDojo's unstable layouts reported separately
- [x] `RDTURBO_CTRL_CACHE` guard: no cache hits without a substep counter
- [x] `RDTURBO_USD_LAST_MODE=auto|flush|last|keep-particles` (invalid values stop the process at start-up); check mode covers articulated objects
- [x] State trace: PhysX values decide `state_compare`; USD-side link transforms are reported separately; `RDTURBO_REPLAY_STRICT=1` stops a replay that runs out of recorded chunks
- [x] `RDTURBO_OBS_DUMP` for chunk-start camera comparisons
- [x] English everywhere; `RDTURBO_*` prefix (upstream owns `ROBODOJO_*`); research extras removed (CRN server patch, DSRL, live dashboard, frame hashes, extension disabling, pose cache, G0.5 loop)
- [x] GPU-free tests (tools, injected code with stubs, patch matrix against pinned upstream incl. the exact upstream Pi_05 call sequence, interrupted apply, lost state directory, dry-run anchor breakage); launcher checked with a stub server/client/nvidia-smi; CI workflow
- [x] LICENSE (Apache-2.0), NOTICE, THIRD_PARTY_NOTICES; presets shipped in the wheel
- [x] Two independent code reviews on 2026-09-27; confirmed findings addressed in code or docs

## GPU validation (local RTX 5090)

- [x] Multi-task check-mode run (2026-09-27, 11 tasks + control leg): 0 byte/bit mismatches; `USD_LAST` sleeping-body staleness on 5 tasks; CUDA 700 on both laptop tasks with `USD_LAST` on (docs/validation.md)
- [x] Laptop replay A/B (2026-09-27, 6 replays): physics bit-identical with `USD_LAST` on/off (only the laptop links' USD matrices differ), scores identical on 10 layouts, CUDA 700 not reproduced (0/6)
- [ ] Closed-loop CUDA 700 attribution on `store_laptop_and_headphones*`: 2-3 runs per arm, speedup preset, `USD_LAST=0` vs `1`, same commit
- [ ] Later: write back bodies that fell asleep during an action (PhysX re-reads external USD writes as teleports: must keep replay bit-identical)
- [ ] franka `panda_link1` rotation-only difference: one control leg
- [x] `stack_blocks` replay with this code (`scripts/verify_physics.sh`, 2026-09-27): every switch off x2 vs speedup preset, 5,208/5,208 PhysX states bit-identical
- [x] Headline with this code (`scripts/reproduce_benchmark.sh`, 2026-09-27): upstream 636 s, harness 546 s, speedup 299 s (2.13x vs upstream), official 25 layouts, 2 alternating reps, same machine
- [x] Re-check with the current identifiers (2026-09-28): patched trees identical to the 2026-09-27 build apart from identifier names (16/16 files); `stack_blocks` layouts 0-9 speedup closed loop 115 s, rc=0, every switch counter as on 2026-09-27 (a first run took 215 s while a parallel compile loaded the host CPU: time only on an idle host)
- [x] `VIDEO_ONLY_OBS=1` vs `0` chunk-start frames (`RDTURBO_OBS_DUMP`, 2026-09-27): within the render-noise floor on all three cameras
- [x] Pooled paired score comparison over four tasks (2026-09-28, 125 paired layouts): +0.7 points, 95% CI [-4.2, +5.5]
- [x] Two more runs per arm on each of the four tasks with this code (2026-09-29, 0.1.0a1): all 13 pairs, 325 paired
  layouts, +0.2 points, 95% CI [-3.0, +3.4], official 39/325 and Turbo 40/325 successes, MDE80 4.6 points (6.0 without
  the conveyor task, which had no successes; per task 8-14)
- [ ] optional: more runs per arm where a task scores near zero (conveyor: 0 successes in 75 layouts per arm)
- [x] Speedup on 4 tasks x 25 official layouts (2026-09-28, one run per arm): rigid 2.13x, second robot 2.37x, conveyor 2.10x, fluid `_random` 1.36x
- [x] Re-timed with two runs per arm and the same JAX compilation cache in both arms (2026-09-29, 0.1.0a1): 2.15x, 2.30x, 2.05x, 1.34x (README)
- [ ] Timing on an articulated-object task and a cloth task
- [ ] optional: ABAB repeats; JAX cold vs warm compile cache bitwise

## Upstream watch (checked 2026-09-28)

- [ ] RoboDojo PR #61 changes a pinned file (`tiled_capture_manager.py`); when it merges: update `pins.json`, run
  `apply --dry-run`, then `scripts/verify_physics.sh` and `scripts/reproduce_benchmark.sh` on the new commit
- [ ] RoboDojo PRs #58-#60 and #62 change reset behaviour (not our pinned files); after they merge, re-run the timing
  table on the new commit (docs/prior-work.md, "Upstream fixes in flight")
- [ ] `store_laptop_and_headphones_random` second-batch hang vs #58 (articulation respawn): unverified

## Before a public release

- [x] Numbers in README/docs come from runs of this code, except the predecessor results, which are labelled as such (docs/validation.md lists what this code re-covered)
- [x] The 100 extra layouts are not shipped; the predecessor's 125-layout numbers stay, labelled as not reproducible from this repository
- [x] Pi_05-only for chunk-start upload, `VIDEO_ONLY_OBS` and `USD_LAST` (documented limitation); other switches are policy-independent
- [x] Every injected block read in a patched tree (2026-09-29; 3 reviewers + 3 adversarial verifiers): no high findings; 1 medium and 12 low confirmed and fixed before the first public commit. These fixes change edge cases only (invalid values, verification tooling, logging, non-root pipe size). The 2026-09-29 closed-loop benchmark (speedup preset, 4 tasks, 16 legs, all rc=0) ran the fixed Pi_05 loop with valid switch values; the verification-tooling fixes (replay, check modes) have not been re-run on a GPU yet
- [ ] Name, license and trademark check (similar names exist in the RoboDojo ecosystem; check GitHub and PyPI); README states "unofficial" prominently
- [ ] RoboDojo licence question: MIT LICENSE file vs non-commercial wording in its README (disclosed in README and THIRD_PARTY_NOTICES; keep this item blocking)
- [x] Private preview on GitHub: history squashed into one commit, noreply author only, no tag yet (2026-09-28), clone URL, dated CHANGELOG, `[project.urls]`
- [ ] Before making the repository public: enable GitHub's "keep my email private" / "block pushes that expose my email"; re-check the name
