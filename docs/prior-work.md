# Prior work and positioning

Checked 2026-09-27 (upstream RoboDojo issues/PRs, 79 fork refs, 12 non-fork derivatives, Chinese-language sources,
Isaac Sim / Isaac Lab docs, issues and forums, evaluation frameworks and papers). No third-party end-to-end speedup of
RoboDojo's evaluation client was found. Most individual techniques are known:

| piece | closest prior work |
|---|---|
| video-only observation steps | IsaacLab-Arena PR #1312 skips non-RGB cameras in the video recorder (a modality filter, not a step filter); RLinf's on-demand observation also stops rendering intermediate steps, which RoboDojo's protocol does not allow |
| upload observations only at chunk starts | openpi `ActionChunkBroker` / remote-inference docs; XPolicyLab `GigaWorldPolicy/deploy.py` (2026-07-06) keeps per-step `get_obs` for video without uploading |
| layout sharding | three independent implementations (forks by LIO-H-ZEN and Synapath, a standalone repository by Aero-san); vla-eval shards episodes (arXiv 2603.13966) |
| faster video writing (`memoryview`, x264 preset/threads) | RoboDojo draft PR #16 (async writer, benchmark harness; unmerged); CPhoenixW fork (NVENC, Isaac reuse) |
| deferred PhysX -> USD write-back | NVIDIA forum thread 164502 (moderator, 2023); Isaac Lab's Fabric path flushes at render; IsaacLab PR #7642 (2026-09-15); IsaacSim #819 (2026-09-10) profiles the write-back cost. Inside RoboDojo: the Synapath/ext-RoboDojo fork (commit c968e5d, 2026-09-10) already calls `update_transformations(False, True, True, False)` on RoboDojo's USD path, for render correctness. IsaacLab #6609 (2026-07-19) documents stale camera transforms after reset (the hazard behind the reset carve-out). RoboDojo cannot use Fabric: scorers and cameras read USD, and Fabric does not support PBD particles |
| revoking UI listeners | IsaacSim #819 profiles extension listener costs; OmniGibson revokes the viewport-menubar UsdWatcher in process (for teardown correctness); RLinf's BEHAVIOR "feature slimming" disables Omniverse viewports and caps threads in headless runs. Intercepting `Tf.Notice.Register` as a performance fix: not found |
| 3-channel RGB | upstream Isaac Sim `CameraView`; Isaac Lab (H, W, 3) since 2024 |
| skip the planner for joint actions | vla-eval RoboDojo integration (PR #101, 2026-07-16) |
| offline NVIDIA assets | `omni.client` aliases (NVIDIA docs use the same S3 host as example); IsaacLab #4478/#4479 |
| `gc.freeze` after start-up | CPython (Instagram); sglang-omni #1940 observed the same "no gain once the garbage source is gone" |
| end-to-end simulator-side eval speedup | RLinf BEHAVIOR optimisation (25x; slims the app, renders only chunk-final actions, pipelines, adjusts physics frequency; no score-equivalence check); BEHAVIOR-1K PR #2329 (`--render-every`) |
| bitwise acceptance of a performance patch | newton-physics PR #3995 (2026-09-10: bitwise trajectory match); robomimic `playback_dataset.py` (exact state equality on replay); Box2D/Factorio determinism tests; E3SM bit-for-bit policy |
| trace-level reproducibility and its limits | arXiv 2606.04233 (App. B: bitwise closed-loop trace comparison; argues open-loop action replay is not sufficient evidence for closed-loop claims) |
| dual-path checks, noise-floor reading | GitHub Scientist; Twitter Diffy |
| common random numbers for paired policy comparison | PEGASUS (Ng & Jordan 2000); Strens & Moore 2001 |
| RoboDojo's own speedup | heterogeneous parallel simulation (arXiv 2607.04434, Table 4: 1.63x with pi0.5), already in upstream; orthogonal to this project |

What this project contributes, stated narrowly:

1. A RoboDojo-specific set of host-side speedups measured end to end under an unchanged protocol (every frame rendered,
   same physics and dt; the recommended preset records only the head-camera video), in a reversible, hash-pinned patcher.
2. Revocation of UI-only USD listeners by intercepting their re-registration, traced to RoboDojo launching the GUI
   experience file headless.
3. Episode-level USD write-back control (the per-action flush call itself is used by the Synapath fork) with a reset
   carve-out, a last-substep mode for particle/cloth scenes, and two findings about its limits: the per-action flush
   silently breaks fluid and garment scoring, and bodies that fall asleep mid-action are left stale in USD
   (5 of 11 tasks checked, up to 1.2 mm; unresolved).
4. The verification procedure as a whole: open-loop bitwise replay keyed per (layout, chunk), byte-exact dual paths on
   the same render, and paired closed-loop comparison against a same-configuration noise floor.

## Upstream fixes in flight (checked 2026-09-28)

Open RoboDojo pull requests against 726e9aa that touch the reset path. None was merged when checked; all were marked
cleanly mergeable. #58-#62 come from running RoboDojo through AllenAI's vla-evaluation-harness (one environment, a
scene reset before every episode, CPU physics config, `num_envs = 1`, A100).

| PR | what it fixes | touches a pinned file? | relevance here |
|---|---|---|---|
| #61 | `TiledCaptureManager.reset()` rebuilds every camera view and render product on each reset without releasing the old ones; the PR reports renders growing from 40 ms to 274 ms by the sixth layout | **yes**: `env/camera_manager/capture/tiled_capture_manager.py` (adds a set-up flag and an early return in `reset()`; the RGB3 anchor in `init_cameras()` is untouched) | once merged, `apply` refuses the new file until the pin is updated; `apply --dry-run` in CI will flag it. On the native `eval_policy.sh` path (10 environments per batch) no growth was seen: every switch off, `stack_blocks`, 125 layouts (25 official + 100 extra, traced with this project's private predecessor, 2026-09-24) in 13 batches, render 84.8 ms in the first batch and 79.8-82.2 ms in the other 12, reset 21.4 s then 17.3-18.0 s; the 25-layout runs behind the timing table show the same over 3 batches |
| #60 | the previous room is deleted under the wrong path, so rooms stack up on every reset | no | upstream's own reset-time fix; reset times above do not grow |
| #59 | deleted objects stay in place as invisible colliders (child collision shapes stay enabled) on RoboDojo's CPU-config path | no | RoboDojo's default config says `device: cpu` (PhysX is switched to GPU dynamics later), so this branch is likely taken on the native path too; affects later batches equally with and without this project's switches; not checked here |
| #58 | articulations are not respawned across layouts, so later layouts fail `check_layout_stability`; the PR names `store_laptop_and_headphones_random` as needing respawning it does not add | no | possibly related to the second-batch hang seen on that task in the multi-task check run (docs/validation.md); unverified |
| #62 | three language tasks keep the first layout's instruction | no | none for timing; affects scores of those tasks equally in all arms |
| #48 | articulation reset does not push restored drive targets to PhysX (`swap_blocks` buttons) | no | same |
| #63 | checkpoint name aliases in `scripts/RoboDojo/download_ckpt.sh` | no | none |

Older open PRs: #16 (async video writer, listed above) and #26 (driver preflight). #60 and #61 are upstream fixes for
reset-path costs that grow with the number of resets in one process; they are complementary to this project's switches,
which leave reset handling to upstream except for the USD write-back carve-out.
