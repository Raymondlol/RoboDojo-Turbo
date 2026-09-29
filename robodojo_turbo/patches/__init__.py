"""Patch registry. Order in REGISTRY is application order; `requires` are closed over automatically."""
from __future__ import annotations

from ..engine import Patch
from . import harness as H
from . import ops as O
from . import speedup as S

OBS_MANAGER = "env/observation_manager/obs_manager.py"
CONTROL_MANAGER = H.CONTROL_MANAGER
DIRECT_RL_ENV = "env/environment/isaac/direct_rl_env.py"

REGISTRY = [
    # --- harness -------------------------------------------------------------------------------------------------------
    Patch("trace", "harness", (H.EVAL_ENV,), H.patch_trace, installs=("utils/rdturbo_trace.py",),
          switches=("RDTURBO_TRACE", "RDTURBO_WORKER"), doc="span tracer + bubble report input"),
    Patch("layout_shards", "harness", (H.SEED_MANAGER, H.EVAL_POLICY_SH, H.MAIN), H.patch_layout_shards, requires=("trace",),
          switches=("RDTURBO_LAYOUT_IDS", "RDTURBO_NUM_ENVS", "RDTURBO_EVAL_NUM_FORCE"), doc="evaluate a layout subset per process"),
    Patch("pi05_loop", "harness", (H.PI05_DEPLOY,), H.patch_pi05_loop,
          switches=("RDTURBO_OBS_EVERY_STEP", "RDTURBO_VIDEO_EVERY", "RDTURBO_RECORD_ACTIONS", "RDTURBO_REPLAY_ACTIONS", "RDTURBO_REPLAY_STRICT",
                    "RDTURBO_STATE_TRACE", "RDTURBO_PARTICLE_TRACE", "RDTURBO_OBS_DUMP"),
          doc="Pi_05 batch loop: chunk-start observation upload, video subsampling, record/replay and state tracing"),
    Patch("fast", "harness", (H.ROBOT_MANAGER, H.CONTROL_MANAGER, H.FUNC_PARSER, H.EVAL_ENV, H.SAVE_FILE), H.patch_fast,
          switches=("RDTURBO_FAST", "RDTURBO_VIDEO_CAMS"), doc="lossless copy/recompute removal"),
    Patch("ffmpeg_threads", "harness", (H.SAVE_FILE,), H.patch_ffmpeg_threads, switches=("RDTURBO_FFMPEG_THREADS",),
          doc="cap x264 threads per video stream"),
    Patch("accounting", "harness", (H.MAIN,), H.patch_accounting, switches=("RDTURBO_STATUS", "RDTURBO_STRICT"),
          doc="count batches swallowed by the generic except; optional non-zero exit"),
    # --- speedup switches ----------------------------------------------------------------------------------------------
    Patch("revoke_listeners", "speedup", (H.MAIN,), S.patch_revoke_listeners, installs=("utils/rdturbo_revoke_listeners.py",),
          switches=("RDTURBO_REVOKE_LISTENERS",)),
    Patch("video_only", "speedup", (H.EVAL_ENV,), S.patch_video_only, requires=("pi05_loop",),
          switches=("RDTURBO_VIDEO_ONLY_OBS",), api=((OBS_MANAGER, r"def render_for_capture\("),)),
    Patch("rgb3", "speedup", (S.CAMERA_VIEW, S.TILED_CAPTURE), S.patch_rgb3, switches=("RDTURBO_RGB3",)),
    Patch("usd_last", "speedup", (H.EVAL_ENV,), S.patch_usd_last, requires=("pi05_loop",),
          switches=("RDTURBO_USD_LAST", "RDTURBO_USD_LAST_MODE"),
          api=((CONTROL_MANAGER, r"self\.control_queue = "), (CONTROL_MANAGER, r"def is_empty\(self\)"))),
    Patch("gc_freeze", "speedup", (H.MAIN,), S.patch_gc_freeze, switches=("RDTURBO_GC",)),
    Patch("ctrl_cache", "speedup", (H.CONTROL_MANAGER, H.ROBOT_MANAGER), S.patch_ctrl_cache, switches=("RDTURBO_CTRL_CACHE",),
          api=((DIRECT_RL_ENV, r"self\._sim_step_counter \+= 1"),)),
    Patch("parent_cache", "speedup", (S.RIGID,), S.patch_parent_cache, switches=("RDTURBO_PARENT_CACHE",)),
    Patch("video_mv", "speedup", (H.SAVE_FILE,), S.patch_video_mv, switches=("RDTURBO_VIDEO_MV",)),
    Patch("pci_copy", "speedup", (H.EVAL_ENV,), S.patch_pci_copy, switches=("RDTURBO_PCI_COPY",)),
    Patch("batch_tensor", "speedup", (H.ROBOT_MANAGER,), S.patch_batch_tensor, requires=("ctrl_cache",),
          switches=("RDTURBO_BATCH_TENSOR",)),
    # --- ops -----------------------------------------------------------------------------------------------------------
    Patch("no_planner", "ops", (H.ROBOT_MANAGER,), O.patch_no_planner, switches=("RDTURBO_NO_PLANNER",)),
    Patch("offline_assets", "ops", (H.MAIN,), O.patch_offline_assets, installs=("utils/rdturbo_offline_assets.py",),
          switches=("RDTURBO_OFFLINE_ASSETS", "RDTURBO_NV_MIRROR")),
    Patch("server_memfrac", "ops", (O.PI05_SERVER_SH,), O.patch_server_memfrac, switches=("RDTURBO_SERVER_MEM_FRACTION",)),
]

PROFILES = {
    "harness": {"harness"},
    "speedup": {"harness", "speedup"},
    "all": {"harness", "speedup", "ops"},
}
