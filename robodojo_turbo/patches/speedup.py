"""Speedup switches (profile "speedup"). Every switch is off unless its RDTURBO_* variable is set. Six accept =check,
which runs the new and the upstream code path on the same data and counts byte/bit mismatches (which result the run
continues with differs per switch); USD_LAST=check compares USD with PhysX. See docs/switches.md for mechanics, evidence
and scope.
"""
from __future__ import annotations

import os

from .. import MARK
from ..engine import PatchError, Tree, find_once, indent_block, sub_once
from .harness import CONTROL_MANAGER, EVAL_ENV, MAIN, ROBOT_MANAGER, RUNTIME, SAVE_FILE

CAMERA_VIEW = "env/camera_manager/capture/camera_view.py"
TILED_CAPTURE = "env/camera_manager/capture/tiled_capture_manager.py"
RIGID = "env/scene_manager/objects/rigid.py"
APP_ANCHOR = r"^([ \t]*)simulation_app = app_launcher\.app[ \t]*$"


# ---------------------------------------------------------------------------------------------------------------------
# revoke_listeners

def patch_revoke_listeners(tree: Tree) -> str:
    tree.install(os.path.join(RUNTIME, "rdturbo_revoke_listeners.py"), "utils/rdturbo_revoke_listeners.py")
    s = tree.read(MAIN)
    s = sub_once(s, APP_ANCHOR,
                 lambda m: (f"{m.group(0)}\n{m.group(1)}if (os.environ.get('RDTURBO_REVOKE_LISTENERS') or '').strip() not in ('', '0'):  # {MARK} revoke-listeners\n"
                            f"{m.group(1)}    from utils.rdturbo_revoke_listeners import install as _rdturbo_revoke_install\n"
                            f"{m.group(1)}    _rdturbo_revoke_install(os.environ['RDTURBO_REVOKE_LISTENERS'].strip())"),
                 "main.py AppLauncher (revoke-listeners)")
    tree.write(MAIN, s)
    return "RDTURBO_REVOKE_LISTENERS=1"


# ---------------------------------------------------------------------------------------------------------------------
# video_only: on steps whose observation is only used for the video, skip assembling the full observation

_VO_BOOT = [
    "import os as _rdturbo_vo_os, atexit as _rdturbo_vo_ax  # " + MARK + " video-only",
    "_RDTURBO_VO = (_rdturbo_vo_os.environ.get('RDTURBO_VIDEO_ONLY_OBS') or '0').strip()",
    "_RDTURBO_VO_STAT = {'n': 0, 'frames': 0, 'bad': 0, 'refresh': 0}",
    "_rdturbo_vo_ax.register(lambda: _RDTURBO_VO == 'check' and print('[robodojo-turbo] video-only check', _RDTURBO_VO_STAT, flush=True))",
]

_VO_METHODS = '''
def _rdturbo_vo_frames(self, env_idx_list):  # [robodojo-turbo] video-only: same capture_manager.step and [:, :, :3] slice as ObsManager.get_obs, recording cameras only
    om = self.obs_manager
    cm, cap = om.camera_manager, om.capture_manager
    fr = {e: {"vision": {}} for e in env_idx_list}
    if cm is None or cap is None:
        return fr
    vc = globals().get("_RDTURBO_VIDEO_CAMS") or set()
    ids = [i for i in range(cm.num_cams) if not vc or any(cm.camera_names[e][i] in vc for e in env_idx_list)]
    if not ids:
        return fr
    data = cap.step(env_ids=env_idx_list, cam_ids=ids)
    for k, ith in enumerate(ids):
        env_list = data[k].get("rgb")
        if env_list is None:
            continue
        for idx, e in enumerate(env_idx_list):
            fr[e]["vision"][cm.camera_names[e][ith]] = {"color": env_list[idx]["data"][:, :, :3]}
    return fr

def _rdturbo_vo_fp(self):  # fingerprint of video/scoring bookkeeping + IsaacLab lazy-buffer timestamps (did the old path refresh them?)
    lazy = tuple(tuple(getattr(getattr(k.data, n, None), "timestamp", None) for n in ("_joint_pos", "_body_link_pose_w"))
                 for k in (getattr(self.robot_manager, "robot_key", None) or []))
    w = repr({e: {c: getattr(x, "n_frames", None) for c, x in ws.items()} for e, ws in self.video_writers.items()})
    return (tuple(self.end_flag), tuple(self.success), tuple(self.take_action_cnt), w), lazy

def _rdturbo_video_only_batch(self, env_idx_list, last_frame=False):
    if self.physx_monitor_enabled:
        self._check_physx_broken_envs()
    if env_idx_list is None:
        env_idx_list = list(range(self.num_envs))
    if self.physx_monitor_enabled:
        self._check_endpose_finite(env_idx_list)
    self.obs_manager.render_for_capture()
    fr = self._rdturbo_vo_frames(env_idx_list)
    if _RDTURBO_VO == "check":
        self._rdturbo_vo_check(env_idx_list, fr, last_frame)
    for e in env_idx_list:
        if not self.end_flag[e] or last_frame:
            self._stream_vision(e, fr[e])
    return None

def _rdturbo_vo_check(self, env_idx_list, fr, last_frame):  # on the same render, run the upstream path too; compare recorded frames byte for byte
    st, (s0, l0) = _RDTURBO_VO_STAT, self._rdturbo_vo_fp()
    data = self.obs_manager.get_obs(env_idx_list=env_idx_list)
    s1, l1 = self._rdturbo_vo_fp()
    vc = globals().get("_RDTURBO_VIDEO_CAMS") or set()
    def pick(v):
        o = []
        for cam, cd in (v or {}).items():
            if vc and cam not in vc:
                continue
            c = cd.get("color") if isinstance(cd, dict) else None
            if c is not None:
                c = np.ascontiguousarray(c)
                if c.ndim == 3 and c.shape[2] in (3, 4):
                    o.append((cam, c))
        return o
    bad = [] if s0 == s1 else ["state"]
    st["refresh"] += int(l0 != l1)
    for e in env_idx_list:
        if self.end_flag[e] and not last_frame:
            continue
        a, b = pick(data[e].get("vision")), pick(fr[e].get("vision"))
        if [k for k, _ in a] != [k for k, _ in b] or any(x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y) for (_, x), (_, y) in zip(a, b)):
            bad.append(e)
        st["frames"] += len(b)
    if (_rdturbo_vo_os.environ.get("RDTURBO_RGB3") or "0").strip() == "1":   # RGB3 check: read the same frame through the upstream 4-channel path and compare [..., :3]
        cm, cap = self.obs_manager.camera_manager, self.obs_manager.capture_manager
        for ith in range(cm.num_cams):
            names = {cm.camera_names[e][ith] for e in env_idx_list}
            if vc and not (names & vc):
                continue
            ref, _ = cap.tiled_cameras[ith].get_data("rgb")
            ref = ref.numpy()
            st["rgb3_ch"] = int(ref.shape[-1])
            for e in env_idx_list:
                cam = cm.camera_names[e][ith]
                got = fr[e]["vision"].get(cam, {}).get("color")
                if got is None or not np.array_equal(np.ascontiguousarray(ref[e][:, :, :3]), np.ascontiguousarray(got)):
                    bad.append(("rgb3", e, cam))
    st["n"] += 1
    if bad:
        st["bad"] += 1
        print(f"[robodojo-turbo] video-only CHECK MISMATCH {bad} {st}", flush=True)
'''


def patch_video_only(tree: Tree) -> str:
    s = tree.read(EVAL_ENV)
    m1 = find_once(s, r"^([ \t]*)def get_obs_batch\(self, env_idx_list=None, last_frame=False\):[ \t]*$", "eval_env get_obs_batch")
    m2 = find_once(s, r"^([ \t]*)def _stream_vision\(self, env_idx, frame\):[ \t]*$", "eval_env _stream_vision")
    if m2.start() < m1.start():
        raise PatchError("eval_env: _stream_vision is expected after get_obs_batch")
    i1, i2 = m1.group(1), m2.group(1)
    s = s[: m2.start()] + indent_block(_VO_METHODS, i2) + "\n" + s[m2.start():]   # edit the later anchor first
    s = s[: m1.start()] + (f"{i1}def get_obs_batch(self, env_idx_list=None, last_frame=False, rdturbo_video_only=False):  # {MARK} video-only\n"
                           f"{i1}    if rdturbo_video_only and _RDTURBO_VO in ('1', 'check'):\n"
                           f"{i1}        return self._rdturbo_video_only_batch(env_idx_list, last_frame)") + s[m1.end():]
    tree.write(EVAL_ENV, "\n".join(_VO_BOOT) + "\n" + s)
    return "RDTURBO_VIDEO_ONLY_OBS=1|check (effective only with the Pi_05 loop and RDTURBO_OBS_EVERY_STEP=0)"


# ---------------------------------------------------------------------------------------------------------------------
# rgb3: emit 3-channel RGB from the tiled-reshape kernel (restores upstream Isaac Sim CameraView behaviour)

def patch_rgb3(tree: Tree) -> str:
    s = tree.read(CAMERA_VIEW)
    s = sub_once(s, r"^([ \t]*)output_channels = channels  # Use channels directly \(rgb has 4 channels from rgba\)[ \t]*$",
                 r"\g<1>output_channels = 3 if (annotator_type == 'rgb' and out is not None and tuple(out.shape)[-1] == 3) else channels  # " + MARK + " rgb3",
                 "camera_view output_channels")
    tree.write(CAMERA_VIEW, s)
    s = tree.read(TILED_CAPTURE)
    s = sub_once(s, r'^([ \t]*)channels = spec\["channels"\][ \t]*$',
                 r"\g<1>channels = 3 if (annotator_name == 'rgb' and (__import__('os').environ.get('RDTURBO_RGB3') or '0').strip() == '1') else spec['channels']  # " + MARK + " rgb3",
                 "tiled_capture_manager channels")
    tree.write(TILED_CAPTURE, s)
    return "RDTURBO_RGB3=1"


# ---------------------------------------------------------------------------------------------------------------------
# usd_last: stop per-substep PhysX->USD write-back inside an episode

_UL_BOOT = r'''import os as _rdturbo_ul_os  # [robodojo-turbo] usd-last
_RDTURBO_UL = (_rdturbo_ul_os.environ.get("RDTURBO_USD_LAST") or "0").strip()   # 0 | 1 | check | basecheck
# With all four keys false omni.physx also skips the USD->Fabric sync in InternalScene::updateSimulationOutputs.
_RDTURBO_UL_KEYS = ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd", "/physics/updateParticlesToUsd", "/physics/updateResidualsToUsd")
_RDTURBO_UL_LOCAL = "/physics/outputVelocitiesLocalSpace"
_RDTURBO_UL_PKEY = "/physics/updateParticlesToUsd"
# How the episode is written back to USD (RDTURBO_USD_LAST_MODE):
#   flush          = all write-back off in every substep; after each action update_transformations writes rigid bodies and
#                    articulation links once. It does NOT write fluid particles or cloth points: on fluid/garment tasks
#                    those freeze in USD and the scorers, which read USD, fail every episode (measured 2026-09-27).
#   last           = write-back off in substeps 1..n-1 and restored for the last substep of each action (the substep after
#                    which some env's control queue is empty), so that substep writes everything, particles and cloth included.
#   keep-particles = flush, but leave updateParticlesToUsd on (fixes fluids, not cloth; kept for experiments).
#   auto (default) = last if the scene contains fluid or garment objects, otherwise flush (the path validated on rigid tasks).
_RDTURBO_UL_MODE = (_rdturbo_ul_os.environ.get("RDTURBO_USD_LAST_MODE") or "auto").strip()
if _RDTURBO_UL in ("1", "check") and _RDTURBO_UL_MODE not in ("auto", "flush", "last", "keep-particles"):
    # fail at import, not per batch (RoboDojo's generic except would swallow every batch after its reset)
    print(f"[robodojo-turbo] RDTURBO_USD_LAST_MODE={_RDTURBO_UL_MODE!r} is not one of auto|flush|last|keep-particles", flush=True)
    _rdturbo_ul_os._exit(2)
_RDTURBO_UL_STATE = {"snap": None, "mode": None}
_RDTURBO_UL_ST = {"episodes": 0, "flush": 0, "last_on": 0, "chk": 0, "mode": None, "by_cat": {}, "stale": 0, "snap": None}


def _rdturbo_ul_has_particles(env):
    # In RoboDojo only Fluid and Garment objects use PhysX particle systems.
    if env is None:
        return True
    by_env = env.scene_manager.layout_manager.instance_type_by_env   # list indexed by env: {instance: type}
    per_envs = by_env.values() if isinstance(by_env, dict) else by_env
    return any(t in ("fluid", "garment") for per_env in per_envs if per_env for t in per_env.values())


def _rdturbo_ul_episode_begin(env=None):
    # Called by the Pi_05 loop after reset/setup_scene: switch write-back off for the episode.
    if _RDTURBO_UL not in ("1", "check") or _RDTURBO_UL_STATE["snap"] is not None:
        return
    mode = _RDTURBO_UL_MODE
    if mode == "auto":
        mode = "last" if _rdturbo_ul_has_particles(env) else "flush"
    if mode not in ("flush", "last", "keep-particles"):
        raise ValueError(f"RDTURBO_USD_LAST_MODE={_RDTURBO_UL_MODE!r} (expected auto|flush|last|keep-particles)")
    import carb
    s = carb.settings.get_settings()
    snap = {k: bool(s.get_as_bool(k)) for k in _RDTURBO_UL_KEYS + (_RDTURBO_UL_LOCAL,)}
    _RDTURBO_UL_ST["snap"] = snap
    if not snap["/physics/updateToUsd"]:
        return   # the stage does not write back to USD at all (e.g. Fabric mode): nothing to do
    _RDTURBO_UL_STATE.update(snap=snap, mode=mode)
    _RDTURBO_UL_ST["mode"] = mode
    for k in _RDTURBO_UL_KEYS:
        if k == _RDTURBO_UL_PKEY and mode == "keep-particles":
            continue
        s.set_bool(k, False)
    _RDTURBO_UL_ST["episodes"] += 1


def _rdturbo_ul_episode_end():
    # Episode end, exception, or the next reset(): restore the settings (reset/stability checks write back every substep).
    snap = _RDTURBO_UL_STATE["snap"]
    if snap is None:
        return
    import carb
    s = carb.settings.get_settings()
    if _RDTURBO_UL_STATE["mode"] != "last":
        _rdturbo_ul_flush(snap)
    for k in _RDTURBO_UL_KEYS:
        s.set_bool(k, snap[k])
    _RDTURBO_UL_STATE.update(snap=None, mode=None)


def _rdturbo_ul_flush(snap):
    # Same InternalScene::updateSimulationOutputs as the per-substep write-back, restricted to rigid bodies/links.
    from omni.physx import get_physx_interface
    get_physx_interface().update_transformations(False, snap["/physics/updateToUsd"], snap["/physics/updateVelocitiesToUsd"], snap[_RDTURBO_UL_LOCAL])
    _RDTURBO_UL_ST["flush"] += 1


def _rdturbo_ul_substep_pre(env, env_idx_list):
    # last mode, inside EvalEnv.step after pop() and before sim_step(): if some env's control queue is now empty, the
    # substep loop ends after this substep, so restore write-back for it. Reads the queues directly: control_manager.get_empty
    # would also drain the queues of envs outside env_idx_list.
    st = _RDTURBO_UL_STATE
    if st["snap"] is None or st["mode"] != "last":
        return False
    cq = env.robot_manager.control_manager.control_queue
    if not any(cq[i].is_empty() for i in env_idx_list):
        return False
    import carb
    s = carb.settings.get_settings()
    for k in _RDTURBO_UL_KEYS:
        s.set_bool(k, st["snap"][k])
    _RDTURBO_UL_ST["last_on"] += 1
    return True


def _rdturbo_ul_substep_post(on):
    if on:
        import carb
        s = carb.settings.get_settings()
        for k in _RDTURBO_UL_KEYS:
            s.set_bool(k, False)


def _rdturbo_ul_after_action(env, env_idx_list):
    # After each action's substeps, before scoring/observation/render: flush (not needed in last mode); compare in check modes.
    snap = _RDTURBO_UL_STATE["snap"]
    if snap is not None and _RDTURBO_UL_STATE["mode"] != "last":
        _rdturbo_ul_flush(snap)
    if _RDTURBO_UL in ("check", "basecheck"):
        _rdturbo_ul_compare(env, env_idx_list)


def _rdturbo_ul_compare(env, env_idx_list):
    # USD world pose (what rendering and USD-reading scorers see) vs the PhysX pose, per category: robot links,
    # scene rigid objects, and every rigid-body link of articulated scene objects.
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    from omni.physx import get_physx_interface
    px = get_physx_interface()
    stage = omni.usd.get_context().get_stage()
    items = []
    for key in env.robot_manager.robot_key:
        lp = key.root_physx_view.link_paths
        for e in env_idx_list:
            items.extend(("robot", p) for p in lp[e])
    lm = env.scene_manager.layout_manager
    for e in env_idx_list:
        for inst, typ in lm.instance_type_by_env[e].items():
            if typ not in ("rigid", "articulation"):
                continue   # rooms, lights, ground, tables: get_scene_object would print "not found" for each
            obj = lm.get_scene_object(e, inst)
            p = getattr(obj, "_prim_path", None)
            if not p:
                continue
            if typ == "rigid":
                items.append(("rigid", p))
            elif typ == "articulation":
                for prim in Usd.PrimRange(stage.GetPrimAtPath(p)):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        items.append(("articulation", str(prim.GetPath())))
    st = _RDTURBO_UL_ST
    st["chk"] += 1
    for cat, p in items:
        r = px.get_rigidbody_transformation(p)
        if not r.get("ret_val"):
            continue
        m = UsdGeom.Xformable(stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        pos, rot = r["position"], r["rotation"]   # rotation = (x, y, z, w)
        dpos = max(abs(t[0] - pos[0]), abs(t[1] - pos[1]), abs(t[2] - pos[2]))
        qi = q.GetImaginary()
        a = (q.GetReal(), qi[0], qi[1], qi[2])
        b = (rot[3], rot[0], rot[1], rot[2])
        sg = 1.0 if sum(x * y for x, y in zip(a, b)) >= 0 else -1.0
        drot = max(abs(x - sg * y) for x, y in zip(a, b))   # component-wise quaternion difference (sign aligned)
        c = st["by_cat"].setdefault(cat, {"n": 0, "stale": 0, "max_dpos": 0.0, "max_drot": 0.0})
        c["n"] += 1
        c["max_dpos"] = max(c["max_dpos"], dpos)
        c["max_drot"] = max(c["max_drot"], drot)
        if dpos > 1e-5 or drot > 1e-5:
            c["stale"] += 1
            st["stale"] += 1
            if st["stale"] <= 5:
                print(f"[robodojo-turbo] usd-last STALE {cat} {p} dpos={dpos:.3g} drot={drot:.3g}", flush=True)
'''


def patch_usd_last(tree: Tree) -> str:
    s = tree.read(EVAL_ENV)
    s = sub_once(s, r"^([ \t]*)self\.reward_manager\.step\(env_idx_list=env_idx_list\)[ \t]*$",
                 lambda m: f"{m.group(1)}_rdturbo_ul_after_action(self, env_idx_list)  # {MARK} usd-last\n{m.group(0)}",
                 "eval_env reward_manager.step")
    s = sub_once(s, r"^([ \t]*)super\(\)\.reset\(seed=self\.env_seeds, options=options\)[ \t]*$",
                 lambda m: f"{m.group(1)}_rdturbo_ul_episode_end()  # {MARK} usd-last: reset writes back every substep as upstream\n{m.group(0)}",
                 "eval_env super().reset")
    s = sub_once(s, r"^([ \t]*)super\(\)\.step\(meta_control_list=meta_control_list\)[ \t]*\n\1self\.sim_step\(render=False\)[ \t]*$",
                 lambda m: (f"{m.group(1)}super().step(meta_control_list=meta_control_list)\n"
                            f"{m.group(1)}_rdturbo_ul_on = _rdturbo_ul_substep_pre(self, env_idx_list)  # {MARK} usd-last last-substep\n"
                            f"{m.group(1)}self.sim_step(render=False)\n"
                            f"{m.group(1)}_rdturbo_ul_substep_post(_rdturbo_ul_on)  # {MARK} usd-last last-substep"),
                 "eval_env EvalEnv.step substep")
    tree.write(EVAL_ENV, _UL_BOOT + s)
    return "RDTURBO_USD_LAST=1|check|basecheck, RDTURBO_USD_LAST_MODE=auto|flush|last|keep-particles (episode hooks live in the Pi_05 loop)"


# ---------------------------------------------------------------------------------------------------------------------
# gc_freeze

_GC_HELPERS = r'''
import gc as _rdturbo_gc, time as _rdturbo_gc_time  # [robodojo-turbo] gc
_RDTURBO_GC = set(filter(None, (os.environ.get("RDTURBO_GC") or "").strip().split(",")))   # freeze,log; empty = upstream behaviour
_RDTURBO_GC_ST = {"full": 0, "full_ms": 0.0, "max_ms": 0.0, "t": 0.0}


def _rdturbo_gc_cb(ph, info):
    if ph == "start":
        _RDTURBO_GC_ST["t"] = _rdturbo_gc_time.perf_counter()
    elif info.get("generation") == 2:
        ms = (_rdturbo_gc_time.perf_counter() - _RDTURBO_GC_ST["t"]) * 1e3
        _RDTURBO_GC_ST["full"] += 1
        _RDTURBO_GC_ST["full_ms"] += ms
        _RDTURBO_GC_ST["max_ms"] = max(_RDTURBO_GC_ST["max_ms"], ms)


if "log" in _RDTURBO_GC:
    _rdturbo_gc.callbacks.append(_rdturbo_gc_cb)
if "freeze" in _RDTURBO_GC:
    _rdturbo_gc.freeze()   # long-lived Kit/extension objects: full collections during scene setup only scan new objects


def _rdturbo_gc_pre(where):
    # before every batch reset and env.close(): report the episode's full collections, then unfreeze so the batch's garbage can go
    if "log" in _RDTURBO_GC:
        print(f"[robodojo-turbo] gc {where}: full={_RDTURBO_GC_ST['full']} full_ms={_RDTURBO_GC_ST['full_ms']:.0f} max_ms={_RDTURBO_GC_ST['max_ms']:.0f} frozen={_rdturbo_gc.get_freeze_count()}", flush=True)
        _RDTURBO_GC_ST.update(full=0, full_ms=0.0, max_ms=0.0)
    if "freeze" in _RDTURBO_GC:
        _rdturbo_gc.unfreeze()


def _rdturbo_gc_post():
    # after every batch reset (incl. setup_scene): collect once and freeze again
    if "freeze" in _RDTURBO_GC:
        _rdturbo_gc.collect()
        _rdturbo_gc.freeze()
        _RDTURBO_GC_ST.update(full=0, full_ms=0.0, max_ms=0.0)


'''


def patch_gc_freeze(tree: Tree) -> str:
    s = tree.read(MAIN)
    m = find_once(s, r"^def main\(\):", "main.py def main (gc)")
    s = s[: m.start()] + _GC_HELPERS.lstrip("\n") + s[m.start():]
    s = sub_once(s, r"^([ \t]*)env\.reset\(seed=env\.env_seeds\)[ \t]*$",
                 lambda m: f"{m.group(1)}_rdturbo_gc_pre('reset')  # {MARK} gc\n{m.group(0)}\n{m.group(1)}_rdturbo_gc_post()  # {MARK} gc",
                 "main.py env.reset")
    import re
    s, n = re.subn(r"^([ \t]*)env\.close\(\)[ \t]*$", lambda m: f"{m.group(1)}_rdturbo_gc_pre('close')  # {MARK} gc\n{m.group(0)}", s, flags=re.M)
    if n < 1:
        raise PatchError("main.py env.close(): anchor not found")
    tree.write(MAIN, s)
    return f"RDTURBO_GC=freeze[,log] (close hooks x{n})"


# ---------------------------------------------------------------------------------------------------------------------
# ctrl_cache: control_robot re-resolves a control that pop() resolved in the same substep; reuse it

_CC_BOOT_CM = """import os as _rdturbo_cc_os  # [robodojo-turbo] ctrl-cache
_RDTURBO_CC = {"1": "1", "check": "check"}.get((_rdturbo_cc_os.environ.get("RDTURBO_CTRL_CACHE") or "").strip(), "0")   # 0 | 1 | check; anything else = upstream
"""

_CC_BOOT_RM = """import os as _rdturbo_cc_os  # [robodojo-turbo] ctrl-cache
_RDTURBO_CC = {"1": "1", "check": "check"}.get((_rdturbo_cc_os.environ.get("RDTURBO_CTRL_CACHE") or "").strip(), "0")   # 0 | 1 | check; anything else = upstream
_RDTURBO_CC_ST = {"hit": 0, "miss": 0, "chk": 0, "bad": 0}


def _rdturbo_cc_action(rm, mc, env_idx):
    # pop() already resolved this control (gripper clamped against the current joint value); resolving it again in the
    # same substep gives the same result (the clamp is idempotent and reads the same joint_pos). Valid only when pop() found
    # every key (nothing filled from prev_control) and the substep counter has not moved since; otherwise take the upstream path.
    at = getattr(mc, "_rdturbo_resolved_at", None)
    if _RDTURBO_CC == "0" or at is None or at != getattr(rm.sim, "_sim_step_counter", None):
        if _RDTURBO_CC != "0":
            _RDTURBO_CC_ST["miss"] += 1
        return mc.get_action(rm, env_idx=env_idx)
    got = mc.control_info_dict
    _RDTURBO_CC_ST["hit"] += 1
    if _RDTURBO_CC != "check":
        return got
    ref = mc.get_action(rm, env_idx=env_idx)
    _RDTURBO_CC_ST["chk"] += 1
    for k in ref:
        a, b = ref.get(k), got.get(k)
        if (a is None) != (b is None):
            _RDTURBO_CC_ST["bad"] += 1
            continue
        if a is None:
            continue
        for f in ("position", "velocity"):
            x = torch.tensor(a[f], dtype=torch.float32)
            y = torch.tensor(b[f], dtype=torch.float32)
            if x.shape != y.shape or not torch.equal(x, y):
                _RDTURBO_CC_ST["bad"] += 1
                if _RDTURBO_CC_ST["bad"] <= 5:
                    print(f"[robodojo-turbo] ctrl-cache MISMATCH env={env_idx} {k}.{f} ref={a[f]} got={b[f]}", flush=True)
    return ref


"""


def patch_ctrl_cache(tree: Tree) -> str:
    sc, sr = tree.read(CONTROL_MANAGER), tree.read(ROBOT_MANAGER)
    sc = sub_once(sc, r"^([ \t]*)return MetaControl\(meta_ctrl_dict\)[ \t]*$",
                  lambda m: (f"{m.group(1)}_rdturbo_mc = MetaControl(meta_ctrl_dict)  # {MARK} ctrl-cache\n"
                             f"{m.group(1)}if _RDTURBO_CC != '0' and all(k in new_meta_ctrl.control_info_dict for k in obs_list):\n"
                             f"{m.group(1)}    _rdturbo_mc._rdturbo_resolved_at = getattr(self.robot_manager.sim, '_sim_step_counter', None)\n"
                             f"{m.group(1)}return _rdturbo_mc"),
                  "control_manager return MetaControl")
    sr = sub_once(sr, r"^([ \t]*)meta_control = meta_control_list\[env_idx\]\.get_action\(\s*self,\s*env_idx=env_idx,\s*\)  # get action dict[ \t]*$",
                  lambda m: f"{m.group(1)}meta_control = _rdturbo_cc_action(self, meta_control_list[env_idx], env_idx)  # {MARK} ctrl-cache",
                  "robot_manager control_robot get_action")
    m = find_once(sr, r"^from env\.robot_manager\.control_manager import .*$", "robot_manager control_manager import")
    sr = sr[: m.end()] + "\n" + _CC_BOOT_RM + sr[m.end():]
    tree.write(CONTROL_MANAGER, _CC_BOOT_CM + sc)
    tree.write(ROBOT_MANAGER, sr)
    return "RDTURBO_CTRL_CACHE=1|check"


# ---------------------------------------------------------------------------------------------------------------------
# parent_cache: read a static parent Xform's world matrix from USD once per RigidPrim view

_PT_BOOT = """import os as _rdturbo_pt_os  # [robodojo-turbo] parent-cache
_RDTURBO_PT = {"1": "1", "check": "check"}.get((_rdturbo_pt_os.environ.get("RDTURBO_PARENT_CACHE") or "").strip(), "0")   # 0 | 1 | check; anything else = upstream
_RDTURBO_PT_ST = {"hit": 0, "miss": 0, "fallback": 0, "chk": 0, "bad": 0}


def _rdturbo_pt_same(a, b):
    if hasattr(a, "equal") and hasattr(b, "equal"):
        return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and bool(a.equal(b))
    import numpy as _np
    a, b = _np.asarray(a), _np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and _np.array_equal(a, b)


def _rdturbo_pt_local_pose(obj):
    # Adapted from NVIDIA Isaac Sim 5.1 isaacsim.core.prims rigid_prim.py, RigidPrim.get_local_poses (physics-handle branch),
    # SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES; SPDX-License-Identifier: Apache-2.0.
    # Same algorithm and order; modified 2026 by RoboDojo-Turbo (parent world matrix cached per view). See RoboDojo-Turbo's NOTICE.
    # Only difference: the parent's (a static category Xform) world matrix is read from USD once per view; views are rebuilt
    # when objects are respawned, which invalidates the cache.
    v = getattr(obj, "_prim_view", None)
    if (_RDTURBO_PT == "0" or v is None or type(v).__name__ != "RigidPrim" or getattr(obj, "_backend", "torch") == "warp"
            or not getattr(v, "_is_valid", False) or not v.is_physics_handle_valid() or v.count != 1):
        if _RDTURBO_PT != "0":
            _RDTURBO_PT_ST["fallback"] += 1
        return obj.get_local_pose()
    bu, dev = v._backend_utils, v._device
    idx = bu.resolve_indices(None, v.count, dev)
    wpos, wrot = v.get_world_poses(indices=idx)
    tf = v.__dict__.get("_rdturbo_pt_tf")
    if tf is None:
        import numpy as _np
        from pxr import Usd as _Usd, UsdGeom as _UsdGeom
        from isaacsim.core.utils.prims import get_prim_parent as _gpp
        m = _np.zeros(shape=(1, 4, 4), dtype=_np.float32)
        m[0] = _np.array(_UsdGeom.Xformable(_gpp(v._prims[0])).ComputeLocalToWorldTransform(_Usd.TimeCode.Default()), dtype="float32")
        tf = v._rdturbo_pt_tf = bu.convert(m, dtype="float32", device=dev)
        _RDTURBO_PT_ST["miss"] += 1
    else:
        _RDTURBO_PT_ST["hit"] += 1
    t, q = bu.get_local_from_world(tf, wpos, wrot, dev)
    t, q = t[0], q[0]
    if _RDTURBO_PT != "check":
        return t, q
    rt, rq = obj.get_local_pose()
    _RDTURBO_PT_ST["chk"] += 1
    if not (_rdturbo_pt_same(rt, t) and _rdturbo_pt_same(rq, q)):
        _RDTURBO_PT_ST["bad"] += 1
        if _RDTURBO_PT_ST["bad"] <= 5:
            print(f"[robodojo-turbo] parent-cache MISMATCH {getattr(v, 'prim_paths', ['?'])[0]} ref={rt},{rq} got={t},{q}", flush=True)
    return rt, rq


"""


def patch_parent_cache(tree: Tree) -> str:
    s = tree.read(RIGID)
    s = sub_once(s, r"^([ \t]*)obj_translation, obj_orientation = self\.get_local_pose\(\)[ \t]*$",
                 lambda m: f"{m.group(1)}obj_translation, obj_orientation = _rdturbo_pt_local_pose(self)  # {MARK} parent-cache",
                 "rigid.py get_local_pose")
    tree.write(RIGID, _PT_BOOT + s)
    return "RDTURBO_PARENT_CACHE=1|check"


# ---------------------------------------------------------------------------------------------------------------------
# video_mv: write video frames to ffmpeg without the tobytes() copy (the same C-order bytes by construction: frame is
# C-contiguous uint8 here, and the pipe write consumes the buffer before returning); 1 MiB pipe when running as root

_VW_BOOT = """import os as _rdturbo_vw_os  # [robodojo-turbo] video-mv
_RDTURBO_VW = "1" if (_rdturbo_vw_os.environ.get("RDTURBO_VIDEO_MV") or "").strip() == "1" else "0"   # 0 | 1
_RDTURBO_VW_ST = {"frames": 0, "pipe": None}
"""


def patch_video_mv(tree: Tree) -> str:
    import re
    s = tree.read(SAVE_FILE)
    s = sub_once(s, r"^([ \t]*)self\.proc\.stdin\.write\(frame\.tobytes\(\)\)[ \t]*$",
                 lambda m: (f"{m.group(1)}if _RDTURBO_VW != '0':  # {MARK} video-mv\n"
                            f"{m.group(1)}    _rdturbo_mv = memoryview(frame).cast('B')\n"
                            f"{m.group(1)}    _RDTURBO_VW_ST['frames'] += 1\n"
                            f"{m.group(1)}    self.proc.stdin.write(_rdturbo_mv)\n"
                            f"{m.group(1)}else:\n"
                            f"{m.group(1)}    self.proc.stdin.write(frame.tobytes())"),
                 "save_file stdin.write")
    m2 = find_once(s, r"^([ \t]*)self\.proc = subprocess\.Popen\(", "save_file Popen")
    ind = m2.group(1)
    close = re.compile(r"^" + re.escape(ind) + r"\)[ \t]*$", re.M).search(s, m2.end())
    if not close:
        raise PatchError("save_file Popen: closing parenthesis not found")
    # root only: an unprivileged user's pipes share a budget (fs.pipe-user-pages-soft, 64 MiB by default); once it is
    # used up, every new pipe of that user, in any process, shrinks to 8 KiB
    ins = (f"\n{ind}if _RDTURBO_VW != '0' and _rdturbo_vw_os.geteuid() == 0:  # {MARK} video-mv: 1 MiB pipe (holds a whole 640x480x3 frame)\n"
           f"{ind}    try:\n"
           f"{ind}        import fcntl as _rdturbo_fcntl\n"
           f"{ind}        _RDTURBO_VW_ST['pipe'] = _rdturbo_fcntl.fcntl(self.proc.stdin.fileno(), getattr(_rdturbo_fcntl, 'F_SETPIPE_SZ', 1031), 1 << 20)\n"
           f"{ind}    except OSError as _rdturbo_ex:\n"
           f"{ind}        _RDTURBO_VW_ST['pipe'] = f'err {{_rdturbo_ex}}'")
    s = s[: close.end()] + ins + s[close.end():]
    tree.write(SAVE_FILE, _VW_BOOT + s)
    return "RDTURBO_VIDEO_MV=1"


# ---------------------------------------------------------------------------------------------------------------------
# pci_copy: process_control_info makes K deepcopies of the control dict; two-level copy + deep-copy of untouched leaves

_CP_BOOT = """import os as _rdturbo_cp_os  # [robodojo-turbo] pci-copy
_RDTURBO_CP = {"1": "1", "check": "check"}.get((_rdturbo_cp_os.environ.get("RDTURBO_PCI_COPY") or "").strip(), "0")   # 0 | 1 | check; anything else = upstream
_RDTURBO_CP_ST = {"fast": 0, "chk": 0, "bad": 0}


def _rdturbo_cp_copy(ci):
    # two-level copy: new top-level and per-robot dicts; leaves shared for now (overwritten next, or deep-copied by _rdturbo_cp_fix)
    return {k: (dict(v) if type(v) is dict else deepcopy(v)) for k, v in ci.items()}


def _rdturbo_cp_fix(lst, ci):
    # any leaf still identical to the input object is deep-copied => structurally identical to one deepcopy per entry
    for d in lst:
        for k, v in ci.items():
            dv = d.get(k)
            if type(v) is dict and type(dv) is dict:
                for f, x in v.items():
                    if dv.get(f) is x:
                        dv[f] = deepcopy(x)


"""

_CP_METHOD = '''def process_control_info(self, control_info, env_idx):  # [robodojo-turbo] pci-copy
    if _RDTURBO_CP == "0" or any(getattr(r, "type", None) != "target" for r in self.robot_manager.robot_list):
        return self._rdturbo_pci_orig(control_info, env_idx)   # support arms: support_arm_action.pop has side effects, so never run twice
    got = self._rdturbo_pci_orig(control_info, env_idx, _rdturbo_copy=_rdturbo_cp_copy)
    _rdturbo_cp_fix(got, control_info)
    _RDTURBO_CP_ST["fast"] += 1
    if _RDTURBO_CP != "check":
        return got
    import pickle as _rdturbo_pk
    ref = self._rdturbo_pci_orig(control_info, env_idx)
    _RDTURBO_CP_ST["chk"] += 1
    if _rdturbo_pk.dumps(ref, 4) != _rdturbo_pk.dumps(got, 4):   # types, float bits and sharing structure
        _RDTURBO_CP_ST["bad"] += 1
        if _RDTURBO_CP_ST["bad"] <= 5:
            print(f"[robodojo-turbo] pci-copy MISMATCH env={env_idx}", flush=True)
    return ref

'''


def patch_pci_copy(tree: Tree) -> str:
    s = tree.read(EVAL_ENV)
    m = find_once(s, r"^([ \t]*)def process_control_info\(self, control_info, env_idx\):[ \t]*$", "eval_env process_control_info")
    copy_line = "control_info_list = [deepcopy(control_info) for _ in range(interpolation_nums)]"
    if s.count(copy_line) != 1:
        raise PatchError(f"eval_env process_control_info copy: anchor matched {s.count(copy_line)} times (expected exactly 1)")
    ind = m.group(1)
    s = s[: m.start()] + indent_block(_CP_METHOD, ind) + "\n" + f"{ind}def _rdturbo_pci_orig(self, control_info, env_idx, _rdturbo_copy=deepcopy):" + s[m.end():]
    s = s.replace(copy_line, "control_info_list = [_rdturbo_copy(control_info) for _ in range(interpolation_nums)]  # " + MARK + " pci-copy", 1)
    tree.write(EVAL_ENV, _CP_BOOT + s)
    return "RDTURBO_PCI_COPY=1|check (off automatically when a support robot is present)"


# ---------------------------------------------------------------------------------------------------------------------
# batch_tensor: build the per-robot joint-target tensors in one torch.tensor call instead of three per env

_BT_BOOT = """import os as _rdturbo_bt_os  # [robodojo-turbo] batch-tensor
_RDTURBO_BT = {"1": "1", "check": "check"}.get((_rdturbo_bt_os.environ.get("RDTURBO_BATCH_TENSOR") or "").strip(), "0")   # 0 | 1 | check; anything else = upstream
_RDTURBO_BT_ST = {"fast": 0, "slow": 0, "chk": 0, "bad": 0}


def _rdturbo_bt_fill(rm, robot, mcl, plan_lst, dev, ap, av, gp):
    # Same float32 conversion and the same _rdturbo_cc_action call order as the per-env loop; the velocity rows are left at 0
    # when every control's velocity is the int 0 (upstream writes 0 too). Any other shape/type falls back to per-row writes.
    an, gn = rm.process_name(robot.arm_name), rm.process_name(robot.gripper_name)
    mcs = [_rdturbo_cc_action(rm, mcl[e], e) for e in plan_lst]
    pa = [mc[an]["position"] for mc in mcs]
    pg = [mc[gn]["position"] for mc in mcs]
    fast = (len(mcs) > 0 and all(isinstance(r, (list, tuple)) and len(r) == ap.shape[1] for r in pa)
            and all(isinstance(r, (list, tuple)) and len(r) == gp.shape[1] for r in pg)
            and all(type(mc[an]["velocity"]) is int and mc[an]["velocity"] == 0 for mc in mcs))
    if fast:
        a = torch.tensor(pa, device=dev, dtype=torch.float32)
        g = torch.tensor(pg, device=dev, dtype=torch.float32)
    if not fast or _RDTURBO_BT == "check":
        for i, mc in enumerate(mcs):
            ap[i] = torch.tensor(mc[an]["position"], device=dev, dtype=torch.float32)
            av[i] = torch.tensor(mc[an]["velocity"], device=dev, dtype=torch.float32)
            gp[i][:] = torch.tensor(mc[gn]["position"], device=dev, dtype=torch.float32)
        if not fast:
            _RDTURBO_BT_ST["slow"] += 1
            return
        _RDTURBO_BT_ST["chk"] += 1
        if not (torch.equal(a, ap) and torch.equal(g, gp) and not bool(av.any())):
            _RDTURBO_BT_ST["bad"] += 1
            if _RDTURBO_BT_ST["bad"] <= 5:
                print(f"[robodojo-turbo] batch-tensor MISMATCH {robot.robot_name}", flush=True)
        return
    ap.copy_(a)
    gp.copy_(g)
    _RDTURBO_BT_ST["fast"] += 1

"""


def patch_batch_tensor(tree: Tree) -> str:
    s = tree.read(ROBOT_MANAGER)
    if "_rdturbo_cc_action" not in s:
        raise PatchError("batch_tensor needs the ctrl_cache patch applied first")
    s = sub_once(s, r"^([ \t]*)for id, env_idx in enumerate\(plan_lst\):[ \t]*$",
                 lambda m: (f"{m.group(1)}if _RDTURBO_BT != '0':  # {MARK} batch-tensor\n"
                            f"{m.group(1)}    _rdturbo_bt_fill(self, robot, meta_control_list, plan_lst, sim.device, arm_position, arm_velocity, gripper_position)\n"
                            f"{m.group(1)}for id, env_idx in (enumerate(plan_lst) if _RDTURBO_BT == '0' else ()):"),
                 "robot_manager control_robot loop")
    m = find_once(s, r"^_RDTURBO_CC_ST = .*$", "robot_manager _RDTURBO_CC_ST")
    s = s[: m.end()] + "\n" + _BT_BOOT + s[m.end():]
    tree.write(ROBOT_MANAGER, s)
    return "RDTURBO_BATCH_TENSOR=1|check"
