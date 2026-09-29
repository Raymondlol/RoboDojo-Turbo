"""Harness patches (profile "harness"): tracing, layout sharding, the Pi_05 evaluation loop, the lossless fast path,
and exit-status accounting. With every RDTURBO_* switch unset, the patched tree behaves like upstream.
"""
from __future__ import annotations

import os
import re

from .. import MARK
from ..engine import PatchError, Tree, find_once, indent_block, sub_once

RUNTIME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime")

EVAL_ENV = "src/eval_client/eval_env.py"
MAIN = "src/eval_client/main.py"
SEED_MANAGER = "env/seed_manager/seed_manager.py"
EVAL_POLICY_SH = "scripts/eval_policy.sh"
PI05_DEPLOY = "XPolicyLab/policy/Pi_05/deploy.py"
SAVE_FILE = "utils/save_file.py"
ROBOT_MANAGER = "env/robot_manager/robot_manager.py"
CONTROL_MANAGER = "env/robot_manager/control_manager.py"
FUNC_PARSER = "env/reward_manager/func_parser.py"


# ---------------------------------------------------------------------------------------------------------------------
# trace: span tracer + video fps that follows video subsampling

def patch_trace(tree: Tree) -> str:
    tree.install(os.path.join(RUNTIME, "rdturbo_trace.py"), "utils/rdturbo_trace.py")
    s = tree.read(EVAL_ENV)
    # install the tracer at the end of EvalEnv.__init__ (the method right before the first `def close(self):`)
    m = find_once(s, r"^([ \t]*)def close\(self\):", "eval_env close()")
    ind = m.group(1) + "    "
    block = (f"{ind}if (os.environ.get('RDTURBO_TRACE') or '').strip() not in ('', '0'):  # {MARK} tracing\n"
             f"{ind}    from utils.rdturbo_trace import install_env_tracing as _rdturbo_install_tracing\n"
             f"{ind}    _rdturbo_install_tracing(self)\n\n")
    s = s[: m.start()] + block + s[m.start():]
    # keep real-time playback when only every k-th in-chunk step is rendered for the video
    s = sub_once(s, r"^([ \t]*)fps = self\.obs_manager\.collect_freq[ \t]*$",
                 lambda m: (f"{m.group(0)}\n{m.group(1)}if (os.environ.get('RDTURBO_OBS_EVERY_STEP') or '1').strip() == '0' and int((os.environ.get('RDTURBO_VIDEO_EVERY') or '').strip() or '1') > 1:  # {MARK} video subsampling\n"
                            f"{m.group(1)}    fps = fps / int(os.environ['RDTURBO_VIDEO_EVERY'].strip())"),
                 "eval_env video fps")
    tree.write(EVAL_ENV, s)
    return "tracer installed; EvalEnv hooks RDTURBO_TRACE, video fps follows RDTURBO_VIDEO_EVERY"


# ---------------------------------------------------------------------------------------------------------------------
# layout_shards: evaluate a subset of layouts per process, override num_envs, lift the per-task eval_nums cap

def patch_layout_shards(tree: Tree) -> str:
    s = tree.read(SEED_MANAGER)
    m = find_once(s, r"^([ \t]*)all_layout_ids = list\(range\(len\(matching_files\)\)\)[ \t]*$", "seed_manager layout list")
    ind = m.group(1)
    block = (f"\n{ind}_rdturbo_ids = (os.environ.get('RDTURBO_LAYOUT_IDS') or '').strip()  # {MARK} layout sharding\n"
             f"{ind}if _rdturbo_ids:\n"
             f"{ind}    from utils.rdturbo_trace import parse_id_spec as _rdturbo_parse\n"
             f"{ind}    _rdturbo_keep = set(_rdturbo_parse(_rdturbo_ids))\n"
             f"{ind}    all_layout_ids = [i for i in all_layout_ids if i in _rdturbo_keep]\n"
             f"{ind}    print(f'{MARK} layout shard {{_rdturbo_ids}} -> {{len(all_layout_ids)}} layouts')\n"
             f"{ind}    if not all_layout_ids:\n"
             f"{ind}        raise ValueError(f'RDTURBO_LAYOUT_IDS={{_rdturbo_ids!r}} keeps none of the {{len(matching_files)}} layouts')\n")
    s = s[: m.end()] + block + s[m.end():]
    if not re.search(r"^import os\s*$", s, re.M):
        s = "import os\n" + s
    tree.write(SEED_MANAGER, s)

    s = tree.read(EVAL_POLICY_SH)
    s = sub_once(s, r'^num_envs="\$\(python3 -c .*\)"[ \t]*$',
                 lambda m: f'{m.group(0)}\nnum_envs="${{RDTURBO_NUM_ENVS:-$num_envs}}"  # {MARK} per-worker num_envs override',
                 "eval_policy.sh num_envs")
    tree.write(EVAL_POLICY_SH, s)

    # upstream uses min(EVAL_NUM, task yaml eval_nums): a shard asking for more than eval_nums (25 for most tasks)
    # is silently truncated. RDTURBO_EVAL_NUM_FORCE=1 lets EVAL_NUM win.
    s = tree.read(MAIN)
    s = sub_once(s, r"^([ \t]*)eval_num = min\(int\(_env_eval_num\), int\(eval_num\)\)[ \t]*$",
                 lambda m: (f"{m.group(1)}eval_num = (int(_env_eval_num) if (os.environ.get('RDTURBO_EVAL_NUM_FORCE') or '0').strip() == '1'"
                            f" else min(int(_env_eval_num), int(eval_num)))  # {MARK} eval-num"),
                 "main.py eval_num cap")
    tree.write(MAIN, s)
    return "RDTURBO_LAYOUT_IDS / RDTURBO_NUM_ENVS / RDTURBO_EVAL_NUM_FORCE"


# ---------------------------------------------------------------------------------------------------------------------
# pi05_loop: the Pi_05 batch loop with on-demand observation upload, video subsampling and verification hooks

PI05_LOOP = r'''
import sys as _rdturbo_sys  # [robodojo-turbo] pi05-loop
# eval_one_episode_batch below is adapted from XPolicyLab policy/Pi_05/deploy.py @ bb9a0b5 (Apache License 2.0).
# Modified 2026 by RoboDojo-Turbo: optional chunk-start-only observation upload, video subsampling, action record/replay,
# state/particle tracing, observation dumps, and episode hooks for RDTURBO_USD_LAST. See RoboDojo-Turbo's NOTICE file.


def _rdturbo_env_mod(TASK_ENV):
    return _rdturbo_sys.modules.get(type(TASK_ENV).__module__)


_RDTURBO_RR = {"rec": None, "rep": None, "trace": None, "ptrace": None, "miss": 0, "hit": 0, "merged": set()}


def _rdturbo_path(var, stem, ext=".pkl"):
    # RDTURBO_<X>=1 -> <dir of RDTURBO_TRACE>/<stem>_<worker><ext>; any other non-empty value is used as the path
    v = (os.environ.get(var) or "").strip()
    if v in ("", "0"):
        return None
    if v == "1":
        tr = (os.environ.get("RDTURBO_TRACE") or "").strip()
        v = os.path.join(os.path.dirname(tr) if tr else ".", stem + "_" + (os.environ.get("RDTURBO_WORKER") or "w0").strip() + ext)
    return v


def _rdturbo_rec_path():
    # with replay active, RDTURBO_RECORD_ACTIONS=1 records to actions_rec_<worker>.pkl so it never overwrites the replay file
    rep = _rdturbo_path("RDTURBO_REPLAY_ACTIONS", "actions")
    rec = _rdturbo_path("RDTURBO_RECORD_ACTIONS", "actions_rec" if rep else "actions")
    if rec and rep and os.path.abspath(rec) == os.path.abspath(rep):
        rec = os.path.splitext(rec)[0] + "_rec.pkl"
    return rec


def _rdturbo_actions(TASK_ENV, model_client, env_idx_list, chunk_idx):
    # Verification tooling (off by default): RDTURBO_RECORD_ACTIONS stores every (layout, chunk) action chunk the run
    # executes; RDTURBO_REPLAY_ACTIONS feeds a recorded file instead of asking the server, so two runs differ only in physics.
    rp = _rdturbo_path("RDTURBO_REPLAY_ACTIONS", "actions")
    rec = _rdturbo_rec_path()
    if not rp and not rec:
        return model_client.call(func_name="get_action_batch", obs=env_idx_list)   # the upstream call, nothing else
    import pickle
    seeds = TASK_ENV.env_seeds
    keys = [(int(seeds[e]), int(chunk_idx)) for e in env_idx_list]
    if rp:
        if _RDTURBO_RR["rep"] is None:
            with open(rp, "rb") as fh:
                _RDTURBO_RR["rep"] = pickle.load(fh)
        missing = [k for k in keys if k not in _RDTURBO_RR["rep"]]
        _RDTURBO_RR["hit"] += len(keys) - len(missing)
        _RDTURBO_RR["miss"] += len(missing)
        if missing:
            print(f"[robodojo-turbo] replay MISS chunk={chunk_idx} keys={missing} (live policy actions for these envs only)", flush=True)
            if (os.environ.get("RDTURBO_REPLAY_STRICT") or "0").strip() == "1":
                # an open-loop replay that asks the policy server is no longer open loop: stop instead of mixing
                print("[robodojo-turbo] RDTURBO_REPLAY_STRICT=1: recorded actions missing; exiting with status 5", flush=True)
                os._exit(5)
            live = model_client.call(func_name="get_action_batch", obs=env_idx_list)
            actions = [_RDTURBO_RR["rep"].get(k, a) for k, a in zip(keys, live)]
        else:
            actions = [_RDTURBO_RR["rep"][k] for k in keys]
    else:
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
    if rec:
        if _RDTURBO_RR["rec"] is None:
            _RDTURBO_RR["rec"] = {}
        for k, a in zip(keys, actions):
            _RDTURBO_RR["rec"][k] = a
    return actions


def _rdturbo_f64(x):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def _rdturbo_usd_link_poses(obj):
    # USD world transform of every rigid-body prim under an articulated scene object (what base_link scorers read)
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    out = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(obj._prim_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            out.append(np.asarray(m, dtype=np.float64).ravel())
    return out


def _rdturbo_state_trace(TASK_ENV, env_idx_list, chunk_idx, action_idx):
    # Verification tooling (off by default): RDTURBO_STATE_TRACE records, after every action step and per env, float64
    # robot joint positions and link poses (PhysX), scene rigid-object poses (same accessor as the scorers) and, for
    # articulated scene objects, joint positions (PhysX) under key (layout, chunk, step); the USD world transform of every
    # articulated link (what USD-reading scorers see) goes under (layout, chunk, step, "usd"), so that a physics verdict
    # never mixes in USD write-back state.
    if not _rdturbo_path("RDTURBO_STATE_TRACE", "state"):
        return
    import numpy as np
    if _RDTURBO_RR["trace"] is None:
        _RDTURBO_RR["trace"] = {}
    rm = TASK_ENV.robot_manager
    lm = TASK_ENV.scene_manager.layout_manager
    seeds = TASK_ENV.env_seeds
    for e in env_idx_list:
        parts, usd = [], []
        for key in rm.robot_key:
            parts.append(_rdturbo_f64(key.data.joint_pos[e]))
            parts.append(_rdturbo_f64(key.data.body_link_pose_w[e]))
        for inst, typ in sorted(lm.instance_type_by_env[e].items()):
            if typ == "rigid":
                pos, rot = lm.get_instance_pose(e, inst_name=inst)
                parts.append(_rdturbo_f64(pos))
                parts.append(_rdturbo_f64(rot))
            elif typ == "articulation":
                obj = lm.get_scene_object(e, inst)
                try:
                    parts.append(_rdturbo_f64(obj.get_current_joint_positions()))
                    usd.extend(_rdturbo_usd_link_poses(obj))
                except Exception as ex:   # tracing must never break the evaluation; a NaN marks the gap
                    print(f"[robodojo-turbo] state-trace articulation {inst} env {e}: {ex!r}", flush=True)
                    parts.append(np.array([np.nan]))
        k = (int(seeds[e]), int(chunk_idx), int(action_idx))
        _RDTURBO_RR["trace"][k] = np.concatenate(parts)
        if usd:
            _RDTURBO_RR["trace"][k + ("usd",)] = np.concatenate(usd)


def _rdturbo_particle_trace(TASK_ENV, env_idx_list, chunk_idx, action_idx):
    # Verification tooling (off by default): RDTURBO_PARTICLE_TRACE records the USD-side points that fluid and garment
    # scorers read (FluidObject.get_particle_positions = PointInstancer positions; GarmentObject.sample_mesh_vertices =
    # USD points in world frame), float32, after every action step.
    if not _rdturbo_path("RDTURBO_PARTICLE_TRACE", "particles"):
        return
    import numpy as np
    if _RDTURBO_RR["ptrace"] is None:
        _RDTURBO_RR["ptrace"] = {}
    lm = TASK_ENV.scene_manager.layout_manager
    seeds = TASK_ENV.env_seeds
    for e in env_idx_list:
        for inst, typ in sorted(lm.instance_type_by_env[e].items()):
            if typ not in ("fluid", "garment"):
                continue
            obj = lm.get_scene_object(e, inst)
            try:
                pts = obj.get_particle_positions()[0] if typ == "fluid" else obj.sample_mesh_vertices()[0]
            except Exception as ex:
                print(f"[robodojo-turbo] particle-trace {typ} {inst} env {e}: {ex!r}", flush=True)
                continue
            if pts is not None:
                _RDTURBO_RR["ptrace"][(int(seeds[e]), int(chunk_idx), int(action_idx), typ + ":" + inst)] = np.array(pts, dtype=np.float32)


def _rdturbo_obs_dump(TASK_ENV, obs_list, env_idx_list, chunk_idx):
    # Verification tooling (off by default): RDTURBO_OBS_DUMP stores the camera images uploaded at every chunk start
    # (the only observation Pi_05 consumes), one compressed npz per chunk, keyed "<layout>/<camera>".
    d = _rdturbo_path("RDTURBO_OBS_DUMP", "obs_dump", ext="")
    if not d:
        return
    import numpy as np
    os.makedirs(d, exist_ok=True)
    seeds = TASK_ENV.env_seeds
    arrs = {}
    for obs, e in zip(obs_list, env_idx_list):
        for cam, cd in ((obs or {}).get("vision") or {}).items():
            c = cd.get("color") if isinstance(cd, dict) else None
            if c is not None:
                arrs[f"{int(seeds[e])}/{cam}"] = np.ascontiguousarray(c)
    first = min(int(seeds[e]) for e in env_idx_list) if env_idx_list else -1   # named by batch and chunk: a restarted
    np.savez_compressed(os.path.join(d, f"chunk_l{first:05d}_c{int(chunk_idx):03d}.npz"), **arrs)   # client cannot overwrite


def _rdturbo_flush(TASK_ENV=None):
    import pickle
    # counters of the check modes, cumulative over the process (Kit does not run atexit reliably, so print after each batch)
    mods = [_rdturbo_env_mod(TASK_ENV) if TASK_ENV is not None else None] + [
        _rdturbo_sys.modules.get(n) for n in ("env.robot_manager.robot_manager", "env.scene_manager.objects.rigid", "utils.save_file")]
    for mod in mods:
        for name in ("_RDTURBO_VO_STAT", "_RDTURBO_UL_ST", "_RDTURBO_CC_ST", "_RDTURBO_PT_ST", "_RDTURBO_VW_ST", "_RDTURBO_CP_ST", "_RDTURBO_BT_ST"):
            v = getattr(mod, name, None) if mod is not None else None
            if v and any(x for x in v.values() if isinstance(x, (int, float)) and not isinstance(x, bool)):
                print(f"[robodojo-turbo] batch-stats {name} {v}", flush=True)
    for p, k in ((_rdturbo_rec_path(), "rec"), (_rdturbo_path("RDTURBO_STATE_TRACE", "state"), "trace"),
                 (_rdturbo_path("RDTURBO_PARTICLE_TRACE", "particles"), "ptrace")):
        if p and _RDTURBO_RR[k] is not None:
            if p not in _RDTURBO_RR["merged"] and os.path.exists(p):
                # first write of this process to an existing file: a restarted client resumes the run, keep what the
                # earlier process recorded (entries re-evaluated now take precedence)
                with open(p, "rb") as fh:
                    for kk, vv in pickle.load(fh).items():
                        _RDTURBO_RR[k].setdefault(kk, vv)
            _RDTURBO_RR["merged"].add(p)
            with open(p + ".tmp", "wb") as fh:
                pickle.dump(_RDTURBO_RR[k], fh)
            os.replace(p + ".tmp", p)
    if _rdturbo_path("RDTURBO_REPLAY_ACTIONS", "actions"):
        print(f"[robodojo-turbo] replay hit={_RDTURBO_RR['hit']} miss={_RDTURBO_RR['miss']}", flush=True)


def eval_one_episode_batch(TASK_ENV, model_client):
    # [robodojo-turbo] USD_LAST episode hooks: no-op unless the usd_last patch is installed and RDTURBO_USD_LAST is set
    mod = _rdturbo_env_mod(TASK_ENV)
    begin, end = getattr(mod, "_rdturbo_ul_episode_begin", None), getattr(mod, "_rdturbo_ul_episode_end", None)
    if begin is not None:
        begin(TASK_ENV)
    try:
        return _rdturbo_eval_one_episode_batch(TASK_ENV, model_client)
    finally:
        if end is not None:
            end()
        _rdturbo_flush(TASK_ENV)   # also after a batch that raised


def _rdturbo_eval_one_episode_batch(TASK_ENV, model_client):
    # With RDTURBO_OBS_EVERY_STEP unset (or 1) this is the upstream loop, call for call.
    # RDTURBO_OBS_EVERY_STEP=0: between chunk starts, observations are rendered for the video only and not uploaded
    # (the Pi_05 server overwrites its observation window on every upload and reads it only at get_action_batch).
    obs_every_step = (os.environ.get("RDTURBO_OBS_EVERY_STEP") or "1").strip() != "0"
    video_every = int((os.environ.get("RDTURBO_VIDEO_EVERY") or "").strip() or "1")
    mod = _rdturbo_env_mod(TASK_ENV)
    vo_kw = {"rdturbo_video_only": True} if ((os.environ.get("RDTURBO_VIDEO_ONLY_OBS") or "0").strip() not in ("", "0") and hasattr(mod, "_RDTURBO_VO")) else {}

    model_client.call(func_name="reset")
    chunk_idx = 0
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        _rdturbo_obs_dump(TASK_ENV, obs_list, env_idx_list, chunk_idx)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = _rdturbo_actions(TASK_ENV, model_client, env_idx_list, chunk_idx)
        chunk_idx += 1

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            _rdturbo_state_trace(TASK_ENV, env_idx_list, chunk_idx - 1, action_idx)
            _rdturbo_particle_trace(TASK_ENV, env_idx_list, chunk_idx - 1, action_idx)

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]
            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]

            if obs_every_step:
                model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
            elif video_every > 0 and (action_idx + 1) % video_every == 0:
                TASK_ENV.get_obs_batch(env_idx_list, **vo_kw)  # render for the video only; Pi_05 reads the chunk-start observation
'''


def patch_pi05_loop(tree: Tree) -> str:
    s = tree.read(PI05_DEPLOY)
    m = find_once(s, r"^def eval_one_episode_batch\(TASK_ENV, model_client\):.*\Z", "Pi_05 eval_one_episode_batch", flags=re.M | re.S)
    s = s[: m.start()].rstrip() + "\n" + PI05_LOOP
    if not re.search(r"^import os\s*$", s, re.M):
        s = f"import os  # {MARK}\n" + s
    tree.write(PI05_DEPLOY, s)
    return "Pi_05 eval_one_episode_batch replaced (RDTURBO_OBS_EVERY_STEP / VIDEO_EVERY / RECORD|REPLAY_ACTIONS / STATE|PARTICLE_TRACE / OBS_DUMP)"


# ---------------------------------------------------------------------------------------------------------------------
# fast: lossless copy/recompute removal (RDTURBO_FAST=1); RDTURBO_VIDEO_CAMS limits which cameras are recorded to video

FAST = MARK + " fast"
_FAST_BOOT = ["import os as _rdturbo_os  # " + FAST, '_RDTURBO_FAST = (_rdturbo_os.environ.get("RDTURBO_FAST") or "0").strip() == "1"']


def _fast_boot(s: str, extra=()) -> str:
    return "\n".join(_FAST_BOOT + list(extra)) + "\n" + s


def _apply_subs(s: str, subs, where: str) -> str:
    for name, pat, repl in subs:
        s, n = re.subn(pat, repl, s, flags=re.M)
        if n == 0:
            raise PatchError(f"{where} {name}: anchor not found")
    return s


_EVAL_ENV_FAST_EXTRA = [
    '_RDTURBO_VIDEO_CAMS = set(c.strip() for c in (_rdturbo_os.environ.get("RDTURBO_VIDEO_CAMS") or "").split(",") if c.strip())',
    "_RDTURBO_PRINT = (lambda *a, **k: None) if _RDTURBO_FAST else print",
    "",
    "",
    "def _rdturbo_copy_obs(d):",
    '    """B2: upstream deep-copies every env observation (27.6 MB of images per step). This keeps the same isolation',
    "    (arrays are still copied so the next frame cannot overwrite them) without deepcopy's memo/reflection cost.\"\"\"",
    "    if not _RDTURBO_FAST:",
    "        return deepcopy(d)",
    "    if isinstance(d, dict):",
    "        return {k: _rdturbo_copy_obs(v) for k, v in d.items()}",
    "    if isinstance(d, (list, tuple)):",
    "        return type(d)(_rdturbo_copy_obs(v) for v in d)",
    "    if hasattr(d, 'clone') and hasattr(d, 'dtype'):",
    "        return d.clone()",
    "    if hasattr(d, 'copy') and hasattr(d, 'dtype'):",
    "        return d.copy()",
    "    return d",
]


def patch_fast(tree: Tree) -> str:
    # B1/B6 (robot_manager): whole-tensor clones and per-env deepcopies whose results are copied again downstream anyway
    s = tree.read(ROBOT_MANAGER)
    s = _apply_subs(s, [
        ("B1 joint_state", r"^([ \t]*)joint_state = key\.data\.joint_pos\.clone\(\)[ \t]*$",
         r"\g<1>joint_state = key.data.joint_pos if _RDTURBO_FAST else key.data.joint_pos.clone()  # " + FAST + " B1"),
        ("B1 joints", r"^([ \t]*)joints = deepcopy\(joint_state\[env_idx\]\[gripper_indices\]\)[ \t]*$",
         r"\g<1>joints = joint_state[env_idx][gripper_indices] if _RDTURBO_FAST else deepcopy(joint_state[env_idx][gripper_indices])  # " + FAST + " B1"),
        ("B6 env_origins", r"^([ \t]*)env_origin_pos = deepcopy\(self\.scene\.env_origins\)[ \t]*$",
         r"\g<1>env_origin_pos = self.scene.env_origins if _RDTURBO_FAST else deepcopy(self.scene.env_origins)  # " + FAST + " B6"),
        ("B6 link_pose", r"^([ \t]*)link_pose = key\.data\.body_link_pose_w\.clone\(\)[ \t]*$",
         r"\g<1>link_pose = key.data.body_link_pose_w if _RDTURBO_FAST else key.data.body_link_pose_w.clone()  # " + FAST + " B6"),
        ("B6 pose", r"^([ \t]*)pose = deepcopy\(link_pose\[env_idx\]\[link_idx\]\)[ \t]*$",
         r"\g<1>pose = link_pose[env_idx][link_idx] if _RDTURBO_FAST else deepcopy(link_pose[env_idx][link_idx])  # " + FAST + " B6"),
    ], "robot_manager")
    tree.write(ROBOT_MANAGER, _fast_boot(s))

    # B1 (control_manager): pop() resolves each control twice; the gripper clamp is idempotent, so reuse the first result
    s = tree.read(CONTROL_MANAGER)
    s = _apply_subs(s, [
        ("B1 signature",
         r"^([ \t]*)def update_prev_control\(self, env_idx, new_meta_ctrl: MetaControl\):[ \t]*\n"
         r"([ \t]*)meta_ctrl_dict = new_meta_ctrl\.get_action\(self\.robot_manager, env_idx\)[ \t]*$",
         r"\g<1>def update_prev_control(self, env_idx, new_meta_ctrl: MetaControl, resolved=None):  # " + FAST + " B1\n"
         r"\g<2>meta_ctrl_dict = resolved if resolved is not None else new_meta_ctrl.get_action(self.robot_manager, env_idx)"),
        ("B1 call site", r"^([ \t]*)self\.update_prev_control\(env_idx, meta_ctrl\)[ \t]*$",
         r"\g<1>self.update_prev_control(env_idx, meta_ctrl, resolved=(meta_ctrl.control_info_dict if _RDTURBO_FAST else None))  # " + FAST + " B1"),
    ], "control_manager")
    tree.write(CONTROL_MANAGER, _fast_boot(s))

    # B3 (func_parser): back-to-origin predicates computed end poses for every env to use one
    s = tree.read(FUNC_PARSER)
    s = _apply_subs(s, [
        ("B3 get_real_endpose", r"self\.robot_manager\.get_real_endpose\(robot\)\[env_idx\]",
         r"self.robot_manager.get_real_endpose(robot, env_idx_list=([env_idx] if _RDTURBO_FAST else None))[env_idx]  # " + FAST + " B3"),
    ], "func_parser")
    tree.write(FUNC_PARSER, _fast_boot(s))

    # B2/B5/B8 (eval_env): observation copy, video camera filter, per-step print
    s = tree.read(EVAL_ENV)
    s = _apply_subs(s, [
        ("B2 deepcopy", r"^([ \t]*)env_data = deepcopy\(data\[env_idx\]\)[ \t]*$",
         r"\g<1>env_data = _rdturbo_copy_obs(data[env_idx])  # " + FAST + " B2"),
        ("B8 step print", r"print\((\s*f\"env\{env_idx\} step:)", r"_RDTURBO_PRINT(\g<1>"),
        ("B5 video cams", r"^([ \t]*)for cam_key, cam_data in vision\.items\(\):[ \t]*$",
         r"\g<1>for cam_key, cam_data in vision.items():\n"
         r"\g<1>    if _RDTURBO_VIDEO_CAMS and cam_key not in _RDTURBO_VIDEO_CAMS:  # " + FAST + " B5\n"
         r"\g<1>        continue"),
    ], "eval_env")
    tree.write(EVAL_ENV, _fast_boot(s, _EVAL_ENV_FAST_EXTRA))

    # B5 (save_file): x264 preset ultrafast (file size only; videos are not scored)
    s = tree.read(SAVE_FILE)
    m = find_once(s, r'(\n(\s*)"-crf",)', "save_file -crf")
    ind = m.group(2)
    ins = f'\n{ind}*(["-preset", "ultrafast"] if (__import__("os").environ.get("RDTURBO_FAST") or "0").strip() == "1" else []),  # {FAST} preset'
    s = s[: m.start()] + ins + s[m.start():]
    tree.write(SAVE_FILE, s)
    return "RDTURBO_FAST (B1,B2,B3,B5,B6,B8) / RDTURBO_VIDEO_CAMS"


# ---------------------------------------------------------------------------------------------------------------------
# ffmpeg_threads: optional cap on x264 threads (default x264 spawns 1.5x visible cores per stream)

def patch_ffmpeg_threads(tree: Tree) -> str:
    s = tree.read(SAVE_FILE)
    m = find_once(s, r'(\n(\s*)"-vcodec",\n\s*"libx264",)', "save_file libx264")
    ind = m.group(2)
    ins = (f'\n{ind}*(["-threads", __import__("os").environ["RDTURBO_FFMPEG_THREADS"]] '
           f'if __import__("os").environ.get("RDTURBO_FFMPEG_THREADS") else []),  # {MARK} ffmpeg threads')
    s = s[: m.end()] + ins + s[m.end():]
    tree.write(SAVE_FILE, s)
    return "RDTURBO_FFMPEG_THREADS=<n> (unset = upstream)"


# ---------------------------------------------------------------------------------------------------------------------
# accounting: upstream's generic `except Exception` in main.py consumes a failing batch's seeds and the run still exits 0.

_ACCOUNTING_HELPERS = r'''
_RDTURBO_SWALLOWED = []   # [robodojo-turbo] accounting
_RDTURBO_ACC = {"ok": True}


def _rdturbo_swallowed(e):
    _RDTURBO_SWALLOWED.append(f"{type(e).__name__}: {e}"[:300])


def _rdturbo_accounting(env, eval_num):
    import json as _j
    st = {"eval_num": int(eval_num), "success_nums": int(getattr(env, "success_nums", 0)),
          "fail_nums": int(getattr(env, "fail_nums", 0)), "unstable_nums": int(getattr(env, "unstable_nums", 0)),
          "swallowed_batches": len(_RDTURBO_SWALLOWED), "swallowed": _RDTURBO_SWALLOWED[:20]}
    st["ok"] = st["swallowed_batches"] == 0 and st["success_nums"] + st["fail_nums"] + st["unstable_nums"] >= st["eval_num"]
    print(f"[robodojo-turbo] accounting {st}", flush=True)
    path = os.environ.get("RDTURBO_STATUS")
    if not path and (os.environ.get("RDTURBO_TRACE") or "").strip() not in ("", "0"):
        path = os.path.join(os.path.dirname(os.environ["RDTURBO_TRACE"]), "status_" + (os.environ.get("RDTURBO_WORKER") or "w0").strip() + ".json")
    if path:
        with open(path, "w") as fh:
            _j.dump(st, fh)
    _RDTURBO_ACC["ok"] = st["ok"]


def _rdturbo_strict_exit():
    # after upstream deleted its resume manifest and closed the policy client, so a rerun of the tag starts fresh
    if not _RDTURBO_ACC["ok"] and (os.environ.get("RDTURBO_STRICT") or "0").strip() == "1":
        print("[robodojo-turbo] RDTURBO_STRICT=1 and the run lost batches or episodes: exiting with status 4", flush=True)
        os._exit(4)


'''


def patch_accounting(tree: Tree) -> str:
    s = tree.read(MAIN)
    s = sub_once(s, r"^([ \t]*)else:\n\1    env\.seed_manager\.eval_step\(\)[ \t]*$",
                 lambda m: (f"{m.group(1)}else:\n{m.group(1)}    _rdturbo_swallowed(e)  # {MARK} accounting\n"
                            f"{m.group(1)}    env.seed_manager.eval_step()"),
                 "main.py generic-except eval_step")
    s = sub_once(s, r"^([ \t]*)_delete_resume_manifest\(env\)[ \t]*$",
                 lambda m: f"{m.group(1)}_rdturbo_accounting(env, eval_num)  # {MARK} accounting\n{m.group(0)}",
                 "main.py end of eval loop")
    s = sub_once(s, r"^([ \t]*)_close_model_client\(env\)[ \t]*$",
                 lambda m: f"{m.group(0)}\n{m.group(1)}_rdturbo_strict_exit()  # {MARK} accounting",
                 "main.py _close_model_client")
    m = find_once(s, r"^def main\(\):", "main.py def main")
    s = s[: m.start()] + _ACCOUNTING_HELPERS.lstrip("\n") + s[m.start():]
    tree.write(MAIN, s)
    return "batches swallowed by the generic except are counted; RDTURBO_STATUS / RDTURBO_STRICT=1"
