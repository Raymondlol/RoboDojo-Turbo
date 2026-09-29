"""GPU-free tests of the code the patches inject into upstream files. The injected source is exec'd from the patch
modules' string constants, so these tests cover exactly what lands in a patched RoboDojo tree."""
import copy
import json
import os
import pickle
import queue
import random
import sys
import types

import numpy as np
import pytest

from robodojo_turbo.patches import harness as H
from robodojo_turbo.patches import ops as O
from robodojo_turbo.patches import speedup as S


def _ns(src, **extra):
    ns = {"deepcopy": copy.deepcopy, "np": np, "os": os, **extra}
    exec(src, ns)
    return ns


# --------------------------------------------------------------------------------------------------- pci_copy / fast

def _rand_ci(rng):
    return {f"r{k}": {"position": [rng.random() for _ in range(7)], "velocity": 0, "gain": [rng.random()]}
            for k in range(rng.randint(1, 3))} | {"flag": rng.random() > 0.5}


@pytest.mark.parametrize("seed", range(50))
def test_pci_copy_isomorphic_to_deepcopy(seed):
    ns = _ns(S._CP_BOOT)
    rng = random.Random(seed)
    ci, k = _rand_ci(rng), rng.randint(1, 12)
    ref = [copy.deepcopy(ci) for _ in range(k)]
    got = [ns["_rdturbo_cp_copy"](ci) for _ in range(k)]
    ns["_rdturbo_cp_fix"](got, ci)
    assert pickle.dumps(ref, 4) == pickle.dumps(got, 4)
    for d in got:
        for key, v in ci.items():
            if isinstance(v, dict):
                assert d[key] is not v and all(d[key][f] is not x for f, x in v.items() if isinstance(x, list))


def test_fast_obs_copy_isolated_from_source():
    src = "\n".join(line for line in H._EVAL_ENV_FAST_EXTRA if not line.startswith(("_RDTURBO_VIDEO_CAMS", "_RDTURBO_PRINT")))
    ns = _ns(src, _RDTURBO_FAST=True)
    a = np.zeros((2, 2), np.uint8)
    out = ns["_rdturbo_copy_obs"]({"vision": {"cam_head": {"color": a}}, "joint": [1.0]})
    a[0, 0] = 7
    assert out["vision"]["cam_head"]["color"][0, 0] == 0


# --------------------------------------------------------------------------------------------------- ops / speedup

def test_no_planner_stub_exits_hard(monkeypatch):
    def fake_exit(code):
        raise SystemExit(code)
    monkeypatch.setattr(os, "_exit", fake_exit)
    ns = _ns(O._NP_STUB)
    stub = ns["_RdturboNoPlanner"]("left_arm")
    with pytest.raises(SystemExit) as e:
        stub.solve_ik
    assert e.value.code == 3
    with pytest.raises(AttributeError):
        stub.__deepcopy__   # dunder probes (copy/pickle) must not look like planner use


def test_gc_helpers_default_off(monkeypatch):
    import gc
    monkeypatch.delenv("RDTURBO_GC", raising=False)
    before = (gc.get_freeze_count(), len(gc.callbacks))
    _ns(S._GC_HELPERS)
    assert (gc.get_freeze_count(), len(gc.callbacks)) == before


class _MC:
    control_info_dict = {"a": 1}

    def __init__(self, at):
        self._rdturbo_resolved_at = at

    def get_action(self, rm, env_idx):
        return {"a": 2}


def test_ctrl_cache_hits_only_with_matching_step_counter(monkeypatch):
    monkeypatch.setenv("RDTURBO_CTRL_CACHE", "1")
    ns = _ns(S._CC_BOOT_RM, torch=None)
    rm = types.SimpleNamespace(sim=types.SimpleNamespace(_sim_step_counter=7))
    assert ns["_rdturbo_cc_action"](rm, _MC(7), 0) == {"a": 1}          # hit
    assert ns["_rdturbo_cc_action"](rm, _MC(6), 0) == {"a": 2}          # counter moved: miss
    no_counter = types.SimpleNamespace(sim=object())
    assert ns["_rdturbo_cc_action"](no_counter, _MC(None), 0) == {"a": 2}   # counter missing: never hit (None == None guard)


def test_accounting_reports_and_strict_exit(monkeypatch, tmp_path):
    status = tmp_path / "status.json"
    monkeypatch.setenv("RDTURBO_STATUS", str(status))
    monkeypatch.setenv("RDTURBO_STRICT", "1")
    codes = []
    monkeypatch.setattr(os, "_exit", lambda c: codes.append(c))
    ns = _ns(H._ACCOUNTING_HELPERS)
    env = types.SimpleNamespace(success_nums=3, fail_nums=7, unstable_nums=0)
    ns["_rdturbo_accounting"](env, 10)
    ns["_rdturbo_strict_exit"]()
    assert json.loads(status.read_text())["ok"] is True and codes == []
    ns["_rdturbo_swallowed"](RuntimeError("boom"))
    ns["_rdturbo_accounting"](env, 10)
    st = json.loads(status.read_text())
    assert st["ok"] is False and st["swallowed_batches"] == 1 and codes == []   # no exit before upstream's cleanup
    ns["_rdturbo_strict_exit"]()
    assert codes == [4]


# --------------------------------------------------------------------------------------------------- usd_last

KEYS = ["/physics/updateToUsd", "/physics/updateVelocitiesToUsd", "/physics/updateParticlesToUsd", "/physics/updateResidualsToUsd"]


@pytest.fixture
def ul(monkeypatch):
    store = {k: True for k in KEYS}
    store["/physics/outputVelocitiesLocalSpace"] = False
    flushed = []

    class Settings:
        def get_as_bool(self, k):
            return store.get(k, False)

        def get(self, k):
            return store.get(k)

        def set_bool(self, k, v):
            store[k] = v

    carb = types.ModuleType("carb")
    carb.settings = types.SimpleNamespace(get_settings=lambda: Settings())
    physx = types.ModuleType("omni.physx")
    physx.get_physx_interface = lambda: types.SimpleNamespace(update_transformations=lambda *a: flushed.append(a))
    omni = types.ModuleType("omni")
    omni.physx = physx
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.physx", physx)

    def make(mode=None, usd_last="1"):
        monkeypatch.setenv("RDTURBO_USD_LAST", usd_last)
        if mode is None:
            monkeypatch.delenv("RDTURBO_USD_LAST_MODE", raising=False)
        else:
            monkeypatch.setenv("RDTURBO_USD_LAST_MODE", mode)
        return _ns(S._UL_BOOT)
    return make, store, flushed


class _Q:
    def __init__(self, n):
        self.q = queue.Queue()
        for i in range(n):
            self.q.put(i)

    def is_empty(self):
        return self.q.empty()

    def pop(self):
        return self.q.get()


def _env(types_per_env, n_envs=3, substeps=10):
    cq = [_Q(substeps) for _ in range(n_envs)]
    by_env = [dict(types_per_env) for _ in range(n_envs)]   # RoboDojo: a list indexed by env
    return types.SimpleNamespace(scene_manager=types.SimpleNamespace(layout_manager=types.SimpleNamespace(instance_type_by_env=by_env)),
                                 robot_manager=types.SimpleNamespace(control_manager=types.SimpleNamespace(control_queue=cq))), cq


def _run_action(ns, env, cq, store):
    on_at, during = [], []
    i = 0
    while not any(q.is_empty() for q in cq):
        for q in cq:
            q.pop()
        on = ns["_rdturbo_ul_substep_pre"](env, [0, 1, 2])
        during.append(tuple(store[k] for k in KEYS))
        ns["_rdturbo_ul_substep_post"](on)
        if on:
            on_at.append(i)
        i += 1
    ns["_rdturbo_ul_after_action"](env, [0, 1, 2])
    return on_at, during


def test_usd_last_auto_rigid_scene_uses_flush(ul):
    make, store, flushed = ul
    ns = make()
    env, cq = _env({"b0": "rigid", "t": "table"})
    ns["_rdturbo_ul_episode_begin"](env)
    assert ns["_RDTURBO_UL_STATE"]["mode"] == "flush" and not any(store[k] for k in KEYS)
    on_at, during = _run_action(ns, env, cq, store)
    assert on_at == [] and all(d == (False,) * 4 for d in during) and len(flushed) == 1
    ns["_rdturbo_ul_episode_end"]()
    assert all(store[k] for k in KEYS)


@pytest.mark.parametrize("kind", ["garment", "fluid"])
def test_usd_last_auto_particle_scene_writes_back_on_last_substep(ul, kind):
    make, store, flushed = ul
    ns = make()
    env, cq = _env({"x": kind, "b": "rigid"})
    ns["_rdturbo_ul_episode_begin"](env)
    assert ns["_RDTURBO_UL_STATE"]["mode"] == "last"
    on_at, during = _run_action(ns, env, cq, store)
    assert on_at == [9] and during[9] == (True,) * 4 and all(d == (False,) * 4 for d in during[:9])
    assert flushed == [] and not any(store[k] for k in KEYS)
    ns["_rdturbo_ul_episode_end"]()
    assert all(store[k] for k in KEYS) and flushed == []


def test_usd_last_keep_particles_and_invalid_mode(ul, monkeypatch):
    make, store, _ = ul
    ns = make("keep-particles")
    env, _ = _env({"x": "fluid"})
    ns["_rdturbo_ul_episode_begin"](env)
    assert store["/physics/updateParticlesToUsd"] is True and store["/physics/updateToUsd"] is False
    ns["_rdturbo_ul_episode_end"]()

    class _Exit(Exception):
        pass

    def _exit(code):
        raise _Exit(code)
    monkeypatch.setattr(os, "_exit", _exit)
    with pytest.raises(_Exit) as ei:    # an invalid mode stops the process at import, before any batch runs
        make("fast")
    assert ei.value.args == (2,)
    make("fast", usd_last="0")          # only checked when the switch is on


def test_usd_last_off_or_fabric_is_noop(ul):
    make, store, flushed = ul
    env, _ = _env({"x": "rigid"})
    off = make(usd_last="0")
    off["_rdturbo_ul_episode_begin"](env)
    assert off["_RDTURBO_UL_STATE"]["snap"] is None and all(store[k] for k in KEYS)
    store["/physics/updateToUsd"] = False   # e.g. a Fabric stage: nothing is written to USD anyway
    on = make()
    on["_rdturbo_ul_episode_begin"](env)
    assert on["_RDTURBO_UL_STATE"]["snap"] is None and store["/physics/updateVelocitiesToUsd"] is True


# --------------------------------------------------------------------------------------------------- Pi_05 loop

class FakeClient:
    def __init__(self, chunk=3):
        self.calls = []
        self.chunk = chunk

    def call(self, func_name=None, obs=None):
        self.calls.append((func_name, None if obs is None else len(obs)))
        if func_name == "get_action_batch":
            return [[(e, i) for i in range(self.chunk)] for e in obs]
        return None


def _fake_env_class(module_name, steps_per_env=5, n=2):
    class FakeEnv:
        num_envs = n

        def __init__(self):
            self.env_seeds = [100 + e for e in range(n)]
            self.done = [0] * n
            self.obs_calls = []

        def is_episode_end(self):
            return all(d >= steps_per_env for d in self.done)

        def get_running_env_idx_list(self):
            return [e for e in range(n) if self.done[e] < steps_per_env]

        def get_obs_batch(self, env_idx_list=None, last_frame=False, **kw):
            self.obs_calls.append((tuple(env_idx_list), tuple(sorted(kw))))
            return [{"vision": {"cam_head": {"color": np.zeros((2, 2, 3), np.uint8)}}} for _ in env_idx_list]

        def take_action_batch(self, actions, env_idx_list):
            for e in env_idx_list:
                self.done[e] += 1
    FakeEnv.__module__ = module_name
    return FakeEnv


def _loop_ns(monkeypatch, name="rdturbo_fake_eval_env", vo=False):
    mod = types.ModuleType(name)
    if vo:
        mod._RDTURBO_VO = "1"
    monkeypatch.setitem(sys.modules, name, mod)
    return _ns(H.PI05_LOOP), _fake_env_class(name)


def test_pi05_loop_defaults_match_upstream_call_sequence(monkeypatch):
    for v in ("RDTURBO_OBS_EVERY_STEP", "RDTURBO_VIDEO_EVERY", "RDTURBO_VIDEO_ONLY_OBS", "RDTURBO_RECORD_ACTIONS", "RDTURBO_REPLAY_ACTIONS"):
        monkeypatch.delenv(v, raising=False)
    ns, Env = _loop_ns(monkeypatch)
    env, client = Env(), FakeClient(chunk=3)
    ns["eval_one_episode_batch"](env, client)
    # upstream: reset; per chunk: update_obs, get_action; after every non-final in-chunk action: update_obs
    expected = [("reset", None)]
    done = [0, 0]
    while not all(d >= 5 for d in done):
        running = [e for e in range(2) if done[e] < 5]
        expected += [("update_obs_batch", len(running)), ("get_action_batch", len(running))]
        for i in range(3):
            for e in running:
                done[e] += 1
            if all(d >= 5 for d in done) or i + 1 == 3:
                break
            running = [e for e in running if done[e] < 5]
            expected.append(("update_obs_batch", len(running)))
    assert client.calls == expected


def test_pi05_loop_chunk_start_upload_and_video_only(monkeypatch):
    monkeypatch.setenv("RDTURBO_OBS_EVERY_STEP", "0")
    monkeypatch.setenv("RDTURBO_VIDEO_ONLY_OBS", "1")
    ns, Env = _loop_ns(monkeypatch, vo=True)
    env, client = Env(), FakeClient(chunk=3)
    ns["eval_one_episode_batch"](env, client)
    uploads = [c for c in client.calls if c[0] == "update_obs_batch"]
    gets = [c for c in client.calls if c[0] == "get_action_batch"]
    assert len(uploads) == len(gets)                          # one upload per chunk start
    video_only = [c for c in env.obs_calls if c[1] == ("rdturbo_video_only",)]
    assert video_only and all(c[1] in ((), ("rdturbo_video_only",)) for c in env.obs_calls)


def test_pi05_loop_record_then_replay(monkeypatch, tmp_path):
    rec = tmp_path / "actions.pkl"
    monkeypatch.setenv("RDTURBO_RECORD_ACTIONS", str(rec))
    ns, Env = _loop_ns(monkeypatch, name="rdturbo_fake_rec")
    env, client = Env(), FakeClient(chunk=3)
    ns["eval_one_episode_batch"](env, client)
    recorded = pickle.loads(rec.read_bytes())
    assert recorded and all(k[0] in (100, 101) for k in recorded)
    monkeypatch.delenv("RDTURBO_RECORD_ACTIONS")
    monkeypatch.setenv("RDTURBO_REPLAY_ACTIONS", str(rec))
    ns2, Env2 = _loop_ns(monkeypatch, name="rdturbo_fake_rep")
    env2, client2 = Env2(), FakeClient(chunk=3)
    ns2["eval_one_episode_batch"](env2, client2)
    assert not [c for c in client2.calls if c[0] == "get_action_batch"] and ns2["_RDTURBO_RR"]["hit"] > 0 and ns2["_RDTURBO_RR"]["miss"] == 0


def test_empty_switch_values_mean_upstream(monkeypatch):
    # an exported-but-empty switch must not turn a fast path on (off.env and shells produce these)
    for boot, name, var in ((S._CC_BOOT_RM, "_RDTURBO_CC", "RDTURBO_CTRL_CACHE"), (S._UL_BOOT, "_RDTURBO_UL", "RDTURBO_USD_LAST")):
        monkeypatch.setenv(var, "")
        assert _ns(boot)[name] == "0"
        monkeypatch.setenv(var, " check ")
        assert _ns(boot)[name] == "check"


def test_switch_values_outside_the_set_mean_upstream(monkeypatch):
    boots = ((S._CC_BOOT_CM, "_RDTURBO_CC", "RDTURBO_CTRL_CACHE"), (S._CC_BOOT_RM, "_RDTURBO_CC", "RDTURBO_CTRL_CACHE"),
             (S._PT_BOOT, "_RDTURBO_PT", "RDTURBO_PARENT_CACHE"), (S._CP_BOOT, "_RDTURBO_CP", "RDTURBO_PCI_COPY"),
             (S._BT_BOOT, "_RDTURBO_BT", "RDTURBO_BATCH_TENSOR"))
    for boot, name, var in boots:
        for off in ("", " ", "0", "false", "off", "True"):
            monkeypatch.setenv(var, off)
            assert _ns(boot, torch=None)[name] == "0", (var, off)
        for on in ("1", " 1 ", "check"):
            monkeypatch.setenv(var, on)
            assert _ns(boot, torch=None)[name] == on.strip(), (var, on)
    for v, want in (("1", "1"), ("check", "0"), (" ", "0"), ("on", "0")):   # VIDEO_MV has no check mode
        monkeypatch.setenv("RDTURBO_VIDEO_MV", v)
        assert _ns(S._VW_BOOT)["_RDTURBO_VW"] == want


def _no_tooling(monkeypatch):
    for v in ("RDTURBO_OBS_EVERY_STEP", "RDTURBO_VIDEO_EVERY", "RDTURBO_VIDEO_ONLY_OBS", "RDTURBO_RECORD_ACTIONS",
              "RDTURBO_REPLAY_ACTIONS", "RDTURBO_REPLAY_STRICT", "RDTURBO_STATE_TRACE", "RDTURBO_PARTICLE_TRACE", "RDTURBO_OBS_DUMP"):
        monkeypatch.delenv(v, raising=False)


def test_pi05_loop_needs_no_env_seeds_with_tooling_off(monkeypatch):
    # an evaluation env without env_seeds (e.g. XPolicyLab's debug client) must run as upstream does
    _no_tooling(monkeypatch)
    ns, Env = _loop_ns(monkeypatch, name="rdturbo_fake_noseeds")
    env, client = Env(), FakeClient(chunk=3)
    del env.env_seeds
    ns["eval_one_episode_batch"](env, client)
    assert client.calls[0] == ("reset", None) and any(c[0] == "get_action_batch" for c in client.calls)


def test_pi05_loop_partial_replay_miss_keeps_recorded_envs(monkeypatch, tmp_path):
    _no_tooling(monkeypatch)
    rep = tmp_path / "actions.pkl"
    # chunk 0 recorded for both envs, chunk 1 only for layout 101 (marked so recorded actions can be told apart)
    rep.write_bytes(pickle.dumps({(100, 0): [("rec", 100, 0, i) for i in range(3)], (101, 0): [("rec", 101, 0, i) for i in range(3)],
                                  (101, 1): [("rec", 101, 1, i) for i in range(3)]}))
    monkeypatch.setenv("RDTURBO_REPLAY_ACTIONS", str(rep))
    monkeypatch.setenv("RDTURBO_RECORD_ACTIONS", "1")
    monkeypatch.setenv("RDTURBO_TRACE", str(tmp_path / "trace_w0.jsonl"))
    ns, Env = _loop_ns(monkeypatch, name="rdturbo_fake_partial")
    env, client = Env(), FakeClient(chunk=3)
    got, real = [], ns["_rdturbo_actions"]
    ns["_rdturbo_actions"] = lambda *a: got.append(real(*a)) or got[-1]    # the loop looks it up in this namespace
    ns["eval_one_episode_batch"](env, client)
    assert got[1][1][0][0] == "rec" and got[1][0][0][0] != "rec"      # chunk 1: env of layout 101 keeps its recording
    assert ns["_RDTURBO_RR"]["miss"] == 1 and ns["_RDTURBO_RR"]["hit"] >= 3   # counted per env
    rec = pickle.loads((tmp_path / "actions_rec_w0.pkl").read_bytes())
    assert {(100, 0), (101, 0), (100, 1), (101, 1)} <= set(rec)          # record-while-replay keeps every executed chunk


def test_flush_merges_what_an_earlier_process_recorded(monkeypatch, tmp_path):
    _no_tooling(monkeypatch)
    rec = tmp_path / "actions.pkl"
    rec.write_bytes(pickle.dumps({(7, 0): ["before restart"]}))           # written by the process that crashed
    monkeypatch.setenv("RDTURBO_RECORD_ACTIONS", str(rec))
    ns, Env = _loop_ns(monkeypatch, name="rdturbo_fake_restart")
    ns["eval_one_episode_batch"](Env(), FakeClient(chunk=3))
    merged = pickle.loads(rec.read_bytes())
    assert merged[(7, 0)] == ["before restart"] and (100, 0) in merged


def test_video_every_whitespace_means_one(monkeypatch):
    _no_tooling(monkeypatch)
    monkeypatch.setenv("RDTURBO_OBS_EVERY_STEP", "0")
    monkeypatch.setenv("RDTURBO_VIDEO_EVERY", " ")
    ns, Env = _loop_ns(monkeypatch, name="rdturbo_fake_ve")
    env, client = Env(), FakeClient(chunk=3)
    ns["eval_one_episode_batch"](env, client)
    starts = len([c for c in client.calls if c[0] == "get_action_batch"])
    assert len(env.obs_calls) > starts                                   # in-chunk frames still rendered for the video


def test_revoke_listeners_selection_is_tolerant():
    from robodojo_turbo.runtime import rdturbo_revoke_listeners as R
    for v in ("1", " 1 ", "1\r", "all", "True", "on"):
        assert R.select(v) == R.DENY, v
    assert R.select("nonexistent") == {}
