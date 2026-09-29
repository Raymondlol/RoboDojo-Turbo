"""Pure-Python tests for the evaluation tooling (no Isaac Sim needed)."""
import json
import os
import pickle
import tempfile
import time

import numpy as np
import pytest

from robodojo_turbo.runtime.rdturbo_trace import install_env_tracing
from robodojo_turbo.tools import paired_compare, particle_compare, state_compare
from robodojo_turbo.tools.merge import merge_results
from robodojo_turbo.tools.report import analyze_client, build_report, exclusive_times, render_md, union_length
from robodojo_turbo.tools.shard import format_id_spec, parse_id_spec, plan_shards


def test_id_spec_roundtrip():
    assert parse_id_spec("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]
    assert format_id_spec([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert parse_id_spec(format_id_spec(range(25))) == list(range(25))


def test_plan_shards_covers_every_layout_once():
    for n, w, e in [(25, 1, 10), (25, 2, 7), (25, 2, 5), (25, 3, 5), (25, 5, 5), (25, 4, 10), (7, 3, 10), (25, 2, 13)]:
        plan = plan_shards(n, w, e)
        ids = [i for row in plan for i in parse_id_spec(row["ids"])]
        assert sorted(ids) == list(range(n)), (n, w, e, plan)
        assert len(ids) == len(set(ids))
        b = [row["batches"] for row in plan]
        assert max(b) - min(b) <= 1, (n, w, e, plan)


def test_plan_shards_examples():
    assert [r["ids"] for r in plan_shards(25, 2, 7)] == ["0-13", "14-24"]
    assert [r["ids"] for r in plan_shards(25, 2, 5)] == ["0-14", "15-24"]
    assert [r["ids"] for r in plan_shards(25, 3, 5)] == ["0-9", "10-19", "20-24"]
    assert [r["ids"] for r in plan_shards(25, 1, 10)] == ["0-24"]


def _result(details):
    return {"details": {str(i): d for i, d in enumerate(details)}, "eval_time": len(details)}


def test_merge_scores_like_official():
    r0 = _result([{"layout_id": 0, "success": True, "score": 1.0}, {"layout_id": 1, "success": False, "score": 0.15}])
    r1 = _result([{"layout_id": 2, "success": False, "score": 0.0}, {"layout_id": 3, "success": True, "score": 1.0}])
    m = merge_results([r0, r1], expected_ids=range(5), sources=["a", "b"])
    assert m["eval_time"] == 4 and m["successes"] == 2
    assert abs(m["success_rate"] - 0.5) < 1e-9
    assert abs(m["score"] - (1.0 + 0.15 + 0.0 + 1.0) / 4 * 100) < 1e-9
    assert m["missing"] == [4] and m["duplicates"] == []
    assert [d["layout_id"] for d in m["details"].values()] == [0, 1, 2, 3]


def test_merge_flags_duplicates():
    r0 = _result([{"layout_id": 0, "success": True, "score": 1.0}])
    r1 = _result([{"layout_id": 0, "success": False, "score": 0.0}])
    m = merge_results([r0, r1], expected_ids=range(1))
    assert m["duplicates"] == [0] and m["eval_time"] == 1 and m["details"]["0"]["success"] is True


def test_merge_reports_robodojo_unstable_layouts():
    r = _result([{"layout_id": i, "success": False, "score": 0.0} for i in (0, 1, 2, 3, 5, 6, 7, 9)])
    m = merge_results([r], expected_ids=range(10), unstable_count=2)
    assert m["missing"] == [] and m["unstable_layouts"] == [4, 8] and m["eval_time"] == 8
    m = merge_results([r], expected_ids=range(10), unstable_count=1)     # count does not explain the gap: still missing
    assert m["missing"] == [4, 8] and m["unstable_layouts"] == []


def test_exclusive_times_nested():
    spans = [
        {"id": 1, "parent": None, "kind": "episode", "dt": 10.0, "t0": 0, "t1": 10},
        {"id": 2, "parent": 1, "kind": "sim_step", "dt": 4.0, "t0": 0, "t1": 4},
        {"id": 3, "parent": 1, "kind": "reward", "dt": 3.0, "t0": 4, "t1": 7},
        {"id": 4, "parent": 3, "kind": "obs", "dt": 1.0, "t0": 5, "t1": 6},
        {"id": 5, "parent": 1, "kind": "ws", "fn": "get_action_batch", "dt": 2.0, "t0": 7, "t1": 9},
    ]
    excl, cnt = exclusive_times(spans)
    assert abs(excl["episode"] - 1.0) < 1e-9
    assert abs(excl["reward"] - 2.0) < 1e-9
    assert abs(excl["obs"] - 1.0) < 1e-9
    assert abs(excl["ws:get_action_batch"] - 2.0) < 1e-9
    assert cnt["sim_step"] == 1


def test_union_length():
    assert union_length([(0, 2), (1, 3), (5, 6)]) == 4.0
    assert union_length([]) == 0.0


def test_tracer_and_install_on_fake_env():
    class FakeClient:
        def call(self, func_name=None, obs=None):
            time.sleep(0.001)
            return [[0] * 3] * (len(obs) if isinstance(obs, list) else 1)

    class FakeEnv:
        num_envs = 3

        def __init__(self):
            self.model_client = FakeClient()

        def get_obs_batch(self, env_idx_list=None, last_frame=False):
            return [{"vision": {"cam": {"color": bytes(10)}}} for _ in (env_idx_list or range(self.num_envs))]

        def take_action_batch(self, actions_list, env_idx_list=None):
            return None

        def is_episode_end(self):
            self.get_obs_batch([0], last_frame=True)  # nested call must become a child span
            return True

        def reset(self, seed=None, options=None):
            return None

        def run_eval(self):
            self.reset(seed=[0, 1, None])
            obs = self.get_obs_batch([0, 1])
            self.model_client.call(func_name="update_obs_batch", obs=obs)
            acts = self.model_client.call(func_name="get_action_batch", obs=[0, 1])
            self.take_action_batch([a[0] for a in acts], [0, 1])
            self.is_episode_end()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.jsonl")
        env = FakeEnv()
        assert install_env_tracing(env, path) is not None
        env.run_eval()
        rows = [json.loads(line) for line in open(path)]
        kinds = [(r.get("kind"), r.get("fn")) for r in rows if "dt" in r]
        assert ("episode", None) in kinds and ("ws", "update_obs_batch") in kinds and ("ws", "get_action_batch") in kinds
        obs_in_reward = [r for r in rows if r.get("kind") == "obs" and r.get("last")]
        reward = [r for r in rows if r.get("kind") == "reward"][0]
        assert obs_in_reward and obs_in_reward[0]["parent"] == reward["id"]
        up = [r for r in rows if r.get("fn") == "update_obs_batch"][0]
        assert up["n"] == 2 and up["bytes"] == 20
        rep = analyze_client({"w0": rows})
        assert rep["episodes"] == 2 and rep["workers"] == 1
        assert rep["per_worker"]["w0"]["counts"]["sim_step"] == 1


def test_build_report_multi_worker(tmp_path):
    def mk(w, off):
        rows = [{"kind": "install", "ev": 1, "t": off, "w": w, "num_envs": 5, "layout_ids": "0-4", "obs_every_step": "0", "video_every": "5"},
                {"id": 1, "parent": None, "kind": "episode", "dt": 10.0, "t0": off, "t1": off + 10, "w": w},
                {"id": 2, "parent": 1, "kind": "ws", "fn": "get_action_batch", "dt": 4.0, "t0": off + 1, "t1": off + 5, "w": w},
                {"id": 3, "parent": 1, "kind": "sim_step", "dt": 3.0, "t0": off + 5, "t1": off + 8, "w": w, "n": 5},
                {"id": 4, "parent": None, "kind": "reset", "dt": 1.0, "t0": off - 1, "t1": off, "w": w, "n": 5}]
        p = tmp_path / f"trace_{w}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(p)
    paths = [mk("w0", 100.0), mk("w1", 102.0)]
    gpu = tmp_path / "gpu.csv"
    gpu.write_text("timestamp, memory.used [MiB], utilization.gpu [%], power.draw [W]\n"
                   "2026/09/06 17:00:00.000, 9000 MiB, 5 %, 100 W\n2026/09/06 17:00:01.000, 9500 MiB, 60 %, 200 W\n")
    rep = build_report(paths, gpu={"sim": str(gpu)}, baseline_wall_s=60.0, title="t")
    c = rep["client"]
    assert c["workers"] == 2 and c["episodes"] == 10
    assert abs(c["policy_requests_s_sum"] - 8.0) < 1e-9 and abs(c["policy_requests_s_union"] - 6.0) < 1e-9
    assert abs(c["policy_overlap_ratio"] - 8.0 / 6.0) < 1e-9
    assert rep["gpu"]["sim"]["mem_peak_mib"] == 9500 and abs(rep["gpu"]["sim"]["util_mean"] - 32.5) < 1e-9
    md = render_md(rep)
    assert "Bubble report" in md and "w1" in md


def test_paired_compare_mcnemar_and_mde():
    A = {i: (1.0 if i < 5 else 0.0, i < 5) for i in range(20)}
    B = {i: (1.0 if 2 <= i < 7 else 0.0, 2 <= i < 7) for i in range(20)}
    r = paired_compare.compare(A, B, boot=2000)
    assert r["mcnemar"] == {"only_b": 2, "only_a": 2, "p": 1.0}
    assert r["n"] == 20 and abs(r["mean_diff"]) < 1e-9 and r["mde80_score"] > 0
    assert paired_compare.mcnemar_exact(0, 6) < 0.05


def test_paired_compare_pools_pairs_with_a_stratified_bootstrap(tmp_path, capsys):
    def res(path, scores):
        path.write_text(json.dumps({"details": {str(i): {"layout_id": i, "score": v, "success": v >= 1.0}
                                                for i, v in enumerate(scores)}}))
        return str(path)
    # two "tasks" that share layout ids 0-3: pooling must keep them apart
    a1 = res(tmp_path / "a1.json", [1, 0, 0, 0]); b1 = res(tmp_path / "b1.json", [1, 1, 0, 0])
    a2 = res(tmp_path / "a2.json", [0, 0, 1, 1]); b2 = res(tmp_path / "b2.json", [0, 0, 1, 0])
    assert paired_compare.main([a1, b1, a2, b2, "--boot", "500"]) == 0
    out = capsys.readouterr().out
    assert "-- pair 1" in out and "-- pair 2" in out and "pooled over 2 pairs" in out
    assert "n=8 layouts" in out and "only-B 1, only-A 1" in out
    with pytest.raises(SystemExit):
        paired_compare.main([a1, b1, a2])            # odd number of results


def test_state_compare_bitwise_and_divergence(tmp_path):
    a = {(0, 0, s): np.arange(5, dtype=np.float64) + s for s in range(4)}
    b = {k: v.copy() for k, v in a.items()}
    b[(0, 0, 2)] = b[(0, 0, 2)] + 1e-12
    pa, pb = tmp_path / "a.pkl", tmp_path / "b.pkl"
    pa.write_bytes(pickle.dumps(a))
    pb.write_bytes(pickle.dumps(b))
    r = state_compare.compare(a, b)
    assert r["identical"] == 3 and r["first_divergence"][0][0] == (0, 0, 2)
    assert state_compare.main([str(pa), str(pa)]) == 0
    assert state_compare.main([str(pa), str(pb)]) == 1


def test_particle_compare_flags_frozen_points(tmp_path):
    rng = np.random.default_rng(0)
    ref = {(0, 0, s, "fluid:w"): rng.random((50, 3)).astype(np.float32) for s in range(6)}
    frozen = {k: ref[(0, 0, 0, "fluid:w")].copy() for k in ref}
    pr, pf = tmp_path / "r.pkl", tmp_path / "f.pkl"
    pr.write_bytes(pickle.dumps(ref))
    pf.write_bytes(pickle.dumps(frozen))
    st = particle_compare.frozen_stats(frozen)
    assert st[(0, "fluid:w")][0] == 1.0 and particle_compare.frozen_stats(ref)[(0, "fluid:w")][0] == 0.0
    assert particle_compare.main([str(pr), str(pr)]) == 0
    assert particle_compare.main([str(pr), str(pf)]) == 1


def test_state_compare_usd_keys_do_not_decide(tmp_path):
    import pickle
    phys = {(0, 0, s): np.arange(4, dtype=np.float64) for s in range(3)}
    a = {**phys, **{(0, 0, s, "usd"): np.zeros(16) for s in range(3)}}
    b = {**phys, **{(0, 0, s, "usd"): np.full(16, 1e-4) for s in range(3)}}   # stale USD, identical physics
    pa, pb = tmp_path / "a.pkl", tmp_path / "b.pkl"
    pa.write_bytes(pickle.dumps(a))
    pb.write_bytes(pickle.dumps(b))
    assert state_compare.main([str(pa), str(pb)]) == 0
    b[(0, 0, 1)] = b[(0, 0, 1)] + 1e-12
    pb.write_bytes(pickle.dumps(b))
    assert state_compare.main([str(pa), str(pb)]) == 1
