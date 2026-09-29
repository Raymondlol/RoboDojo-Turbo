"""Patch matrix against a real upstream checkout (GPU-free). Set RDTURBO_TEST_UPSTREAM to a pristine RoboDojo tree at the
pinned commit with XPolicyLab checked out (git archive of both is enough; assets and third_party are not needed):

    git clone https://github.com/RoboDojo-Benchmark/RoboDojo && cd RoboDojo && git checkout <pinned commit>
    git submodule update --init XPolicyLab   # at the pinned XPolicyLab commit
    RDTURBO_TEST_UPSTREAM=$PWD pytest -q tests/test_patch_matrix.py

For every profile and every single patch: apply, compile, refuse a second apply, revert, and require a byte-exact tree.
"""
import hashlib
import os
import shutil
import sys
import types

import pytest

from robodojo_turbo import engine
from robodojo_turbo.patches import REGISTRY
from robodojo_turbo.patches import harness as H

UPSTREAM = os.environ.get("RDTURBO_TEST_UPSTREAM")
pytestmark = pytest.mark.skipif(not UPSTREAM, reason="RDTURBO_TEST_UPSTREAM not set")
DIRS = ("src", "env", "utils", "scripts", "XPolicyLab/policy/Pi_05", "XPolicyLab/policy/Pi_0", "env_cfg")


def _copy(dst):
    for d in DIRS:
        s = os.path.join(UPSTREAM, d)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst, d), symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return str(dst)


def _digest(root):
    out = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            p = os.path.join(base, f)
            if os.path.islink(p):
                out[os.path.relpath(p, root)] = "link:" + os.readlink(p)
            else:
                with open(p, "rb") as fh:
                    out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


@pytest.fixture
def tree(tmp_path):
    root = _copy(tmp_path / "rd")
    return root, _digest(root)


@pytest.mark.parametrize("profile", ["harness", "speedup", "all"])
def test_profile_apply_revert_exact(tree, profile):
    root, before = tree
    patches = engine.select(REGISTRY, profile)
    engine.apply(root, patches, log=lambda *_: None)
    assert engine.status(root)["patched"] and not engine.status(root)["modified_since_apply"]
    with pytest.raises(engine.PatchError):
        engine.apply(root, patches, log=lambda *_: None)
    engine.revert(root, log=lambda *_: None)
    assert _digest(root) == before


@pytest.mark.parametrize("name", [p.name for p in REGISTRY])
def test_single_patch_apply_revert_exact(tree, name):
    root, before = tree
    engine.apply(root, engine.select(REGISTRY, None, only=[name]), log=lambda *_: None)
    engine.revert(root, log=lambda *_: None)
    assert _digest(root) == before


def test_refuses_unpinned_files(tree):
    root, _ = tree
    main = os.path.join(root, "src/eval_client/main.py")
    with open(main, "a") as fh:
        fh.write("\n# local edit\n")
    patches = engine.select(REGISTRY, "all")
    with pytest.raises(engine.PatchError, match="pinned"):
        engine.apply(root, patches, log=lambda *_: None)
    engine.apply(root, patches, force_unpinned=True, log=lambda *_: None)
    assert engine.status(root)["unpinned"]
    engine.revert(root, log=lambda *_: None)


def test_interrupted_apply_reverts_and_removes_installed_files(tree, monkeypatch):
    root, before = tree
    patches = engine.select(REGISTRY, "all")
    last = patches[-1]
    broken = engine.Patch(last.name, last.profile, last.targets, lambda tr: (_ for _ in ()).throw(KeyboardInterrupt()),
                          last.requires, last.installs, last.switches, last.api, last.doc)
    with pytest.raises(KeyboardInterrupt):
        engine.apply(root, patches[:-1] + [broken], log=lambda *_: None)
    assert engine.load_manifest(root) is None and _digest(root) == before   # utils/rdturbo_*.py removed too
    engine.apply(root, patches, log=lambda *_: None)                        # and the tree can be patched again
    engine.revert(root, log=lambda *_: None)
    assert _digest(root) == before


def test_missing_target_stale_state_and_exclude_typo(tree):
    root, before = tree
    patches = engine.select(REGISTRY, "all")
    with pytest.raises(engine.PatchError, match="unknown patch names"):
        engine.select(REGISTRY, "all", exclude=["usd_lsat"])
    rigid = os.path.join(root, "env/scene_manager/objects/rigid.py")
    saved = open(rigid, "rb").read()
    os.remove(rigid)
    with pytest.raises(engine.PatchError, match="missing"):
        engine.apply(root, patches, force_unpinned=True, log=lambda *_: None)
    assert not os.path.exists(os.path.join(root, engine.STATE_DIR))
    with open(rigid, "wb") as fh:
        fh.write(saved)
    os.makedirs(os.path.join(root, engine.STATE_DIR, "backup"))
    with pytest.raises(engine.PatchError, match="without a manifest"):
        engine.apply(root, patches, log=lambda *_: None)
    shutil.rmtree(os.path.join(root, engine.STATE_DIR))
    assert _digest(root) == before


def test_revert_refuses_to_discard_edits_without_force(tree):
    root, before = tree
    engine.apply(root, engine.select(REGISTRY, "all"), log=lambda *_: None)
    main = os.path.join(root, "src/eval_client/main.py")
    with open(main, "a") as fh:
        fh.write("#x\n")
    with pytest.raises(engine.PatchError, match="changed since apply"):
        engine.revert(root, log=lambda *_: None)
    engine.revert(root, log=lambda *_: None, force=True)
    assert _digest(root) == before


def test_dry_run_finds_broken_anchors(tree):
    root, before = tree
    patches = engine.select(REGISTRY, "all")
    assert engine.dry_run(root, patches) == []
    main = os.path.join(root, "src/eval_client/main.py")
    s = open(main).read()
    with open(main, "w") as fh:
        fh.write(s.replace("_delete_resume_manifest(env)\n", "_delete_resume_manifest_renamed(env)\n"))
    problems = engine.dry_run(root, patches)
    assert any(p.startswith("anchor: accounting") for p in problems) and any(p.startswith("pin: ") for p in problems)
    assert engine.load_manifest(root) is None


def test_lost_state_dir_is_detected_and_explained(tree):
    root, before = tree
    engine.apply(root, engine.select(REGISTRY, "all"), log=lambda *_: None)
    shutil.rmtree(os.path.join(root, engine.STATE_DIR))          # what `git clean -fd` does
    st = engine.status(root)
    assert st["patched"] and st["status"] == "state-missing" and "utils/rdturbo_trace.py" in st["marked_files"]
    with pytest.raises(engine.PatchError, match="git -C .* checkout --"):
        engine.revert(root, log=lambda *_: None)
    with pytest.raises(engine.PatchError, match="restore the upstream files"):
        engine.apply(root, engine.select(REGISTRY, "all"), log=lambda *_: None)
    from robodojo_turbo import cli
    assert any("is missing" in p for p in cli.switch_problems(root, environ={}))


def test_force_revert_keeps_files_replaced_after_apply(tree):
    root, before = tree
    engine.apply(root, engine.select(REGISTRY, "all"), log=lambda *_: None)
    save = os.path.join(root, "utils/save_file.py")
    with open(save, "w") as fh:
        fh.write("# a newer upstream version\n")
    engine.revert(root, log=lambda *_: None, force=True)
    assert open(save).read() == "# a newer upstream version\n"
    after = _digest(root)
    assert {k for k in before if before[k] != after.get(k)} == {"utils/save_file.py"}


def test_pins_on_patched_tree_checks_backups(tree, capsys):
    root, _ = tree
    from robodojo_turbo import cli
    engine.apply(root, engine.select(REGISTRY, "all"), log=lambda *_: None)
    assert cli.main(["pins", "--root", root]) == 0
    assert "tree is patched" in capsys.readouterr().out


def test_check_env_values_and_exclude_dependents(tree):
    root, _ = tree
    from robodojo_turbo import cli
    engine.apply(root, engine.select(REGISTRY, "all"), log=lambda *_: None)
    assert cli.switch_problems(root, environ={"RDTURBO_CTRL_CACHE": "", "RDTURBO_FAST": "1", "RDTURBO_GC": "freeze,log"}) == []
    probs = cli.switch_problems(root, environ={"RDTURBO_CTRL_CACHE": "false", "RDTURBO_GC": "1", "RDTURBO_USD_LAST_MODE": "keep_particles"})
    assert len(probs) == 3 and all("RDTURBO_" in p for p in probs)
    with pytest.raises(engine.PatchError, match="required by .*usd_last"):
        engine.select(REGISTRY, "all", exclude=["pi05_loop"])


def test_pi05_loop_defaults_replay_upstream_calls_exactly(tree, monkeypatch):
    """Run the real upstream Pi_05 eval_one_episode_batch and our replacement (all switches off) on the same fake env."""
    from tests.test_injected import FakeClient, _fake_env_class
    root, _ = tree
    for v in ("RDTURBO_OBS_EVERY_STEP", "RDTURBO_VIDEO_EVERY", "RDTURBO_VIDEO_ONLY_OBS", "RDTURBO_RECORD_ACTIONS", "RDTURBO_REPLAY_ACTIONS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setitem(sys.modules, "rdturbo_fake_up", types.ModuleType("rdturbo_fake_up"))
    up_src = open(os.path.join(root, H.PI05_DEPLOY)).read()
    up = {}
    exec(compile(up_src, "upstream_deploy", "exec"), up)
    ours = {"os": os}
    exec(compile(H.PI05_LOOP, "pi05_loop", "exec"), ours)
    for chunk in (1, 3, 50):
        Env = _fake_env_class("rdturbo_fake_up", steps_per_env=7, n=3)
        e1, c1 = Env(), FakeClient(chunk)
        up["eval_one_episode_batch"](e1, c1)
        e2, c2 = Env(), FakeClient(chunk)
        ours["eval_one_episode_batch"](e2, c2)
        assert c1.calls == c2.calls and e1.obs_calls == e2.obs_calls and e1.done == e2.done


def _shard_block(src):
    """The injected layout-shard block of seed_manager.py, dedented so it runs on its own."""
    import textwrap
    lines = src.splitlines()
    i = next(k for k, l in enumerate(lines) if "_rdturbo_ids = " in l)
    j = next(k for k in range(i, len(lines)) if "raise ValueError" in lines[k])
    return textwrap.dedent("\n".join(lines[i:j + 1]))


def test_layout_shard_spec_is_stripped_and_must_keep_a_layout(tree, monkeypatch):
    root, _ = tree
    engine.apply(root, engine.select(REGISTRY, None, only=["layout_shards"]), log=lambda *_: None)
    block = _shard_block(open(os.path.join(root, "env/seed_manager/seed_manager.py")).read())
    sys.path.insert(0, root)
    try:
        for spec, want in ((" ", list(range(5))), ("1-2", [1, 2])):
            monkeypatch.setenv("RDTURBO_LAYOUT_IDS", spec)
            ns = {"os": os, "matching_files": list(range(5)), "all_layout_ids": list(range(5))}
            exec(block, ns)
            assert ns["all_layout_ids"] == want
        monkeypatch.setenv("RDTURBO_LAYOUT_IDS", "9")
        with pytest.raises(ValueError, match="keeps none"):
            exec(block, {"os": os, "matching_files": list(range(5)), "all_layout_ids": list(range(5))})
    finally:
        sys.path.remove(root)
        for m in ("utils.rdturbo_trace", "utils"):
            sys.modules.pop(m, None)
