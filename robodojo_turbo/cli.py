"""robodojo-turbo command line. Run it with the Python of your RoboDojo conda env: after `pip install .` as `robodojo-turbo`,
from a checkout as `python -m robodojo_turbo` (same commands).

  robodojo-turbo list                                   patches, profiles and runtime switches
  robodojo-turbo apply  --root <RoboDojo> [--profile all|speedup|harness] [--only a,b] [--exclude c] [--force-unpinned] [--dry-run]
  robodojo-turbo revert --root <RoboDojo> [--force]     restore the upstream files byte for byte
  robodojo-turbo status --root <RoboDojo>
  robodojo-turbo pins   --root <RoboDojo>               check the upstream files against the pinned hashes
  robodojo-turbo check-env --root <RoboDojo>            tree fully applied and unedited; every RDTURBO_* value valid and backed by its patch
  robodojo-turbo env    [--preset speedup|harness|off|verify]   print a preset (load it with: set -a; source <file>; set +a)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from . import __version__
from .engine import (STATE_DIR, PatchError, Tree, apply, check_pins, check_root, dry_run, load_manifest, load_pins,
                     pinned_commits, revert, select, status)
from .patches import PROFILES, REGISTRY

PRESETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


def _csv(v):
    return [x for x in (v or "").split(",") if x]


def _presets():
    return sorted(f[:-4] for f in os.listdir(PRESETS) if f.endswith(".env"))


def cmd_list(_a) -> int:
    for p in REGISTRY:
        deps = f" requires={','.join(p.requires)}" if p.requires else ""
        print(f"{p.profile:8s} {p.name:16s} {', '.join(p.switches)}{deps}")
    print("\nprofiles: " + "; ".join(f"{k} = {'+'.join(sorted(v))}" for k, v in PROFILES.items()))
    print("presets (robodojo_turbo/presets): " + ", ".join(_presets()) + "; the speedup preset needs --profile all")
    return 0


def cmd_apply(a) -> int:
    patches = select(REGISTRY, a.profile, _csv(a.only), _csv(a.exclude))
    if a.dry_run:
        print("would apply: " + ", ".join(p.name for p in patches))
        problems = dry_run(a.root, patches)
        for line in problems:
            print("  problem: " + line)
        fatal = [x for x in problems if not (a.force_unpinned and x.startswith("pin: "))]
        print("dry run: every patch applies to a scratch copy" if not fatal else f"dry run: {len(fatal)} problems")
        return 0 if not fatal else 1
    man = apply(a.root, patches, force_unpinned=a.force_unpinned)
    root = os.path.abspath(a.root)
    print(f"applied {len(man['patches'])} patches to {root}; backups in {STATE_DIR}/ "
          f"(undo with: python -m robodojo_turbo revert --root {root}; revert before git clean/stash/pull on that tree)")
    return 0


def cmd_revert(a) -> int:
    revert(a.root, force=a.force)
    return 0


def cmd_status(a) -> int:
    print(json.dumps(status(a.root), indent=1))
    return 0


def cmd_pins(a) -> int:
    check_root(a.root)
    tree, pins = Tree(a.root), load_pins()
    print(f"pinned upstream: {pinned_commits()}")
    st = status(a.root)
    if st.get("hint"):          # injected code without a state directory: mismatches below are its edits, not drift
        print("note: " + st["hint"])
    man = load_manifest(a.root)
    if man is not None:   # patched: the pinned upstream files are the backups now
        problems = []
        for rel, sha in sorted(man["backups"].items()):
            want = pins["files"].get(rel)
            if want is None or sha not in (want if isinstance(want, list) else [want]):
                problems.append(f"{rel}: pre-apply sha256 {sha[:12]} not in pinned set")
        print(f"tree is patched; checked the pre-apply upstream files kept in {STATE_DIR}/backup")
    else:
        problems = check_pins(tree, REGISTRY, pins)
    for p in problems:
        print("  " + p)
    print("all target files match the pins" if not problems else f"{len(problems)} mismatches")
    return 0 if not problems else 1


# accepted values; unset or empty always means the upstream behaviour
_BOOL = {"0", "1"}
_TRI = {"0", "1", "check"}
_VALUES = {
    "RDTURBO_FAST": _BOOL, "RDTURBO_RGB3": _BOOL, "RDTURBO_NO_PLANNER": _BOOL, "RDTURBO_OFFLINE_ASSETS": _BOOL,
    "RDTURBO_STRICT": _BOOL, "RDTURBO_REPLAY_STRICT": _BOOL, "RDTURBO_OBS_EVERY_STEP": _BOOL, "RDTURBO_OVERLAP_START": _BOOL, "RDTURBO_EVAL_NUM_FORCE": _BOOL,
    "RDTURBO_VIDEO_ONLY_OBS": _TRI, "RDTURBO_CTRL_CACHE": _TRI, "RDTURBO_PARENT_CACHE": _TRI, "RDTURBO_VIDEO_MV": _BOOL,
    "RDTURBO_PCI_COPY": _TRI, "RDTURBO_BATCH_TENSOR": _TRI,
    "RDTURBO_USD_LAST": {"0", "1", "check", "basecheck"}, "RDTURBO_USD_LAST_MODE": {"auto", "flush", "last", "keep-particles"},
}
_PATTERNS = {
    "RDTURBO_GC": (r"(freeze|log)(,(freeze|log))?", "freeze, log or freeze,log"),
    "RDTURBO_VIDEO_EVERY": (r"[0-9]+", "a non-negative integer"),
    "RDTURBO_FFMPEG_THREADS": (r"[1-9][0-9]*", "a positive integer"),
    "RDTURBO_SERVER_MEM_FRACTION": (r"0?\.[0-9]+|1(\.0*)?", "a fraction such as 0.3"),
    "RDTURBO_REVOKE_LISTENERS": (r"0|1|all|[A-Za-z0-9_.]+(,[A-Za-z0-9_.]+)*", "1, all, or a comma-separated list of module/class names"),
}
# switches that only modify another switch, or that the launcher sets per worker from its own arguments
_MODIFIERS = {"RDTURBO_USD_LAST_MODE", "RDTURBO_NV_MIRROR", "RDTURBO_WORKER", "RDTURBO_TRACE", "RDTURBO_STATUS"}


def _active(name: str, value: str) -> bool:
    if name == "RDTURBO_OBS_EVERY_STEP":
        return value == "0"          # 1 / unset = upstream behaviour
    if name == "RDTURBO_VIDEO_EVERY":
        return value not in ("", "1")
    return value not in ("", "0")


def value_problems(environ=None) -> list:
    environ = os.environ if environ is None else environ
    out = []
    for name, raw in sorted(environ.items()):
        v = raw.strip()
        if not v:
            continue
        if name in _VALUES and v not in _VALUES[name]:
            out.append(f"{name}={raw!r}: accepted values are {'|'.join(sorted(_VALUES[name]))} (unset = upstream)")
        elif name in _PATTERNS and not re.fullmatch(_PATTERNS[name][0], v):
            out.append(f"{name}={raw!r}: expected {_PATTERNS[name][1]}")
    return out


def active_switches(environ=None) -> list:
    environ = os.environ if environ is None else environ
    names = {sw for p in REGISTRY for sw in p.switches} | {"RDTURBO_OVERLAP_START"}
    return [f"{n}={environ[n].strip()}" for n in sorted(names - _MODIFIERS) if n in environ and _active(n, environ[n].strip())]


def switch_problems(root: str, environ=None) -> list:
    """Problems that make a run on `root` not what its switches say: tree not fully applied, files edited after apply,
    a switch value the patched code would not read as intended, or a switch set whose patch is not in the tree."""
    environ = os.environ if environ is None else environ
    st = status(root)
    out = value_problems(environ)
    if not st.get("patched"):
        return out + ["tree is not patched"]
    if st.get("status") == "state-missing":
        return out + [f"the tree carries injected code but {STATE_DIR}/ is missing; {st.get('hint')}"]
    if st.get("status") != "applied":
        out.append(f"manifest status is {st.get('status')!r}, not 'applied' (an apply was interrupted: revert and apply again)")
    if st.get("modified_since_apply"):
        out.append(f"files changed since apply: {st['modified_since_apply']}")
    have = set(st.get("patches") or [])
    missing = []
    for p in REGISTRY:
        if p.name in have:
            continue
        for sw in p.switches:
            if sw not in _MODIFIERS and _active(sw, environ.get(sw, "").strip()):
                out.append(f"{sw}={environ[sw]} is set but patch {p.name!r} is not applied to this tree")
                missing.append(p.name)
    if missing:
        out.append(f"fix: revert and apply with a profile that includes {sorted(set(missing))} (the speedup preset needs "
                   f"--profile all), or unset those switches")
    return out


def cmd_check_env(a) -> int:
    probs = switch_problems(a.root)
    act = active_switches()
    print("active switches: " + (" ".join(act) if act else "none (upstream behaviour)"))
    for p in probs:
        print("  " + p)
    print("switches and tree agree" if not probs else f"{len(probs)} problems")
    return 0 if not probs else 1


def cmd_env(a) -> int:
    path = os.path.join(PRESETS, f"{a.preset}.env")
    if not os.path.exists(path):
        print(f"no preset {a.preset!r}; available: {', '.join(_presets())}", file=sys.stderr)
        return 2
    with open(path) as fh:
        sys.stdout.write(fh.read())
    return 0


def main(argv=None) -> int:
    via_m = os.path.basename(sys.argv[0]) == "__main__.py"      # `python -m robodojo_turbo` vs the console script
    prog = "python -m robodojo_turbo" if via_m else "robodojo-turbo"
    ap = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"robodojo-turbo {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("apply")
    p.add_argument("--root", required=True)
    p.add_argument("--profile", default="all", choices=sorted(PROFILES))
    p.add_argument("--only", default="", help="comma-separated patch names (dependencies are added)")
    p.add_argument("--exclude", default="", help="comma-separated patch names to leave out")
    p.add_argument("--force-unpinned", action="store_true", help="patch even if upstream files differ from the pinned hashes")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_apply)
    for name, fn in (("revert", cmd_revert), ("status", cmd_status), ("pins", cmd_pins), ("check-env", cmd_check_env)):
        q = sub.add_parser(name)
        q.add_argument("--root", required=True)
        if name == "revert":
            q.add_argument("--force", action="store_true", help="revert even if patched files were edited after apply")
        q.set_defaults(fn=fn)
    q = sub.add_parser("env")
    q.add_argument("--preset", default="speedup")
    q.set_defaults(fn=cmd_env)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except PatchError as e:
        print(f"robodojo-turbo: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"robodojo-turbo: {e} (state directory: {os.path.join(os.path.abspath(getattr(a, 'root', '.')), STATE_DIR)})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
