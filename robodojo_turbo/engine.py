"""Patch engine: select patches, check upstream pins and APIs, back up, apply, record, and revert byte-exactly.

A RoboDojo checkout is patched in place. Before any patch runs, every upstream file that the selected patches may
rewrite is copied to <root>/.robodojo_turbo/backup/ and a manifest is written. `revert` restores those copies, deletes
every installed runtime module and removes the state directory, then verifies that each restored file matches the
hash recorded before patching.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import re
import shutil
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import MARK, __version__

STATE_DIR = ".robodojo_turbo"
MANIFEST = "manifest.json"
PINS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pins.json")


class PatchError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Patch:
    name: str
    profile: str                       # harness | speedup | ops
    targets: Tuple[str, ...]           # upstream files this patch may rewrite (relative to the RoboDojo root)
    fn: Callable[["Tree"], str]
    requires: Tuple[str, ...] = ()
    installs: Tuple[str, ...] = ()     # files this patch creates inside the tree
    switches: Tuple[str, ...] = ()     # runtime environment variables that turn it on
    api: Tuple[Tuple[str, str], ...] = ()   # (relpath, regex) the patched code relies on at runtime
    doc: str = ""


class Tree:
    """File access relative to a RoboDojo root; patches only read/write through this object."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.installed: List[str] = []

    def path(self, rel: str) -> str:
        return os.path.join(self.root, rel)

    def exists(self, rel: str) -> bool:
        return os.path.exists(self.path(rel))

    def read(self, rel: str) -> str:
        with open(self.path(rel), encoding="utf-8") as fh:
            return fh.read()

    def write(self, rel: str, text: str) -> None:
        p = self.path(rel)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)

    def install(self, src: str, rel: str) -> None:
        dst = self.path(rel)
        if os.path.exists(dst):
            raise PatchError(f"{rel} already exists in the tree; refusing to overwrite a file we did not create")
        shutil.copy2(src, dst)
        self.installed.append(rel)


# ---------------------------------------------------------------------------------------------------------------------
# text helpers used by the patch modules

def sub_once(text: str, pattern: str, repl, what: str, flags: int = re.M) -> str:
    out, n = re.subn(pattern, repl, text, flags=flags)
    if n != 1:
        raise PatchError(f"{what}: anchor matched {n} times (expected exactly 1)")
    return out


def find_once(text: str, pattern: str, what: str, flags: int = re.M) -> "re.Match":
    hits = list(re.finditer(pattern, text, flags))
    if len(hits) != 1:
        raise PatchError(f"{what}: anchor matched {len(hits)} times (expected exactly 1)")
    return hits[0]


def indent_block(block: str, ind: str) -> str:
    """Indent every non-empty line of a dedented block by `ind`; keeps a trailing newline."""
    return "".join((ind + line if line.strip() else line) + "\n" for line in block.strip("\n").split("\n"))


# ---------------------------------------------------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pins(path: str = PINS_FILE) -> dict:
    with open(path) as fh:
        return json.load(fh)


def pinned_commits() -> str:
    up = load_pins().get("upstream", {})
    xp = up.get("XPolicyLab")
    xp = " or ".join(c[:7] for c in xp) if isinstance(xp, list) else str(xp)[:7]
    return f"RoboDojo {str(up.get('RoboDojo'))[:7]} with XPolicyLab {xp}"


def check_root(root: str) -> None:
    if not os.path.isdir(os.path.join(root, "src", "eval_client")):
        raise PatchError(f"{os.path.abspath(root)} is not a RoboDojo checkout (no src/eval_client)")


def marked_files(root: str) -> List[str]:
    """Upstream files that carry an injected-code marker, and our runtime modules: finds a patched tree whose state
    directory was deleted (git clean -fd, git stash -u, a copy without dotfiles)."""
    tree = Tree(root)
    out = [rel for rel in sorted(load_pins()["files"]) if tree.exists(rel) and MARK in tree.read(rel)]
    utils = tree.path("utils")
    if os.path.isdir(utils):
        out += [os.path.join("utils", f) for f in sorted(os.listdir(utils)) if f.startswith("rdturbo_") and f.endswith(".py")]
    return out


def precheck(root: str) -> List[str]:
    """Why `root` cannot be patched as it is (empty = it can): patched already, with or without a state directory.
    Shared by apply and apply --dry-run."""
    root_abs = os.path.abspath(root)
    if load_manifest(root) is not None:
        return [f"{root_abs} is already patched by RoboDojo-Turbo; run `python -m robodojo_turbo revert --root {root_abs}` first"]
    marked = marked_files(root)
    if marked:
        return [f"{root_abs} already carries injected code but has no {STATE_DIR}/ manifest (deleted by git clean, "
                f"git stash -u or a copy without dotfiles?); " + restore_hint(root, marked)]
    if os.path.exists(os.path.join(root, STATE_DIR)):
        return [f"{os.path.join(root_abs, STATE_DIR)} exists without a manifest (an apply was killed before it changed any "
                "file); remove that directory and apply again"]
    return []


def restore_hint(root: str, files: Sequence[str]) -> str:
    root = os.path.abspath(root)
    own = [f for f in files if os.path.basename(f).startswith("rdturbo_")]   # modules this tool installed: delete
    xp = [f for f in files if f.startswith("XPolicyLab/") and f not in own]
    rd = [f for f in files if not f.startswith("XPolicyLab/") and f not in own]
    cmds = []
    if rd:
        cmds.append(f"git -C {root} checkout -- " + " ".join(rd))
    if xp:
        cmds.append(f"git -C {root}/XPolicyLab checkout -- " + " ".join(f[len('XPolicyLab/'):] for f in xp))
    if own:
        cmds.append("rm " + " ".join(os.path.join(root, f) for f in own))
    return "restore the upstream files with:\n  " + "\n  ".join(cmds)


def resolve(registry: Sequence[Patch], names: Iterable[str]) -> List[Patch]:
    """Close `names` over `requires` and return the patches in registry (application) order."""
    by_name = {p.name: p for p in registry}
    want, stack = set(), list(names)
    while stack:
        n = stack.pop()
        if n not in by_name:
            raise PatchError(f"unknown patch {n!r}; known: {', '.join(by_name)}")
        if n in want:
            continue
        want.add(n)
        stack.extend(by_name[n].requires)
    return [p for p in registry if p.name in want]


def select(registry: Sequence[Patch], profile: Optional[str], only: Sequence[str] = (), exclude: Sequence[str] = ()) -> List[Patch]:
    from .patches import PROFILES
    if only:
        names = list(only)
    else:
        if profile not in PROFILES:
            raise PatchError(f"unknown profile {profile!r}; known: {', '.join(PROFILES)}")
        names = [p.name for p in registry if p.profile in PROFILES[profile]]
    unknown = sorted(set(exclude) - {p.name for p in registry})
    if unknown:
        raise PatchError(f"unknown patch names in --exclude: {unknown}; known: {', '.join(p.name for p in registry)}")
    names = [n for n in names if n not in set(exclude)]
    chosen = resolve(registry, names)
    dropped = [n for n in exclude if n in {p.name for p in chosen}]
    if dropped:
        need = {n: [p.name for p in chosen if n in p.requires] for n in dropped}
        raise PatchError("cannot exclude " + "; ".join(f"{n} (required by {', '.join(v)})" for n, v in need.items())
                         + ": exclude those too")
    return chosen


def check_pins(tree: Tree, patches: Sequence[Patch], pins: dict) -> List[str]:
    problems = []
    for rel in sorted({t for p in patches for t in p.targets}):
        if not tree.exists(rel):
            problems.append(f"{rel}: missing")
            continue
        want = pins["files"].get(rel)
        got = sha256_file(tree.path(rel))
        if want is None:
            problems.append(f"{rel}: no pin")
        elif got not in (want if isinstance(want, list) else [want]):
            problems.append(f"{rel}: sha256 {got[:12]} not in pinned set")
    return problems


def check_api(tree: Tree, patches: Sequence[Patch]) -> List[str]:
    problems = []
    for p in patches:
        for rel, rx in p.api:
            if not tree.exists(rel) or not re.search(rx, tree.read(rel), re.M):
                problems.append(f"{p.name}: needs /{rx}/ in {rel}")
    return problems


def _state(tree: Tree) -> str:
    return tree.path(STATE_DIR)


def load_manifest(root: str) -> Optional[dict]:
    p = os.path.join(os.path.abspath(root), STATE_DIR, MANIFEST)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _write_manifest(tree: Tree, man: dict) -> None:
    p = os.path.join(_state(tree), MANIFEST)
    with open(p + ".tmp", "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    os.replace(p + ".tmp", p)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compile_check(tree: Tree, rels: Iterable[str]) -> List[str]:
    errors = []
    for rel in rels:
        if rel.endswith(".py") and tree.exists(rel):
            try:
                compile(tree.read(rel), rel, "exec")
            except SyntaxError as e:
                errors.append(f"{rel}:{e.lineno}: {e.msg}")
    return errors


def apply(root: str, patches: Sequence[Patch], force_unpinned: bool = False, log=print) -> dict:
    check_root(root)
    tree = Tree(root)
    blocked = precheck(root)
    if blocked:
        raise PatchError(blocked[0])
    if not patches:
        raise PatchError("no patches selected")
    targets = sorted({t for p in patches for t in p.targets})
    missing = [rel for rel in targets if not tree.exists(rel)]
    if missing:
        raise PatchError(f"target files missing from {root}: {missing} (not a RoboDojo checkout with XPolicyLab?)")
    planned = [rel for p in patches for rel in p.installs]
    taken = [rel for rel in planned if tree.exists(rel)]
    if taken:
        raise PatchError(f"files this tool would create already exist: {taken}; refusing to overwrite files we did not create")
    for rel in targets:
        if tree.exists(rel):
            text = tree.read(rel)
            if MARK in text:
                raise PatchError(f"{rel} already contains {MARK!r} (patched by this tool after its state directory was "
                                 f"deleted); " + restore_hint(root, marked_files(root) or [rel]))
    problems = check_pins(tree, patches, load_pins())
    if problems:
        msg = (f"upstream files do not match the pinned versions ({pinned_commits()}):\n  " + "\n  ".join(problems))
        if not force_unpinned:
            raise PatchError(msg + "\n(use --force-unpinned to try anyway; physics/scoring claims then do not apply)")
        log("WARNING: " + msg)
    api = check_api(tree, patches)
    if api:
        raise PatchError("upstream APIs the patched code calls are missing:\n  " + "\n  ".join(api))

    state = _state(tree)
    os.makedirs(os.path.join(state, "backup"), exist_ok=False)
    backups = {}
    try:   # no upstream file has been touched yet: on any failure here, drop the state directory
        for rel in targets:
            dst = os.path.join(state, "backup", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(tree.path(rel), dst)
            backups[rel] = sha256_file(dst)
        # `installed` lists every file the selected patches may create: none existed (checked above), so anything at
        # those paths after an interrupted apply is ours and revert removes it
        man = {"tool": "robodojo_turbo", "version": __version__, "applied_at": _now(), "status": "applying",
               "patches": [p.name for p in patches], "backups": backups, "installed": planned, "results": {},
               "unpinned": bool(problems)}
        _write_manifest(tree, man)
    except BaseException:
        shutil.rmtree(state, ignore_errors=True)
        raise
    try:
        for p in patches:
            before = len(tree.installed)
            msg = p.fn(tree)
            man["results"][p.name] = msg
            log(f"{p.name}: {msg}" + (f" (+{len(tree.installed) - before} installed)" if len(tree.installed) > before else ""))
        if sorted(tree.installed) != sorted(planned):
            raise PatchError(f"installed files {sorted(tree.installed)} differ from the declared ones {sorted(planned)}")
        errors = _compile_check(tree, targets + tree.installed)
        if errors:
            raise PatchError("patched files do not compile:\n  " + "\n  ".join(errors))
    except BaseException:   # includes KeyboardInterrupt: never leave a half-patched tree behind
        log("patching failed or was interrupted; restoring the upstream files")
        revert(root, log=log, force=True)
        raise
    man["after"] = {rel: sha256_file(tree.path(rel)) for rel in targets + tree.installed}
    man["status"] = "applied"
    _write_manifest(tree, man)
    return man


def dry_run(root: str, patches: Sequence[Patch]) -> List[str]:
    """Run every selected patch on a scratch copy of its target files: pins, API checks, every regex anchor and a compile
    check, without touching the tree. Returns the problems (empty = the patches would apply)."""
    import tempfile
    check_root(root)
    tree = Tree(root)
    problems = [f"tree: {p}" for p in precheck(root)]
    problems += [f"pin: {p}" for p in check_pins(tree, patches, load_pins())] + [f"api: {p}" for p in check_api(tree, patches)]
    targets = sorted({t for p in patches for t in p.targets})
    with tempfile.TemporaryDirectory(prefix="robodojo-turbo-dry-") as tmp:
        for rel in targets + [r for p in patches for r in p.installs]:
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)), exist_ok=True)
        for rel in targets:
            if tree.exists(rel):
                shutil.copy2(tree.path(rel), os.path.join(tmp, rel))
        scratch = Tree(tmp)
        for p in patches:
            try:
                p.fn(scratch)
            except (PatchError, OSError) as e:
                problems.append(f"anchor: {p.name}: {e}")
        problems += [f"compile: {e}" for e in _compile_check(scratch, targets + scratch.installed)]
    return problems


def revert(root: str, log=print, force: bool = False) -> None:
    check_root(root)
    tree = Tree(root)
    man = load_manifest(root)
    if man is None:
        marked = marked_files(root)
        if marked:
            raise PatchError(f"{root} is patched but its state directory {STATE_DIR}/ is missing; " + restore_hint(root, marked))
        raise PatchError(f"{root} has no RoboDojo-Turbo manifest and no injected code; nothing to revert")
    state = _state(tree)
    drift = status(root).get("modified_since_apply") or []
    if drift and not force:
        raise PatchError(f"files changed since apply: {drift}; revert would discard those edits (use --force to revert anyway)")
    # a drifted file without our marker was replaced after apply (e.g. an upstream update): never put the old backup over it
    replaced = [rel for rel in drift if rel in man["backups"] and tree.exists(rel) and MARK not in tree.read(rel)]
    for rel in replaced:
        log(f"not restoring {rel}: it was replaced after apply and no longer contains injected code")
    for rel in man.get("installed", []):
        p = tree.path(rel)
        if os.path.exists(p):
            os.remove(p)
        stem = os.path.splitext(os.path.basename(rel))[0]
        cache = os.path.join(os.path.dirname(p), "__pycache__")
        if os.path.isdir(cache):
            for f in os.listdir(cache):
                if f.startswith(stem + "."):
                    os.remove(os.path.join(cache, f))
    bad = []
    for rel, sha in man["backups"].items():
        if rel in replaced:
            continue
        src = os.path.join(state, "backup", rel)
        shutil.copy2(src, tree.path(rel))
        if sha256_file(tree.path(rel)) != sha:
            bad.append(rel)
    if bad:
        raise PatchError(f"restored files do not match their backups: {bad}; state kept in {state}")
    shutil.rmtree(state)
    log(f"reverted {len(man['backups']) - len(replaced)} files, removed {len(man.get('installed', []))} installed files")


def status(root: str) -> Dict[str, object]:
    check_root(root)
    tree = Tree(root)
    man = load_manifest(root)
    if man is None:
        marked = marked_files(root)
        if marked:
            return {"patched": True, "status": "state-missing", "marked_files": marked, "modified_since_apply": marked,
                    "hint": restore_hint(root, marked)}
        return {"patched": False}
    drift = [rel for rel, sha in man.get("after", {}).items() if not tree.exists(rel) or sha256_file(tree.path(rel)) != sha]
    return {"patched": True, "status": man.get("status"), "version": man.get("version"), "applied_at": man.get("applied_at"),
            "patches": man.get("patches"), "unpinned": man.get("unpinned"), "modified_since_apply": drift}
