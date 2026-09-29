"""Revoke Python USD-change listeners that a headless evaluation never reads (switch RDTURBO_REVOKE_LISTENERS).

Background: RoboDojo launches Isaac Lab with the full GUI experience file (isaaclab.python.kit) even for
--headless --enable_cameras, so UI extensions are loaded. With use_fabric=False, omni.physx writes every rigid-body pose
back to USD on every physics substep and each write fans out as a Usd.Notice.ObjectsChanged. Five Python listeners
owned by UI extensions consume those notices for UI bookkeeping only; in a py-spy profile of a 10-env stack_blocks
evaluation they took 14.2% of main-thread wall time:
  omni.usd PrimCaching._on_usd_changed (material-library list cache) 8.5%, omni.kit.scripting ScriptManager 2.0%,
  omni.kit.widget.viewport ViewportWidget 1.9%, omni.kit.widget.stage StageModel 1.0%, omni.usd UsdWatcher 0.8%.
All five callbacks only read USD and update Python-side sets/flags whose readers are windows/panels. The assets contain
no OmniScriptingAPI prims, cameras use replicator render products (no viewport), so revoking them changes neither
physics, pixels nor scores. Prior art: IsaacSim issue #819 profiles the same listener costs; OmniGibson revokes the
viewport-menubar UsdWatcher in-process for teardown correctness.

Mechanism (both steps are needed):
  1. Replace pxr.Tf.Notice.Register with a filter: when the callback belongs to a deny-listed module and the notice is
     ObjectsChanged, return a dummy listener. Needed because every batch closes/re-opens the stage and extensions
     re-register on StageEventType.OPENED.
  2. Sweep existing holders with gc (the listeners registered on the startup stage exist before we can import pxr),
     Revoke() them and set the attribute to None -- extension cleanup code is always `if x: x.Revoke()`.
Never touched: any listener of omni.physx*, omni.replicator*, omni.syntheticdata, isaacsim.*, isaaclab*; StageModel's
LayerInfoDidChange; omni.usd.get_watcher() is not called (it would create and register a watcher).

Usage (patched into main.py right after AppLauncher):
    from utils.rdturbo_revoke_listeners import install; install(os.environ["RDTURBO_REVOKE_LISTENERS"])
RDTURBO_REVOKE_LISTENERS=1|all revokes all five; a comma-separated list of module or holder class names revokes a subset.
Logs the swept count at install; the intercepted count is logged at process exit only where Python's atexit handlers
run (Kit's fast shutdown usually skips them).
"""
import atexit
import collections
import gc

# module -> (callback qualname prefix, holder class name, attribute holding the listener)
DENY = {
    "omni.usd._impl.utils": ("PrimCaching.", "PrimCaching", "_notice_listener"),
    "omni.usd._impl.watcher": ("UsdWatcher.", "UsdWatcher", "_objects_changed"),
    "omni.kit.widget.viewport.widget": ("ViewportWidget.", "ViewportWidget", "_ViewportWidget__stage_listener"),
    "omni.kit.widget.stage.stage_model": ("StageModel._on_objects_changed", "StageModel", "_StageModel__stage_listener"),
    "omni.kit.scripting.scripts.script_manager": ("ScriptManager.", "ScriptManager", "_usd_listener"),
}


class DummyListener:
    """Stand-in for a revoked listener: extension code only does `if x: x.Revoke()`."""

    def Revoke(self):
        pass

    def __bool__(self):
        return True

    def __repr__(self):
        return "<robodojo-turbo DummyListener>"


_state = {"installed": False, "hits": collections.Counter(), "swept": collections.Counter(), "orig": None, "deny": {}}


def _owner(cb):
    f = getattr(cb, "__func__", cb)
    return (getattr(f, "__module__", "") or ""), (getattr(f, "__qualname__", "") or "")


def _is_tf_listener(v):
    t = type(v)
    return t.__module__ == "pxr.Tf" and t.__name__ == "Listener"


def select(selection):
    """'1'/'all' = every entry; otherwise a comma-separated list of module names or holder class names."""
    sel = str(selection).strip()
    if sel.lower() in ("1", "all", "true", "yes", "on"):
        return dict(DENY)
    want = {x.strip() for x in sel.split(",") if x.strip()}
    return {k: v for k, v in DENY.items() if k in want or v[1] in want}


def install(selection="1", log=print):
    from pxr import Tf, Usd  # importable only after Kit has started

    if _state["installed"]:
        return _state
    deny = select(selection)
    if not deny:   # a selection that names nothing known would patch Register and revoke nothing
        log(f"[robodojo-turbo revoke-listeners] {selection!r} selects none of {sorted(DENY)}; nothing revoked")
        return _state
    orig = Tf.Notice.Register

    def _register(notice_type, callback, sender):
        mod, qn = _owner(callback)
        rule = deny.get(mod)
        if rule is not None and notice_type is Usd.Notice.ObjectsChanged and qn.startswith(rule[0]):
            _state["hits"][mod] += 1
            return DummyListener()
        return orig(notice_type, callback, sender)

    Tf.Notice.Register = staticmethod(_register)  # subclasses (Usd.Notice.ObjectsChanged, ...) resolve it through the MRO
    for o in gc.get_objects():
        try:
            t = type(o)
            rule = deny.get(getattr(t, "__module__", "") or "")
            if rule is None or t.__name__ != rule[1]:
                continue
            v = getattr(o, rule[2], None)
            if v is not None and _is_tf_listener(v):
                v.Revoke()
                setattr(o, rule[2], None)
                _state["swept"][t.__module__] += 1
        except Exception as e:  # best effort; must never break the evaluation
            log(f"[robodojo-turbo revoke-listeners] sweep error on {type(o).__name__}: {e}")
    _state.update(installed=True, orig=orig, deny=deny)
    log(f"[robodojo-turbo revoke-listeners] swept {dict(_state['swept'])}; deny={sorted(deny)}")
    atexit.register(report, log)
    return _state


def report(log=print):
    log(f"[robodojo-turbo revoke-listeners] intercepted {dict(_state['hits'])}; swept {dict(_state['swept'])}")
