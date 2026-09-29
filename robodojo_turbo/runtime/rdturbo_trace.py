"""Lightweight span tracer for evaluation "bubble" accounting (installed into <RoboDojo>/utils/rdturbo_trace.py).

No third-party dependencies. Enabled only when RDTURBO_TRACE=<path.jsonl> is set; RDTURBO_WORKER names the worker.
Record format (one JSON object per line):
  span : {"t0", "t1", "dt", "kind", "id", "parent", "depth", "w", ...meta}
  event: {"t", "kind", "ev": 1, "w", ...meta}
Client-side kinds: reset / episode / sim_step / obs / render / reward / ws(fn=update_obs_batch|get_action_batch|reset) / video_save
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, List, Optional


def parse_id_spec(spec: str) -> List[int]:
    """'0-12,20,22-23' -> [0..12, 20, 22, 23] (deduplicated, ascending)."""
    ids: set = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            ids.update(range(a, b + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


class Tracer:
    def __init__(self, path: str, worker: Optional[str] = None):
        self.path = path
        self.worker = worker or (os.environ.get("RDTURBO_WORKER") or "w0").strip()
        self._lock = threading.Lock()
        self._local = threading.local()
        self._seq = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a", buffering=1)

    def _stack(self) -> list:
        st = getattr(self._local, "stack", None)
        if st is None:
            st = self._local.stack = []
        return st

    def write(self, rec: dict) -> None:
        rec.setdefault("w", self.worker)
        line = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
        with self._lock:
            self._fh.write(line)

    def event(self, kind: str, **meta: Any) -> None:
        self.write({"t": time.time(), "kind": kind, "ev": 1, **meta})

    @contextmanager
    def span(self, kind: str, **meta: Any):
        st = self._stack()
        with self._lock:
            self._seq += 1
            sid = self._seq
        parent = st[-1] if st else None
        st.append(sid)
        t0 = time.time()
        try:
            yield
        finally:
            t1 = time.time()
            st.pop()
            self.write({"t0": t0, "t1": t1, "dt": t1 - t0, "kind": kind, "id": sid, "parent": parent, "depth": len(st), **meta})

    def wrap(self, fn: Callable, kind: str, meta_fn: Optional[Callable[..., dict]] = None) -> Callable:
        @functools.wraps(fn)
        def inner(*a: Any, **k: Any) -> Any:
            meta: dict = {}
            if meta_fn is not None:
                try:
                    meta = meta_fn(*a, **k) or {}
                except Exception:  # metadata is decoration only; never break the evaluation
                    meta = {}
            with self.span(kind, **meta):
                return fn(*a, **k)

        return inner


_TRACER: Optional[Tracer] = None


def get_tracer() -> Optional[Tracer]:
    """Process-wide tracer created lazily from RDTURBO_TRACE; None when unset."""
    global _TRACER
    if _TRACER is None:
        path = os.environ.get("RDTURBO_TRACE")
        if path:
            _TRACER = Tracer(path)
    return _TRACER


def _obs_bytes(obs_list: Any) -> int:
    """Rough size of the images in a batch of observations (decoded RGB, i.e. what msgpack ships)."""
    total = 0
    try:
        for obs in obs_list or []:
            for cam in (obs.get("vision") or {}).values():
                color = cam.get("color") if isinstance(cam, dict) else None
                nb = getattr(color, "nbytes", None)
                if nb is None and isinstance(color, (bytes, bytearray)):
                    nb = len(color)
                total += int(nb or 0)
    except Exception:
        pass
    return total


def install_env_tracing(env: Any, path: Optional[str] = None) -> Optional[Tracer]:
    """Wrap instance methods of an EvalEnv (instance attributes shadow class methods, so internal self.x() calls are traced too)."""
    tr = Tracer(path) if path else get_tracer()
    if tr is None:
        return None
    env._rdturbo_tracer = tr

    def m_obs(env_idx_list=None, last_frame=False, **_):
        n = len(env_idx_list) if env_idx_list is not None else int(getattr(env, "num_envs", 0))
        return {"n": n, "last": bool(last_frame)}

    def m_step(actions_list, env_idx_list=None, **_):
        return {"n": len(actions_list)}

    def m_reset(seed=None, **_):
        return {"n": len([s for s in (seed or []) if s is not None])}

    env.get_obs_batch = tr.wrap(env.get_obs_batch, "obs", m_obs)
    if hasattr(env, "render"):
        env.render = tr.wrap(env.render, "render")  # nested inside obs / reset; the report splits exclusive time
    env.take_action_batch = tr.wrap(env.take_action_batch, "sim_step", m_step)
    env.is_episode_end = tr.wrap(env.is_episode_end, "reward")
    env.reset = tr.wrap(env.reset, "reset", m_reset)
    env.run_eval = tr.wrap(env.run_eval, "episode")
    if hasattr(env, "save_video"):
        env.save_video = tr.wrap(env.save_video, "video_save")

    mc = getattr(env, "model_client", None)
    if mc is not None and hasattr(mc, "call"):
        def m_call(func_name=None, obs=None, **_):
            m = {"fn": func_name}
            if isinstance(obs, (list, tuple)):
                m["n"] = len(obs)
                if func_name == "update_obs_batch":
                    m["bytes"] = _obs_bytes(obs)
            return m

        mc.call = tr.wrap(mc.call, "ws", m_call)

    tr.event("install", num_envs=int(getattr(env, "num_envs", 0)), pid=os.getpid(),
             layout_ids=(os.environ.get("RDTURBO_LAYOUT_IDS") or "").strip(),
             obs_every_step=(os.environ.get("RDTURBO_OBS_EVERY_STEP") or "1").strip(),
             video_every=(os.environ.get("RDTURBO_VIDEO_EVERY") or "1").strip(),
             switches={k: v for k, v in os.environ.items() if k.startswith("RDTURBO_")})
    return tr
