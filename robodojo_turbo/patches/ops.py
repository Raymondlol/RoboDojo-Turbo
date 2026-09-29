"""Operational patches (profile "ops"): they do not touch physics or pixels but change what the process loads or where it
fetches assets from. Off unless their RDTURBO_* variable is set.
"""
from __future__ import annotations

import os
import re

from .. import MARK
from ..engine import PatchError, Tree, find_once, sub_once
from .harness import MAIN, ROBOT_MANAGER, RUNTIME
from .speedup import APP_ANCHOR

PI05_SERVER_SH = "XPolicyLab/policy/Pi_05/setup_eval_policy_server.sh"

# ---------------------------------------------------------------------------------------------------------------------
# no_planner: do not build the cuRobo planner/IK solver (only end-effector actions use it)

_NP_STUB = '''class _RdturboNoPlanner:  # [robodojo-turbo] no-planner: stand-in for the cuRobo planner; joint-action evaluation never calls it
    def __init__(self, name):
        self._rdturbo_name = name

    def __getattr__(self, a):
        if a.startswith("__"):
            raise AttributeError(a)
        # Fail hard: RoboDojo's main loop catches generic exceptions and would silently skip the batch.
        print(f"[robodojo-turbo] RDTURBO_NO_PLANNER=1 but the planner {self._rdturbo_name}.{a} was used "
              "(end-effector actions need the planner); exiting with status 3", flush=True)
        __import__("os")._exit(3)


'''


def patch_no_planner(tree: Tree) -> str:
    s = tree.read(ROBOT_MANAGER)
    m = find_once(s, r"^    def _setup_planner\(self, robot\):[ \t]*\n", "robot_manager _setup_planner")
    ins = (f"        if (__import__('os').environ.get('RDTURBO_NO_PLANNER') or '0').strip() == '1':  # {MARK} no-planner\n"
           "            if robot.robot_type == 'arm':\n"
           "                self.planner[robot.robot_name] = self.ik_solver[robot.robot_name] = _RdturboNoPlanner(robot.robot_name)\n"
           "            return\n")
    s = s[: m.end()] + ins + s[m.end():]
    c = find_once(s, r"^class RobotManager\b", "robot_manager class RobotManager")
    s = s[: c.start()] + _NP_STUB + s[c.start():]
    tree.write(ROBOT_MANAGER, s)
    return "RDTURBO_NO_PLANNER=1 (joint actions only; any planner use exits with status 3)"


# ---------------------------------------------------------------------------------------------------------------------
# offline_assets

def patch_offline_assets(tree: Tree) -> str:
    tree.install(os.path.join(RUNTIME, "rdturbo_offline_assets.py"), "utils/rdturbo_offline_assets.py")
    s = tree.read(MAIN)
    s = sub_once(s, APP_ANCHOR,
                 lambda m: (f"{m.group(0)}\n{m.group(1)}if (os.environ.get('RDTURBO_OFFLINE_ASSETS') or '0').strip() not in ('', '0'):  # {MARK} offline-assets\n"
                            f"{m.group(1)}    from utils.rdturbo_offline_assets import install as _rdturbo_offline_install\n"
                            f"{m.group(1)}    _rdturbo_offline_install((os.environ.get('RDTURBO_NV_MIRROR') or '~/.cache/robodojo_turbo/nv_mirror').strip())"),
                 "main.py AppLauncher (offline-assets)")
    tree.write(MAIN, s)
    return "RDTURBO_OFFLINE_ASSETS=1, RDTURBO_NV_MIRROR=<dir> (build it with scripts/mirror_nv_assets.sh)"


# ---------------------------------------------------------------------------------------------------------------------
# server_memfrac: upstream hard-codes XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 of the whole card for the Pi_05 server

def patch_server_memfrac(tree: Tree) -> str:
    s = tree.read(PI05_SERVER_SH)
    s = sub_once(s, r"^export XLA_PYTHON_CLIENT_MEM_FRACTION=([0-9.]+)[ \t]*$",
                 r"export XLA_PYTHON_CLIENT_MEM_FRACTION=${RDTURBO_SERVER_MEM_FRACTION:-\g<1>}  # " + MARK + " memfrac",
                 "Pi_05 setup_eval_policy_server.sh MEM_FRACTION")
    tree.write(PI05_SERVER_SH, s)
    return "RDTURBO_SERVER_MEM_FRACTION=<f> overrides the Pi_05 server's XLA_PYTHON_CLIENT_MEM_FRACTION (unset = upstream 0.3)"
