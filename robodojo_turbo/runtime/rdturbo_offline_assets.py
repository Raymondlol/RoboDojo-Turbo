"""Offline NVIDIA assets (switch RDTURBO_OFFLINE_ASSETS): the evaluation stops depending on NVIDIA's S3 bucket.

Background: every RoboDojo scene loads Geometry/camera_stand/00000/object.usd, which references three MDL materials
(Plastic_ABS, Aluminum_Cast, Aluminum_Anodized) and their six default textures on
https://omniverse-content-production.s3.us-west-2.amazonaws.com/. omni.client fetches them at the first reset of every
process, even when ~/.cache/ov/client holds a copy. Without internet, Kit waits out omni.client's retries (~120 s),
Kit's hang detector fires and the evaluation dies. A static scan of all assets referenced by the 54 RoboDojo task
layouts (2026-09-27) found no other file on that host.

scripts/mirror_nv_assets.sh downloads the nine files unchanged (same sha256, mtime = remote Last-Modified) into
<mirror>/<host>/... and writes MANIFEST.json. install() verifies every file against the manifest and then points the
whole host prefix at the mirror with omni.client.set_alias. The files are NVIDIA materials: users obtain them from
NVIDIA under NVIDIA's terms; this project never redistributes them.

A mirror that exists but fails verification exits the process (rc=3); a missing mirror prints one line and stays online.
"""
import hashlib
import json
import os
import sys


def install(mirror):
    import omni.client as oc

    mirror = os.path.abspath(os.path.expanduser(mirror))   # the client runs with cwd = RoboDojo root
    man_path = os.path.join(mirror, "MANIFEST.json")
    if not os.path.exists(man_path):
        print(f"[robodojo-turbo offline-assets] no mirror at {man_path} (run scripts/mirror_nv_assets.sh); staying online", flush=True)
        return
    with open(man_path) as fh:
        man = json.load(fh)
    bad = []
    for e in man["files"]:
        f = os.path.join(mirror, e["local"])
        if not os.path.isfile(f) or os.path.getsize(f) != e["bytes"] or int(os.path.getmtime(f)) != int(e["mtime"]):
            bad.append(e["url"])
            continue
        with open(f, "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != e["sha256"]:
                bad.append(e["url"])
    if bad:
        print(f"[robodojo-turbo offline-assets] MIRROR BAD {bad}", flush=True)
        sys.exit(3)
    for src, rel in man["aliases"].items():
        dst = rel if rel.startswith("file:") else "file:" + os.path.join(mirror, rel).rstrip("/") + "/"
        oc.set_alias(src, dst)
    probe = man["files"][0]["url"]
    res, _ = oc.stat(probe)
    norm = oc.normalize_url(probe)
    print(f"[robodojo-turbo offline-assets] {len(man['files'])} files verified; aliases={man['aliases']}; probe stat={res} -> {norm}", flush=True)
    if res != oc.Result.OK or not norm.startswith("file:"):
        print("[robodojo-turbo offline-assets] ALIAS NOT EFFECTIVE", flush=True)
        sys.exit(3)
