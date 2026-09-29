#!/usr/bin/env bash
# Download the nine NVIDIA material files every RoboDojo scene fetches at runtime (Geometry/camera_stand/00000/object.usd
# references three MDLs on NVIDIA's S3 bucket; six default textures come with them) into a local mirror, verify their
# sha256, and write MANIFEST.json for RDTURBO_OFFLINE_ASSETS=1. Run it on the machine that runs Isaac Sim, once, with
# internet access. The files are NVIDIA materials obtained from NVIDIA under NVIDIA's terms: do not redistribute them.
# Usage: bash scripts/mirror_nv_assets.sh [mirror_dir]   (default: ${RDTURBO_NV_MIRROR:-~/.cache/robodojo_turbo/nv_mirror})
set -euo pipefail
case "${1:-}" in -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; esac
M=${1:-${RDTURBO_NV_MIRROR:-$HOME/.cache/robodojo_turbo/nv_mirror}}
[[ "$M" == "~" || "$M" == "~/"* ]] && M="$HOME${M:1}"   # a quoted ~ is not expanded by the shell
mkdir -p "$M" && M=$(cd "$M" && pwd)                   # absolute: the evaluation client runs from the RoboDojo root
H=omniverse-content-production.s3.us-west-2.amazonaws.com
BASE=https://$H/Materials/2023_1/Base
# path relative to Materials/2023_1/Base, expected sha256 (bytes fetched from the bucket on 2026-09-24)
LIST="Plastics/Plastic_ABS.mdl 996caa235e62010e15fcf8c820ffdf502a3c276fc416d3f95298e325adf44616
Metals/Aluminum_Cast.mdl 78c3edb2398c7a48618ebe011a9a9d09802e56e1af041ea7797b115f934bce53
Metals/Aluminum_Anodized.mdl 5469bfef01fbd8c4b1784ed90884a29d93eb68ce6f27a8ae61803411ea3fb713
Metals/Aluminum_Cast/Aluminum_Cast_BaseColor.png 03ab7494d37e1f6551a26894628c5fe03a65b4d85a09ed11337e184852a580b3
Metals/Aluminum_Cast/Aluminum_Cast_ORM.png 571eba5a99da444b681609a722a29914490c6dae45d230dac471a45875211a4c
Metals/Aluminum_Cast/Aluminum_Cast_N.png 812c0161ea14abc581cb35c47b33dc6005c089f8e82dca6cb4dd4832deb0556c
Metals/Aluminum_Anodized/Aluminum_Anodized_BaseColor.png d1d042502d7d94bca13cee10c63ab5b3801fb0a46e26d79169e13f5b9c7b5a31
Metals/Aluminum_Anodized/Aluminum_Anodized_ORM.png 768f2dbb4f702a9624b912b431efd1a6a8e0ff3e93744cf54f3866ef8f7986e9
Metals/Aluminum_Anodized/Aluminum_Anodized_Normal.png 6dc1cb1b23a9abd766188a85ccbad1a2639d0a9a334f284e359c6c5d4438608e"
mkdir -p "$M"
ENTRIES=""
while read -r rel sha; do
  dst="$M/$H/Materials/2023_1/Base/$rel"
  mkdir -p "$(dirname "$dst")"
  if [[ ! -f "$dst" || "$(sha256sum "$dst" | cut -d' ' -f1)" != "$sha" ]]; then
    curl -fsSR --max-time 120 -o "$dst.part" "$BASE/$rel"   # -R: local mtime = remote Last-Modified (RTX texture cache key)
    mv "$dst.part" "$dst"
  fi
  got=$(sha256sum "$dst" | cut -d' ' -f1)
  [[ "$got" == "$sha" ]] || { echo "sha256 mismatch for $rel: $got" >&2; exit 1; }
  ENTRIES="$ENTRIES{\"url\":\"$BASE/$rel\",\"local\":\"$H/Materials/2023_1/Base/$rel\",\"bytes\":$(stat -c %s "$dst"),\"sha256\":\"$sha\",\"mtime\":$(stat -c %Y "$dst")},"
done <<<"$LIST"
printf '{"aliases":{"https://%s/":"%s/"},"files":[%s]}\n' "$H" "$H" "${ENTRIES%,}" > "$M/MANIFEST.json"
echo "mirror ok: $M (9 files); run evaluations with RDTURBO_OFFLINE_ASSETS=1 RDTURBO_NV_MIRROR=$M"
