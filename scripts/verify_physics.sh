#!/usr/bin/env bash
# Physics-invariance check for the speedup switches on one task (single worker, one batch by default):
#   rec   closed loop, every switch off, records the policy's action chunks (RDTURBO_RECORD_ACTIONS)
#   ref   open-loop replay of those actions, every switch off                 -> state/particle traces
#   ref2  the same replay again (noise floor; matters for fluid/garment scenes, whose physics is not bit-deterministic)
#   fast  open-loop replay with robodojo_turbo/presets/speedup.env             -> state/particle traces
# Verdict (exit status):
#   rigid/articulated scenes: ref, ref2 and fast must be bit-identical on every key (tools.state_compare);
#   fluid/garment scenes (a non-empty particle trace): tools.particle_compare decides (nothing frozen that moves in ref);
#     the state comparison is printed for information only, because ref and ref2 already differ there.
# Closed-loop runs are never bit-reproducible (RTX render noise changes policy decisions), which is why actions are replayed.
# Open-loop equality shows the switches leave physics alone; it does not replace a paired closed-loop score comparison.
#
# Usage: OMNI_KIT_ACCEPT_EULA=YES bash scripts/verify_physics.sh --rd <RoboDojo> --task stack_blocks [--seed 0] \
#            [--layout-ids 0-9] [--prefix vp_<timestamp>] [--extra-preset verify] [--conda-env RoboDojo] [launcher options]
# Results: ${RDTURBO_RUNS:-./runs}/<prefix>_{rec,ref,ref2,fast}; the tree must be patched with --profile all (the fast leg
# uses the speedup preset, which includes the ops switches). Replay legs run with RDTURBO_REPLAY_STRICT=1: a missing recorded
# chunk stops the leg instead of asking the policy server.
set -uo pipefail
case " $* " in *" -h "*|*" --help "*) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0;; esac
RD="${RDTURBO_ROBODOJO_ROOT:-}"; TASK=stack_blocks; SEED=0; IDS=0-9; PREFIX="vp_$(date +%Y%m%d%H%M%S)"; EXTRA=""; ARGS=()
CONDA_ENV="${RDTURBO_CONDA_ENV-RoboDojo}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rd) RD=$2; shift 2;; --task) TASK=$2; shift 2;; --seed) SEED=$2; shift 2;; --layout-ids) IDS=$2; shift 2;;
    --prefix) PREFIX=$2; shift 2;; --extra-preset) EXTRA=$2; shift 2;; --conda-env) CONDA_ENV=$2; shift 2;;
    *) ARGS+=("$1"); shift;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd); PKG=$(cd "$HERE/.." && pwd); PRE="$PKG/robodojo_turbo/presets"
RUNS=$(realpath -m "${RDTURBO_RUNS:-$PWD/runs}")
if [[ -n "$CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || { echo "cannot activate conda env $CONDA_ENV"; exit 2; }
fi
export PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}"
compgen -G "$RUNS/${PREFIX}_*" >/dev/null && { echo "$RUNS/${PREFIX}_* exists; choose another --prefix"; exit 2; }
common=(--rd "$RD" --task "$TASK" --seed "$SEED" --layout-ids "$IDS" --workers 1 --conda-env "$CONDA_ENV" "${ARGS[@]}")
# check both switch sets against the tree before any GPU leg runs
for set_ in "off" "speedup${EXTRA:+ $EXTRA}"; do
  ( set -a; for pr in $set_; do source "$PRE/$pr.env"; done; set +a
    python -m robodojo_turbo check-env --root "$RD" >/dev/null ) || {
    echo "the tree cannot run the '$set_' legs:"; ( set -a; for pr in $set_; do source "$PRE/$pr.env"; done; set +a; python -m robodojo_turbo check-env --root "$RD" ); exit 2; }
done
run(){ local tag=$1; shift; env "$@" bash "$HERE/rdturbo_eval.sh" --tag "$tag" --out "$RUNS/$tag" "${common[@]}"; }

set -a; source "$PRE/off.env"; set +a
run "${PREFIX}_rec" RDTURBO_RECORD_ACTIONS=1 RDTURBO_STATE_TRACE=1 RDTURBO_PARTICLE_TRACE=1 || { echo "record run failed"; exit 1; }
ACT="$RUNS/${PREFIX}_rec/actions_w0.pkl"
[[ -s "$ACT" ]] || { echo "no recorded actions at $ACT"; exit 1; }
for leg in ref ref2; do
  run "${PREFIX}_$leg" RDTURBO_REPLAY_ACTIONS="$ACT" RDTURBO_REPLAY_STRICT=1 RDTURBO_STATE_TRACE=1 RDTURBO_PARTICLE_TRACE=1 || echo "$leg failed"
done
(
  set -a; source "$PRE/speedup.env"; [[ -n "$EXTRA" ]] && source "$PRE/$EXTRA.env"; set +a
  run "${PREFIX}_fast" RDTURBO_REPLAY_ACTIONS="$ACT" RDTURBO_REPLAY_STRICT=1 RDTURBO_STATE_TRACE=1 RDTURBO_PARTICLE_TRACE=1
) || echo "fast replay failed"

R="$RUNS/${PREFIX}"
for leg in ref ref2 fast; do
  [[ -s "${R}_$leg/state_w0.pkl" ]] || { echo "verdict: INVALID (the $leg leg produced no state trace; see ${R}_$leg/meta.txt and client_w0.out)"; exit 2; }
done
PARTICLES=$(python -c "import pickle,sys; print(len(pickle.load(open(sys.argv[1],'rb'))))" "${R}_ref/particles_w0.pkl" 2>/dev/null || echo 0)
echo "==== physics state (robot, rigid objects, articulated objects)"
python -m robodojo_turbo.tools.state_compare "${R}_ref/state_w0.pkl" "${R}_ref2/state_w0.pkl" "${R}_fast/state_w0.pkl"
src=$?
if (( PARTICLES > 0 )); then
  echo "(fluid/garment scene: ref and ref2 are not expected to be bit-identical; the state comparison above is informational)"
  echo "==== fluid particles / cloth points (USD side, what the scorers read)"
  python -m robodojo_turbo.tools.particle_compare "${R}_ref/particles_w0.pkl" "${R}_ref2/particles_w0.pkl" "${R}_fast/particles_w0.pkl"
  rc=$?
else
  rc=$src
fi
echo "==== scores (replay; per layout)"
python -m robodojo_turbo.tools.paired_compare "${R}_ref/merged_result.json" "${R}_fast/merged_result.json" --name-a ref --name-b fast
echo "verdict: $([[ $rc == 0 ]] && echo PASS || echo FAIL) ($( (( PARTICLES > 0 )) && echo 'particle comparison' || echo 'bit-identical state'))"
exit $rc
