#!/usr/bin/env bash
# Same-machine timing benchmark: upstream RoboDojo vs the harness only vs the full speedup preset, in alternating order.
#   upstream  tree reverted to upstream (python -m robodojo_turbo revert), no switches      (--allow-unpatched)
#   harness   tree patched (--profile all), robodojo_turbo/presets/harness.env   (chunk-start upload + fast path)
#   speedup   tree patched (--profile all), robodojo_turbo/presets/speedup.env
# Every leg evaluates the same layouts with the same checkpoint, 1 process x num_envs. Timing = wall_total (policy server
# start -> last client exit); per-batch timing comes from tools.batch_times on the traces (not available for upstream).
# The tree's patch state is checked after every revert/apply (a leg never runs on the wrong tree) and restored on exit.
# Report absolute times with the full conditions (GPU, CPU, driver, Isaac Sim, task, ckpt, layouts, envs, repeats).
#
# Usage: OMNI_KIT_ACCEPT_EULA=YES bash scripts/reproduce_benchmark.sh --rd <RoboDojo> [--task stack_blocks] [--seed 0] \
#            [--layouts 25] [--reps 2] [--legs upstream,harness,speedup] [--prefix bench_<timestamp>] [--conda-env RoboDojo]
#            [launcher options]
# --legs picks which legs run (same alternating order), e.g. --legs upstream,speedup for a two-arm comparison.
# --same-jax-cache gives the upstream and harness legs the JAX compilation cache the speedup preset uses
# (~/.cache/robodojo_turbo/jax), so no arm gains from a warm cache the others lack (about 6 s per run otherwise; the
# first leg of a session still compiles cold).
# Results: ${RDTURBO_RUNS:-./runs}/<prefix>_<leg>_r<rep>
set -uo pipefail
case " $* " in *" -h "*|*" --help "*) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0;; esac
RD="${RDTURBO_ROBODOJO_ROOT:-}"; TASK=stack_blocks; SEED=0; LAYOUTS=25; REPS=2; PREFIX="bench_$(date +%Y%m%d%H%M%S)"; ARGS=()
LEGS=upstream,harness,speedup; SAME_JAX=0
CONDA_ENV="${RDTURBO_CONDA_ENV-RoboDojo}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rd) RD=$2; shift 2;; --task) TASK=$2; shift 2;; --seed) SEED=$2; shift 2;; --layouts) LAYOUTS=$2; shift 2;;
    --reps) REPS=$2; shift 2;; --prefix) PREFIX=$2; shift 2;; --conda-env) CONDA_ENV=$2; shift 2;; --legs) LEGS=$2; shift 2;; --same-jax-cache) SAME_JAX=1; shift;;
    *) ARGS+=("$1"); shift;;
  esac
done
for l in ${LEGS//,/ }; do case $l in upstream|harness|speedup) ;; *) echo "--legs: unknown leg '$l' (upstream, harness, speedup)"; exit 2;; esac; done
HERE=$(cd "$(dirname "$0")" && pwd); PKG=$(cd "$HERE/.." && pwd)
RUNS=$(realpath -m "${RDTURBO_RUNS:-$PWD/runs}")
[[ -n "$RD" && -d "$RD/src/eval_client" ]] || { echo "--rd must point at a RoboDojo checkout"; exit 2; }
if [[ -n "$CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || { echo "cannot activate conda env $CONDA_ENV"; exit 2; }
fi
export PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}"
compgen -G "$RUNS/${PREFIX}_*" >/dev/null && { echo "$RUNS/${PREFIX}_* exists; choose another --prefix"; exit 2; }
RDTURBO(){ python -m robodojo_turbo "$@"; }
patched(){ RDTURBO status --root "$RD" | grep -q '"patched": true'; }
patch_list(){ RDTURBO status --root "$RD" | python -c "import json,sys; print(','.join(json.load(sys.stdin).get('patches') or []))"; }
die(){ echo "reproduce_benchmark: $*" >&2; exit 2; }

# only switch between states this tool manages: upstream, or fully applied by this version with nothing else in the tree
STATE=$(RDTURBO status --root "$RD" | python -c "import json,sys; d=json.load(sys.stdin); print('upstream' if not d.get('patched') else ('hint' if d.get('hint') else d.get('status')))") \
  || die "cannot read the patch status of $RD"
if [[ "$STATE" != upstream && "$STATE" != applied ]]; then
  die "$RD is neither upstream nor cleanly patched by this version; $(RDTURBO status --root "$RD" | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('hint') or 'status: ' + str(d.get('status')))")"
fi
# restore the tree's starting state on every exit path
if patched; then INITIAL=$(patch_list); else INITIAL=""; fi
restore(){
  local ok=1
  if [[ -z "$INITIAL" ]]; then
    if patched; then RDTURBO revert --root "$RD" || ok=0; fi
  elif [[ "$(patched && patch_list)" != "$INITIAL" ]]; then
    if patched; then RDTURBO revert --root "$RD" || ok=0; fi
    [[ $ok == 1 ]] && { RDTURBO apply --root "$RD" --only "$INITIAL" >/dev/null || ok=0; }
  fi
  if [[ $ok == 1 ]]; then
    echo "tree restored to its starting state ($([[ -n "$INITIAL" ]] && echo "patched: $INITIAL" || echo upstream))"
  else
    echo "reproduce_benchmark: could not restore the tree's starting state; check: python -m robodojo_turbo status --root $RD" >&2
  fi
}
trap restore EXIT

to_upstream(){ if patched; then RDTURBO revert --root "$RD" || die "revert failed"; fi; patched && die "tree is still patched after revert"; return 0; }
ALL=$(python -c "from robodojo_turbo.engine import select; from robodojo_turbo.patches import REGISTRY; print(','.join(p.name for p in select(REGISTRY, 'all')))")
to_patched(){   # the full profile, whatever the tree had before
  if patched && [[ "$(patch_list)" != "$ALL" ]]; then RDTURBO revert --root "$RD" || die "revert failed"; fi
  if ! patched; then RDTURBO apply --root "$RD" --profile all >/dev/null || die "apply failed"; fi
  [[ "$(RDTURBO status --root "$RD" | python -c "import json,sys; print(json.load(sys.stdin).get('status'))")" == applied ]] || die "tree is not fully applied"
}
# unset every switch a user shell may carry into a leg; keep the launcher's own configuration
KEEP='^(RDTURBO_CONDA_ENV|RDTURBO_NV_MIRROR|RDTURBO_ROBODOJO_ROOT|RDTURBO_RUNS)$'
unset_args(){ env | grep -oE '^(RDTURBO_[A-Z0-9_]+|XLA_PYTHON_CLIENT_[A-Z0-9_]+|JAX_COMPILATION_CACHE_DIR)=' | tr -d '=' | grep -vE "$KEEP" | sed 's/^/-u /'; }
jax_arg(){ [[ $SAME_JAX == 1 ]] && echo "JAX_COMPILATION_CACHE_DIR=$HOME/.cache/robodojo_turbo/jax"; return 0; }
common=(--rd "$RD" --task "$TASK" --seed "$SEED" --layouts "$LAYOUTS" --workers 1 --conda-env "$CONDA_ENV" "${ARGS[@]}")

leg(){
  local name=$1 rep=$2 tag="${PREFIX}_${1}_r${2}"
  case $name in
    upstream) to_upstream; env $(unset_args) $(jax_arg) bash "$HERE/rdturbo_eval.sh" --tag "$tag" --out "$RUNS/$tag" --allow-unpatched "${common[@]}";;
    harness)  to_patched;  env $(unset_args) $(jax_arg) bash "$HERE/rdturbo_eval.sh" --tag "$tag" --out "$RUNS/$tag" --preset harness "${common[@]}";;
    speedup)  to_patched;  env $(unset_args) bash "$HERE/rdturbo_eval.sh" --tag "$tag" --out "$RUNS/$tag" --preset speedup "${common[@]}";;
  esac
}

for (( r=1; r<=REPS; r++ )); do
  if (( r % 2 )); then order="upstream harness speedup"; else order="speedup harness upstream"; fi
  for name in $order; do
    [[ ",$LEGS," == *",$name,"* ]] || continue
    echo "=== rep $r leg $name"
    leg "$name" "$r" || echo "leg $name rep $r failed"
  done
done

echo; echo "leg rep wall_total_s successes/episodes score"
for f in "$RUNS/${PREFIX}"_*_r*/merged_result.json; do
  [[ -f "$f" ]] || continue
  python -c "
import json, sys, os
d = json.load(open(sys.argv[1])); p = d.get('provenance', {})
print(os.path.basename(os.path.dirname(sys.argv[1])), p.get('wall_total_s'), f\"{d['successes']}/{d['eval_time']}\", round(d['score'], 1))" "$f"
done
