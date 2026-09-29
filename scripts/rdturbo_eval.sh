#!/usr/bin/env bash
# Run a RoboDojo evaluation on one machine with RoboDojo-Turbo's harness: layout shards across W Isaac client processes,
# one policy server process per worker (a shared server interleaves the two-call update_obs/get_action protocol),
# span tracing, GPU sampling, merged score and bubble report.
#
# Prerequisites
#   * a RoboDojo checkout patched once with `python -m robodojo_turbo apply --root <RoboDojo>` (or --allow-unpatched to
#     time plain upstream);
#   * RoboDojo's own environment (conda env with Isaac Sim / Isaac Lab, XPolicyLab policy env, checkpoints);
#   * NVIDIA's Omniverse license read and accepted by you: export OMNI_KIT_ACCEPT_EULA=YES (this script never sets it).
# Switches come from the environment; --preset <a[,b]> sources robodojo_turbo/presets/<a>.env (then <b>.env) first,
# e.g. --preset speedup,verify for the check modes on top of the speedup preset.
#
# Options
#   --tag T            run name (required); results go to --out (default ${RDTURBO_RUNS:-./runs}/T), which must not exist
#   --rd DIR           RoboDojo checkout (or RDTURBO_ROBODOJO_ROOT)
#   --preset P[,P2]    speedup | harness | off | verify (verify is meant on top of speedup: --preset speedup,verify)
#   --task T --ckpt C --seed S --policy Pi_05 --policy-env uv --action-type joint
#   --layouts N        evaluate layouts 0..N-1 (default 25), or --layout-ids 0-9,12 (patched trees only)
#   --workers W        Isaac client processes (patched trees only when W > 1); --num-envs E per process (default 10)
#   --port P[,P2,...]  one distinct policy-server port per worker; --policy-host H to use servers you started elsewhere
#   --gpu G --stagger S --server-wait S --conda-env NAME ('' = use the current environment)
#   --stall-timeout S  stop the run when no worker trace has grown for S seconds (default 1200; 0 = off; patched trees)
#   --allow-unpatched  time an unpatched (upstream) tree; refused if the tree is patched
#
# Example
#   OMNI_KIT_ACCEPT_EULA=YES bash scripts/rdturbo_eval.sh --rd ~/RoboDojo --tag demo --preset speedup \
#       --task stack_blocks --ckpt RoboDojo-sim-arx_x5-joint-0 --seed 0 --layouts 25
set -uo pipefail

TAG=""; RD="${RDTURBO_ROBODOJO_ROOT:-}"; OUT=""; PRESET=""; ALLOW_UNPATCHED=0
POLICY=Pi_05; TASK=stack_blocks; CKPT=RoboDojo-sim-arx_x5-joint-0; PENV=uv; ATYPE=joint
SEED=0; LAYOUTS=25; LAYOUT_IDS=""; WORKERS=1; NUM_ENVS=10; PORT=6000; GPU=0; POLICY_HOST=""; STAGGER=20
CONDA_ENV="${RDTURBO_CONDA_ENV-RoboDojo}"; SERVER_WAIT=900; STALL=1200

die(){ echo "rdturbo_eval: $*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in --allow-unpatched|-h|--help) ;; -*) [[ $# -ge 2 && "$2" != --* ]] || die "$1 needs a value";; esac
  case "$1" in
    --tag) TAG=$2; shift 2;; --rd) RD=$2; shift 2;; --out) OUT=$2; shift 2;; --preset) PRESET=$2; shift 2;;
    --policy) POLICY=$2; shift 2;; --task) TASK=$2; shift 2;; --ckpt) CKPT=$2; shift 2;; --policy-env) PENV=$2; shift 2;;
    --action-type) ATYPE=$2; shift 2;; --seed) SEED=$2; shift 2;; --layouts) LAYOUTS=$2; shift 2;;
    --layout-ids) LAYOUT_IDS=$2; shift 2;; --workers) WORKERS=$2; shift 2;; --num-envs) NUM_ENVS=$2; shift 2;;
    --port) PORT=$2; shift 2;; --gpu) GPU=$2; shift 2;; --policy-host) POLICY_HOST=$2; shift 2;; --stagger) STAGGER=$2; shift 2;;
    --conda-env) CONDA_ENV=$2; shift 2;; --server-wait) SERVER_WAIT=$2; shift 2;; --stall-timeout) STALL=$2; shift 2;;
    --allow-unpatched) ALLOW_UNPATCHED=1; shift;;
    -h|--help) awk 'NR > 1 && /^#/ {sub(/^# ?/, ""); print; next} NR > 1 {exit}' "$0"; exit 0;;
    *) die "unknown argument $1";;
  esac
done
[[ -n "$TAG" ]] || die "--tag is required"
[[ -n "$RD" && -d "$RD/src/eval_client" ]] || die "--rd must point at a RoboDojo checkout (or set RDTURBO_ROBODOJO_ROOT)"
RD=$(cd "$RD" && pwd)
PKG_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "${OUT:-${RDTURBO_RUNS:-$PWD/runs}/$TAG}")   # absolute: the script later runs from $RD
[[ -e "$OUT" ]] && die "$OUT exists; choose another --tag or --out"

# --- guards (all before anything is created or started) -----------------------------------------------------------------
case "${OMNI_KIT_ACCEPT_EULA:-}" in YES|yes|Y|y) ;; *) die "Isaac Sim needs NVIDIA's Omniverse license accepted: read it, then export OMNI_KIT_ACCEPT_EULA=YES";; esac
IFS=, read -ra PRESETS <<<"$PRESET"
for pr in "${PRESETS[@]}"; do
  PF="$PKG_ROOT/robodojo_turbo/presets/$pr.env"
  [[ -f "$PF" ]] || die "no preset $pr; available: $(cd "$PKG_ROOT/robodojo_turbo/presets" && ls *.env | sed 's/\.env$//' | tr '\n' ' ')"
  set -a; source "$PF"; set +a
done
if [[ -n "$CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" || \
    die "cannot activate conda env $CONDA_ENV (RoboDojo's install creates it; pass --conda-env <name>, or --conda-env '' to use the current environment)"
fi
export PYTHONPATH="$RD:$RD/XPolicyLab:$PKG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PKG_VERSION=$(python -c "import robodojo_turbo; print(robodojo_turbo.__version__)" 2>/dev/null || echo unknown)
PKG_GIT=$(git -C "$PKG_ROOT" describe --always --dirty 2>/dev/null || true)
PKG_VERSION="$PKG_VERSION${PKG_GIT:+ (git $PKG_GIT)}"

STATUS=$(python -m robodojo_turbo status --root "$RD") || die "cannot read the patch status of $RD"
PATCHED=0; grep -q '"patched": true' <<<"$STATUS" && PATCHED=1
NOTES=()
if [[ $PATCHED == 1 ]]; then
  if [[ $ALLOW_UNPATCHED == 1 ]]; then
    HINT=$(python -c "import json,sys; print(json.load(sys.stdin).get('hint') or '')" <<<"$STATUS")
    [[ -n "$HINT" ]] && die "--allow-unpatched given but $RD is not upstream: $HINT"
    die "--allow-unpatched given but $RD is patched; run \`python -m robodojo_turbo revert --root $RD\` to time upstream"
  fi
  CHK=$(python -m robodojo_turbo check-env --root "$RD") || die "patched tree and switches disagree:$(printf '\n%s' "$CHK")"
  grep -q '"unpinned": true' <<<"$STATUS" && NOTES+=("WARNING: the tree was patched with --force-unpinned; no physics or scoring claim of this project applies") && UNPINNED=1
else
  [[ $ALLOW_UNPATCHED == 1 ]] || die "$RD is not patched; run: python -m robodojo_turbo apply --root $RD  (or pass --allow-unpatched to time upstream)"
  (( WORKERS > 1 )) && die "--workers > 1 needs the layout_shards patch (upstream would evaluate the same layouts in every worker)"
  [[ -n "$LAYOUT_IDS" ]] && die "--layout-ids needs the layout_shards patch"
  env | grep -q '^RDTURBO_' && NOTES+=("WARNING: RDTURBO_* switches are set but the tree is unpatched; they have no effect")
  NOTES+=("NOTE: unpatched tree: RoboDojo chooses num_envs and caps eval_num per task itself; --num-envs is ignored")
fi
[[ "${RDTURBO_NO_PLANNER:-0}" == 1 && "$ATYPE" != joint ]] && die "RDTURBO_NO_PLANNER=1 is only valid for --action-type joint (end-effector actions need the planner)"
[[ "$POLICY" != Pi_05 ]] && NOTES+=("NOTE: the chunk-start upload, VIDEO_ONLY_OBS and USD_LAST hooks live in the Pi_05 loop; with $POLICY they are no-ops")

SHARDS=$(python -m robodojo_turbo.tools.shard --layouts "$LAYOUTS" --workers "$WORKERS" --num-envs "$NUM_ENVS" ${LAYOUT_IDS:+--layout-ids "$LAYOUT_IDS"}) || die "shard plan failed"
NW=$(grep -c . <<<"$SHARDS")            # actual worker count (capped by layouts / num_envs)
IFS=, read -ra PORTS <<<"$PORT"
(( ${#PORTS[@]} == $(printf '%s\n' "${PORTS[@]}" | sort -u | grep -c .) )) || die "--port has duplicates: $PORT"
(( NW > 1 && ${#PORTS[@]} < NW )) && die "$NW workers need one server port each (--port 6000,6001,...); a shared server interleaves observations between workers"
port_open(){ (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
if [[ -z "$POLICY_HOST" ]]; then
  for (( k=0; k<NW; k++ )); do port_open "${PORTS[$k]}" && die "port ${PORTS[$k]} is already in use (a stale policy server?); refusing to start"; done
fi

# --- run directory ----------------------------------------------------------------------------------------------------
mkdir -p "$OUT"
UNPINNED=${UNPINNED:-0}
RUN_ID="${TAG}_$(date +%Y%m%d%H%M%S)_$$"   # unique per invocation: upstream resumes and saves results by run id
log(){ echo "[rdturbo $(date -Iseconds)] $*" | tee -a "$OUT/meta.txt"; }
for n in "${NOTES[@]}"; do log "$n"; done
log "tag=$TAG run_id=$RUN_ID policy=$POLICY task=$TASK ckpt=$CKPT action_type=$ATYPE seed=$SEED layouts=${LAYOUT_IDS:-0-$((LAYOUTS-1))} workers=$NW num_envs=$NUM_ENVS port=$PORT gpu=$GPU policy_host=${POLICY_HOST:-local} host=$(hostname) nproc=$(nproc)"
log "switches: $(env | grep -E '^(RDTURBO_|XLA_PYTHON_CLIENT_|JAX_COMPILATION_CACHE_DIR)' | sort | tr '\n' ' ')"
log "robodojo=$(git -C "$RD" rev-parse --short HEAD 2>/dev/null) xpolicylab=$(git -C "$RD/XPolicyLab" rev-parse --short HEAD 2>/dev/null) robodojo_turbo=$PKG_VERSION"
echo "$STATUS" > "$OUT/patch_status.json"
echo "$SHARDS" > "$OUT/shards.jsonl"
cat "$OUT/shards.jsonl" >> "$OUT/meta.txt"

SPIDS=(); WPIDS=(); SMI=""
cleanup(){   # on every exit path: stop our process groups (TERM, then KILL after a grace period) and the GPU sampler
  local p alive=0
  [[ -n "$SMI" ]] && kill "$SMI" 2>/dev/null
  for p in "${WPIDS[@]}" "${SPIDS[@]}"; do kill -TERM -- -"$p" 2>/dev/null && alive=1; done
  (( alive )) || return 0
  sleep 5
  for p in "${WPIDS[@]}" "${SPIDS[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'log "interrupted by SIGINT; stopping servers and workers"; exit 130' INT
trap 'log "interrupted by SIGTERM; stopping servers and workers"; exit 143' TERM
trap 'log "interrupted by SIGHUP; stopping servers and workers"; exit 129' HUP

# --- policy servers (local mode): one process per worker ---------------------------------------------------------------
TL=$(date +%s)   # wall_total starts here: comparable across overlap / non-overlap start-up
HOST=${POLICY_HOST:-127.0.0.1}
if [[ -z "$POLICY_HOST" ]]; then
  cd "$RD"
  for (( k=0; k<NW; k++ )); do
    p=${PORTS[$k]}
    CUDA_VISIBLE_DEVICES=$GPU setsid nohup bash scripts/robodojo.sh server --policy-dir "XPolicyLab/policy/$POLICY" --task "$TASK" \
      --ckpt "$CKPT" --policy-env "$PENV" --action-type "$ATYPE" --policy-port "$p" --policy-gpu "$GPU" --bind-host 127.0.0.1 \
      > "$OUT/server_$k.out" 2>&1 < /dev/null &
    SPIDS+=($!); log "server $k pid=${SPIDS[-1]} port=$p starting"
  done
fi
wait_servers(){
  [[ -z "$POLICY_HOST" ]] || return 0
  local k
  for (( k=0; k<${#SPIDS[@]}; k++ )); do
    until port_open "${PORTS[$k]}"; do
      kill -0 "${SPIDS[$k]}" 2>/dev/null || { log "server $k died: $(tail -3 "$OUT/server_$k.out" | tr '\n' ' ')"; exit 1; }
      (( $(date +%s) - TL > SERVER_WAIT )) && { log "server $k port ${PORTS[$k]} not open after ${SERVER_WAIT}s"; exit 1; }
      sleep 2
    done
    kill -0 "${SPIDS[$k]}" 2>/dev/null || { log "port ${PORTS[$k]} is open but server $k (pid ${SPIDS[$k]}) is gone: someone else owns the port"; exit 1; }
    log "server $k port ${PORTS[$k]} open $(( $(date +%s) - TL ))s after server start"
  done
}
[[ "${RDTURBO_OVERLAP_START:-0}" == 1 ]] || wait_servers   # overlap: clients retry the websocket for up to 900 s

# --- GPU sampling ------------------------------------------------------------------------------------------------------
nvidia-smi -i "$GPU" --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 1 > "$OUT/gpu.csv" 2>/dev/null &
SMI=$!

# --- client workers ----------------------------------------------------------------------------------------------------
cd "$RD"; T0=$(date +%s)
while read -r row; do
  i=$(python -c "import json,sys;print(json.loads(sys.argv[1])['worker'])" "$row")
  ids=$(python -c "import json,sys;print(json.loads(sys.argv[1])['ids'])" "$row")
  n=$(python -c "import json,sys;print(json.loads(sys.argv[1])['n'])" "$row")
  WPORT=${PORTS[$(( i % ${#PORTS[@]} ))]}
  ROBODOJO_RUN_ID=${RUN_ID}_w$i RDTURBO_WORKER=w$i RDTURBO_TRACE=$OUT/trace_w$i.jsonl RDTURBO_STATUS=$OUT/status_w$i.json \
  RDTURBO_LAYOUT_IDS=$ids RDTURBO_NUM_ENVS=$NUM_ENVS RDTURBO_EVAL_NUM_FORCE=1 \
    setsid nohup bash scripts/robodojo.sh client --task "$TASK" --policy-dir "XPolicyLab/policy/$POLICY" --policy-host "$HOST" \
      --policy-port "$WPORT" --eval-num "$n" --ckpt "$CKPT" --action-type "$ATYPE" --seed "$SEED" --env-gpu "$GPU" --connect-timeout 30 \
      > "$OUT/client_w$i.out" 2>&1 < /dev/null &
  WPIDS+=($!); log "worker w$i pid=${WPIDS[-1]} layouts=$ids n=$n port=$WPORT"
  sleep "$STAGGER"
done < "$OUT/shards.jsonl"
[[ "${RDTURBO_OVERLAP_START:-0}" == 1 ]] && wait_servers

RC=0
stalled(){   # true when no worker trace has grown for STALL seconds (Kit can hang after a CUDA error while its log keeps growing)
  (( STALL > 0 && PATCHED == 1 )) || return 1
  local newest=0 f m
  for f in "$OUT"/trace_w*.jsonl; do [[ -f "$f" ]] && m=$(stat -c %Y "$f") && (( m > newest )) && newest=$m; done
  (( newest == 0 )) && newest=$T0
  (( $(date +%s) - newest > STALL ))
}
for k in "${!WPIDS[@]}"; do
  while kill -0 "${WPIDS[$k]}" 2>/dev/null; do
    if stalled; then log "no worker trace has grown for ${STALL}s: stopping the run (see client_w*.out)"; RC=1; cleanup; WPIDS=(); SPIDS=(); break 2; fi
    sleep 5
  done
  wait "${WPIDS[$k]}" 2>/dev/null; rc=$?
  (( rc != 0 )) && RC=1
  log "worker w$k exited rc=$rc at +$(( $(date +%s) - T0 ))s"
done
TE=$(date +%s); WALL=$(( TE - TL )); WALL_CLIENTS=$(( TE - T0 ))
log "all workers done wall_total=${WALL}s (server start -> last client exit) wall_clients=${WALL_CLIENTS}s"
cleanup; SMI=""; WPIDS=(); SPIDS=()
log "servers stopped"

# --- merge + report ----------------------------------------------------------------------------------------------------
RES=()
for k in $(seq 0 $(( NW - 1 ))); do
  f=$(find "$RD/eval_result" -type f -path "*/${RUN_ID}_w${k}/_result.json" 2>/dev/null)
  if [[ $(grep -c . <<<"$f") == 1 ]]; then cp "$f" "$OUT/w${k}_result.json"; RES+=("$OUT/w${k}_result.json")
  else log "worker w$k: expected one _result.json for run id ${RUN_ID}_w$k, found $(grep -c . <<<"$f")"; RC=1; fi
done
EXPECT=${LAYOUT_IDS:-0-$((LAYOUTS-1))}
UNSTABLE=$(python -c "import json,glob,sys; print(sum(int(json.load(open(f)).get('unstable_nums', 0)) for f in glob.glob(sys.argv[1] + '/status_w*.json')))" "$OUT" 2>/dev/null || echo 0)
if [[ ${#RES[@]} -gt 0 ]]; then
  python -m robodojo_turbo.tools.merge --strict --expected "$EXPECT" --unstable-count "${UNSTABLE:-0}" --meta tag="$TAG" --meta run_id="$RUN_ID" --meta task="$TASK" --meta ckpt="$CKPT" \
    --meta seed="$SEED" --meta workers="$NW" --meta num_envs="$NUM_ENVS" --meta wall_total_s="$WALL" --meta wall_clients_s="$WALL_CLIENTS" \
    --meta robodojo_turbo="$PKG_VERSION" --meta unpinned="$UNPINNED" \
    --meta switches="$(env | grep -E '^RDTURBO_' | grep -vE '^RDTURBO_(NV_MIRROR|ROBODOJO_ROOT|RUNS|CONDA_ENV|TRACE|STATUS)=' | sort | tr '\n' ',')" \
    "${RES[@]}" > "$OUT/merged_result.json" 2>> "$OUT/meta.txt" || RC=1
  python -m robodojo_turbo.tools.report --out "$OUT" --client "$OUT"/trace_w*.jsonl --gpu-sim "$OUT/gpu.csv" \
    --result "$OUT/merged_result.json" --title "$TAG" >> "$OUT/meta.txt" 2>&1 || log "report failed (no traces on an unpatched tree?)"
fi
for s in "$OUT"/status_w*.json; do
  [[ -f "$s" ]] && ! grep -q '"ok": true' "$s" && { log "accounting: $(cat "$s")"; RC=1; }
done
log "done: wall_total=${WALL}s wall_clients=${WALL_CLIENTS}s rc=$RC $(python -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"successes={d['successes']}/{d['eval_time']} score={d['score']:.1f} missing={d['missing']} unstable={d.get('unstable_layouts', [])}\")" "$OUT/merged_result.json" 2>/dev/null)"
exit $RC
