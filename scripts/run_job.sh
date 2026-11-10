#!/usr/bin/env bash
# Launch a long job detached from the terminal, log it, and notify via Telegram.
#
#   scripts/run_job.sh [-C DIR] [-n NAME] [-e ENV] [--fg] -- CMD [ARGS...]
#   scripts/run_job.sh [--fg]        # no command: runs the job defined in ./job.sh
#                                    # (this is what `runbg` does inside this repo)
#
#   -C DIR   working directory, relative to the repo root (default: repo root).
#            KuaiSearch scripts use hardcoded relative paths: use -C external/KuaiSearch
#   -n NAME  job name for logs/notification (default: derived from CMD and ARGS)
#   -e ENV   conda env (default: sid-search)
#   --fg     run in the foreground (no detach); useful for short tests
#
# Examples:
#   scripts/run_job.sh -C external/KuaiSearch -- python recall/data/process.py
#   scripts/run_job.sh -C external/KuaiSearch -n dsi -- torchrun --nproc_per_node=1 recall/GR/train.py ...
#
# Logs: logs/<YYYYmmdd-HHMMSS>_<NAME>.{out,err} at the repo root (gitignored).
# Every run gets its own timestamped files, so a re-run never overwrites a previous one.
# The .out starts with a header recording the command, cwd, commits of sid-search and of
# the KuaiSearch submodule (with dirty flag), conda env and GPU, so each result can be
# traced back to the exact code that produced it.
#
# Telegram credentials are read from ~/.telegram_env (TG_BOT_TOKEN, TG_CHAT_ID); they are
# never stored in the repo.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO/scripts/run_job.sh"
TELEGRAM_ENV="$HOME/.telegram_env"

JOB_FILE="$REPO/job.sh"

usage() { sed -n '2,17p' "$SELF" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

WORKDIR="" NAME="" ENV_NAME="" FG=0
while [ $# -gt 0 ]; do
  case "$1" in
    -C) WORKDIR="$2"; shift 2 ;;
    -n) NAME="$2"; shift 2 ;;
    -e) ENV_NAME="$2"; shift 2 ;;
    --fg) FG=1; shift ;;
    -h|--help) usage 0 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1 (did you forget '--' before the command?)" >&2; usage ;;
  esac
done
if [ $# -gt 0 ]; then
  CMD=("$@")
  JOB_SOURCE="command line"
else
  # No command given: take the job from job.sh (WORKDIR, NAME, ENV_NAME, CMD=(...)).
  # Command-line -C/-n/-e still override what job.sh says.
  [ -f "$JOB_FILE" ] || { echo "No command after '--' and no $JOB_FILE" >&2; usage; }
  cli_workdir="$WORKDIR" cli_name="$NAME" cli_env="$ENV_NAME"
  CMD=()
  # shellcheck disable=SC1090
  source "$JOB_FILE"
  WORKDIR="${cli_workdir:-${WORKDIR:-}}" NAME="${cli_name:-${NAME:-}}" ENV_NAME="${cli_env:-${ENV_NAME:-}}"
  [ ${#CMD[@]} -gt 0 ] || { echo "$JOB_FILE does not define CMD=(...)" >&2; exit 1; }
  JOB_SOURCE="job.sh"
fi
WORKDIR="${WORKDIR:-.}" ENV_NAME="${ENV_NAME:-sid-search}"

# ---------------------------------------------------------------------------
# Parent: validate everything up front (errors show up in your terminal, not in a
# log you would never look at), then re-launch itself detached.
# ---------------------------------------------------------------------------
if [ -z "${RUN_JOB_CHILD:-}" ]; then
  [ -d "$REPO/$WORKDIR" ] || { echo "Working dir not found: $REPO/$WORKDIR" >&2; exit 2; }
  [ -f "$TELEGRAM_ENV" ] || { echo "Missing $TELEGRAM_ENV" >&2; exit 10; }
  # shellcheck disable=SC1090
  ( source "$TELEGRAM_ENV"; : "${TG_BOT_TOKEN:?missing in $TELEGRAM_ENV}" "${TG_CHAT_ID:?missing in $TELEGRAM_ENV}" )
  [ -d "$HOME/miniconda3/envs/$ENV_NAME" ] || { echo "Conda env not found: $ENV_NAME" >&2; exit 3; }

  if [ -z "$NAME" ]; then
    # Name = script basename (first *.py argument, else the command) + slug of the rest.
    base="" rest=()
    for a in "${CMD[@]}"; do
      if [ -z "$base" ] && [[ "$a" == *.py ]]; then base="$(basename "$a" .py)"; else rest+=("$a"); fi
    done
    [ -n "$base" ] || { base="$(basename "${CMD[0]}")"; rest=("${CMD[@]:1}"); }
    slug="$(printf '%s ' "${rest[@]}" | sed 's/^python[0-9.]* //; s/^-u //' \
            | tr -cs '[:alnum:]' '_' | sed 's/^_//; s/_$//' | cut -c1-60)"
    NAME="$base${slug:+_$slug}"
  fi

  export RUN_JOB_CHILD=1 RUN_JOB_NAME="$NAME" RUN_JOB_STAMP="$(date +%Y%m%d-%H%M%S)"
  export RUN_JOB_SOURCE="$JOB_SOURCE"
  OUT="$REPO/logs/${RUN_JOB_STAMP}_${NAME}.out"
  ERR="$REPO/logs/${RUN_JOB_STAMP}_${NAME}.err"
  mkdir -p "$REPO/logs"
  # Archive the job.sh that defined this run (with its comments) next to its logs.
  if [ "$JOB_SOURCE" = "job.sh" ]; then
    cp "$JOB_FILE" "$REPO/logs/${RUN_JOB_STAMP}_${NAME}.job.sh"
  fi

  if [ "$FG" -eq 1 ]; then
    exec "$SELF" -C "$WORKDIR" -n "$NAME" -e "$ENV_NAME" -- "${CMD[@]}"
  fi
  # setsid: own session, so neither closing the terminal (SIGHUP) nor Ctrl-C reaches it.
  setsid nohup "$SELF" -C "$WORKDIR" -n "$NAME" -e "$ENV_NAME" -- "${CMD[@]}" \
    </dev/null >/dev/null 2>&1 &
  echo "Job '$NAME' launched in background with PID $!"
  echo "  tail -f $OUT"
  echo "  stderr: $ERR"
  exit 0
fi

# ---------------------------------------------------------------------------
# Child: run the job.
# ---------------------------------------------------------------------------
NAME="$RUN_JOB_NAME"
OUT="$REPO/logs/${RUN_JOB_STAMP}_${NAME}.out"
ERR="$REPO/logs/${RUN_JOB_STAMP}_${NAME}.err"
exec >>"$OUT" 2>>"$ERR"

# shellcheck disable=SC1090
source "$TELEGRAM_ENV"
set +u  # conda's activation scripts are not nounset-safe
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

cd "$REPO/$WORKDIR"
export PYTHONUNBUFFERED=1  # unbuffered output for python and anything torchrun spawns

git_state() {  # <dir> -> "<short sha> (clean|dirty)"
  local sha
  sha="$(git -C "$1" rev-parse --short HEAD 2>/dev/null || echo '?')"
  if [ -n "$(git -C "$1" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    echo "$sha (dirty)"
  else
    echo "$sha (clean)"
  fi
}

{
  echo "=== run_job: $NAME"
  echo "start:      $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "host:       $(hostname)    pid: $$"
  echo "cwd:        $PWD"
  printf 'command:   '; printf ' %q' "${CMD[@]}"; echo
  echo "defined in: ${RUN_JOB_SOURCE:-command line}$([ "${RUN_JOB_SOURCE:-}" = job.sh ] && echo " (copy: logs/${RUN_JOB_STAMP}_${NAME}.job.sh)")"
  echo "sid-search: $(git_state "$REPO")"
  echo "KuaiSearch: $(git_state "$REPO/external/KuaiSearch")  base 88a545d"
  echo "conda env:  $ENV_NAME ($(python --version 2>&1))"
  echo "gpu:        $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | paste -sd';' || echo none)"
  echo "==="
}

START=$(date +%s)
RET=0
"${CMD[@]}" || RET=$?
ELAPSED=$(( $(date +%s) - START ))
RUNTIME=$(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo "=== end: $(date '+%Y-%m-%d %H:%M:%S %z')  runtime $RUNTIME  exit $RET"

if [ "$RET" -eq 0 ]; then STATUS="✅ Finalizado correctamente"
else STATUS="❌ Finalizado con error (exit ${RET})"; fi

MSG="Job: ${NAME}
Repo: sid-search ($(git_state "$REPO"))
Command: ${CMD[*]}
Dir: ${WORKDIR}
Server: $(hostname)
Execution time: ${RUNTIME}
----------
${STATUS}"
if [ "$RET" -ne 0 ]; then
  MSG="${MSG}
----------
$(tail -n 5 "$ERR" | cut -c1-200)"
fi

curl -s -X POST "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
  -d "chat_id=${TG_CHAT_ID}" --data-urlencode "text=${MSG:0:4000}" >/dev/null || true

exit "$RET"
