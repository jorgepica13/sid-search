# Job for `runbg` (= scripts/run_job.sh with no arguments).
# Copy to the repo root as job.sh (gitignored) and edit it before each launch:
#   cp scripts/job.example.sh job.sh
# A copy of job.sh, comments included, is archived next to the run's logs.
#
# Describe the experiment here (what, why, expected time, where outputs go):
# --- <phase / step>: <what this run does> ---
# ~<expected time>  -> <output paths>

WORKDIR="external/KuaiSearch"   # relative to repo root; KuaiSearch scripts need their root
NAME=""                         # empty = derived from the command
ENV_NAME="sid-search"
CMD=(python recall/data/process.py)
