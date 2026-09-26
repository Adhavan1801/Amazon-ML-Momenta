#!/usr/bin/env bash
# Run the full pipeline on SageMaker.
#
#  Studio / JupyterLab terminal:
#     bash run_on_sagemaker.sh s3://<bucket> <run-name> [extra run_all.py args, e.g. --from train1]
#  Processing job (called by launch_processing_job.py):
#     bash run_on_sagemaker.sh processing <run-name>
set -euo pipefail

MODE_OR_S3="$1"; RUN="${2:-run1}"; shift 2 || true
EXTRA=("$@")

if [[ "$MODE_OR_S3" == "processing" ]]; then
  # inputs are mounted by SageMaker; outputs in /opt/ml/processing/output are uploaded to S3 by SageMaker
  CODE=/opt/ml/processing/code; DATA=/opt/ml/processing/dataset; UTILS=/opt/ml/processing/utils
  WORK=/opt/ml/processing/work; OUTROOT=/opt/ml/processing/output
  S3=""
else
  S3="${MODE_OR_S3%/}"
  ROOT="$HOME/ber"; CODE="$ROOT/code"; DATA="$ROOT/dataset"; UTILS="$ROOT/utils"
  WORK="$ROOT/work"; OUTROOT="$ROOT/runs/$RUN"
  mkdir -p "$ROOT"
  aws s3 sync "$S3/code/" "$CODE/" --only-show-errors
  aws s3 sync "$S3/dataset/" "$DATA/" --only-show-errors --exclude "*.DS_Store"
  aws s3 sync "$S3/utils/" "$UTILS/" --only-show-errors
fi
mkdir -p "$WORK" "$OUTROOT/output"

# ---- Python 3.11+ environment
PY=python3
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  if command -v conda >/dev/null; then
    [[ -d "$WORK/py311" ]] || conda create -y -q -p "$WORK/py311" python=3.11
    PY="$WORK/py311/bin/python"
  else
    echo "Python >= 3.11 required"; exit 1
  fi
fi
VPY="$PY"
if [[ -x "$WORK/venv/bin/python" ]] || $PY -m venv "$WORK/venv" >/dev/null 2>&1; then
  "$WORK/venv/bin/python" -m pip --version >/dev/null 2>&1 && VPY="$WORK/venv/bin/python"
fi
$VPY -m pip install -q --upgrade pip
$VPY -m pip install -q -r "$CODE/requirements.txt"

# ---- resources
THREADS=$(nproc)
MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024 * 0.9}' /proc/meminfo)
echo "threads=$THREADS mem_gb=$MEM_GB python=$($VPY --version)"

# ---- run
set +e
$VPY -u "$CODE/src/run_all.py" --data "$DATA" --work "$WORK" --out "$OUTROOT/output" \
     --threads "$THREADS" --mem-gb "$MEM_GB" --validate "$UTILS/validate_submission.py" "${EXTRA[@]}" \
     2>&1 | tee "$OUTROOT/run.log"
STATUS=${PIPESTATUS[0]}
set -e
cp "$WORK/artifacts/decision.json" "$OUTROOT/" 2>/dev/null || true
grep -h "\[eval\]\|\"blocking\"\|\"pruned\"\|PASS\|FAIL" "$OUTROOT/run.log" > "$OUTROOT/summary.txt" || true

if [[ -n "$S3" ]]; then
  aws s3 sync "$OUTROOT/" "$S3/runs/$RUN/" --only-show-errors
  echo "Uploaded results to $S3/runs/$RUN/"
fi
exit $STATUS
