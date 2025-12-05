#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh [output-name]
# If output-name is provided, all occurrences of
# "outputs/action_conditioned/<something>" in the JSON will be replaced
# with "outputs/action_conditioned/<output-name>" and the example will
# be run with that output folder.

OUT_NAME=${1:-cuda_graph}
ORIG_JSON="assets/action_conditioned/basic/inference_params.json"
OUTPUT_DIR="outputs/action_conditioned/${OUT_NAME}"

echo "Preparing run with output name: ${OUT_NAME}"

# Prepare output dir, copy the original JSON into the outputs folder and
# modify that copy where any string starting with
# "outputs/action_conditioned/" has the next path component replaced by
# OUT_NAME. The program will be invoked with the copy under outputs.
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
COPY_JSON="${OUTPUT_DIR}/$(basename "${ORIG_JSON}")"
cp "${ORIG_JSON}" "${COPY_JSON}"

python - "${OUT_NAME}" "${COPY_JSON}" <<PY > /dev/null
import json, sys
from pathlib import Path
suffix = sys.argv[1]
orig = Path(sys.argv[2])
def replace_str(s: str) -> str:
  if isinstance(s, str) and s.startswith("outputs/action_conditioned/"):
    rest = s[len("outputs/action_conditioned/"):]
    parts = rest.split('/', 1)
    if len(parts) == 1:
      return f"outputs/action_conditioned/{suffix}"
    else:
      return f"outputs/action_conditioned/{suffix}/{parts[1]}"
  return s

def walk(obj):
  if isinstance(obj, dict):
    return {k: walk(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [walk(v) for v in obj]
  if isinstance(obj, str):
    return replace_str(obj)
  return obj

data = json.loads(orig.read_text())
data2 = walk(data)
# write back to the original file (in-place modification)
orig.write_text(json.dumps(data2, indent=2))

sys.exit(0)
PY

echo "Prepared modified config: ${COPY_JSON} -> outputs at ${OUTPUT_DIR}"

python examples/action_conditioned.py \
  -i "${COPY_JSON}" \
  -o "${OUTPUT_DIR}"

echo "Run finished. Outputs written to ${OUTPUT_DIR}"