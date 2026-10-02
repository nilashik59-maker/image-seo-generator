#!/usr/bin/env bash
# Render the bundled demo loop (green-screen fixture -> seamless micro-animation).
#
#   python -m pip install -r renderer/requirements.txt
#   ./renderer/demo.sh                 # or: PYTHON=/path/to/venv/bin/python ./renderer/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c "import numpy, PIL" 2>/dev/null || {
  echo "error: numpy/Pillow not importable by '$PYTHON'." >&2
  echo "       python -m pip install -r renderer/requirements.txt" >&2
  echo "       or set PYTHON=/path/to/venv/bin/python" >&2
  exit 1
}

"$PYTHON" renderer/micro_anim.py \
  --input examples/fixture_green_screen.jpg \
  --out examples/out/fixture_loop \
  --zones examples/fixture_zones.json \
  --frames 120 --fps 30 \
  --debug-dir examples/out/debug --debug-frames 0,15,30,45,60,75,90,105
