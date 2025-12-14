#!/usr/bin/env zsh
set -euo pipefail

# ---------- resolve script path ----------
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
else
  SCRIPT_PATH="${(%):-%N}"
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"

echo "Script path : $SCRIPT_PATH"
echo "Script dir  : $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Data dir    : $DATA_DIR"
echo

# ---------- sanity check ----------
if [[ ! -d "$DATA_DIR" ]]; then
  echo "❌ Data directory does not exist: $DATA_DIR"
  exit 1
fi

# ---------- convert ----------
find "$DATA_DIR" -type f -name "*.mp4" ! -name "*.tmp.mp4" | while read -r vid; do
  echo "▶ Converting: $vid"

  tmp="${vid}.tmp.mp4"

  ffmpeg -nostdin -y -loglevel error \
    -i "$vid" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -profile:v baseline \
    -level 3.0 \
    -movflags +faststart \
    "$tmp"

  mv "$tmp" "$vid"
done

echo
echo "✅ All videos converted successfully."
