#!/usr/bin/env bash
# run_pipeline.sh — Linux / macOS runner
# Usage: bash run_pipeline.sh [--event_id ID] [--random] [--include_normal]
#                              [--seed N] [--skip_build]
set -euo pipefail
EVENT_ID=""; SKIP=false; RANDOM_PICK=false
INCLUDE_NORMAL=false; SEED=""
DATA_DIR="./data"; OUTPUT_DIR="./output"; SCRIPTS="./scripts"
while [[ $# -gt 0 ]]; do case $1 in
    --event_id)          EVENT_ID="$2";   shift 2 ;;
    --random)            RANDOM_PICK=true; shift  ;;
    --include_normal)    INCLUDE_NORMAL=true; shift ;;
    --seed)              SEED="$2";       shift 2 ;;
    --skip_build)        SKIP=true;       shift   ;;
    --data_dir)          DATA_DIR="$2";   shift 2 ;;
    --output_dir)        OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
esac; done
G="\033[0;32m"; C="\033[0;36m"; Y="\033[1;33m"; N="\033[0m"
step() { echo -e "\n${C}>>  $1${N}"; }
ok()   { echo -e "${G}OK  $1${N}"; }
warn() { echo -e "${Y}!!  $1${N}"; }

step "Checking Python"; python3 --version; ok "Python found"
step "Installing dependencies"
python3 -m pip install numpy pandas scipy plotly --quiet; ok "Ready"
if ! $SKIP; then
    step "Building datasets"; mkdir -p "$OUTPUT_DIR"
    python3 "$SCRIPTS/build_crash_dataset.py" \
        --synshrp2_dir "$DATA_DIR/synshrp2" \
        --ciss_dir     "$DATA_DIR/ciss"     \
        --har_dir      "$DATA_DIR/har"      \
        --output_dir   "$OUTPUT_DIR"
    ok "Output: $OUTPUT_DIR"
else
    warn "Skipping build (--skip_build)"
fi
step "Visualizing"
ARGS=("$SCRIPTS/visualize_crash.py" "--output_dir" "$OUTPUT_DIR")
[[ -n "$EVENT_ID" ]] && ARGS+=("--event_id" "$EVENT_ID")
$RANDOM_PICK && ARGS+=("--random")
$INCLUDE_NORMAL && ARGS+=("--include_normal")
[[ -n "$SEED" ]] && ARGS+=("--seed" "$SEED")
python3 "${ARGS[@]}"
FILE=$(ls -t "$OUTPUT_DIR"/plot_*.html 2>/dev/null | head -1 || true)
if [[ -n "${FILE:-}" ]]; then
    ok "Output: $FILE"
    command -v xdg-open &>/dev/null && xdg-open "$FILE" 2>/dev/null || \
    command -v open     &>/dev/null && open "$FILE"     2>/dev/null || true
fi
echo -e "\n${G}Done. Output: $OUTPUT_DIR${N}\n"
