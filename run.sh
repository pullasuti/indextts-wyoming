#!/bin/bash
# IndexTTS-2.5 Wyoming bridge launcher.
#
# First run: installs uv if missing, clones index-tts at a pinned commit,
# builds its Python 3.11 venv (uv sync) and downloads the model plus its
# auxiliary models (~10 GB). Later runs skip straight to starting the bridge.
#
# Configuration: optional .env next to this script (see .env.example).
# Extra arguments are passed through to bridge_indextts.py, e.g.
#   ./run.sh --speed 1.1
# EXTRA_ARGS in .env may be a string or a bash array.
#
#   SETUP_ONLY=1 ./run.sh   set everything up, don't start the bridge
#   REINSTALL=1 ./run.sh    reinstall Python dependencies
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f .env ]; then
    set -a
    source <(tr -d '\r' < .env)
    set +a
fi

REPO_URL="${REPO_URL:-https://github.com/index-tts/index-tts.git}"
REPO_COMMIT="${REPO_COMMIT:-d9e41aac89fd00b3d71497fddb287b7f24613712}"
MODEL_ID="${MODEL_ID:-IndexTeam/IndexTTS-2.5}"
REPO_DIR="${REPO_DIR:-repo}"
MODEL_DIR="${MODEL_DIR:-$REPO_DIR/checkpoints}"
PROMPT_WAV="${PROMPT_WAV:-voices/nicoleheart.wav}"
VOICE="${VOICE:-nicoleheart}"
URI="${URI:-tcp://0.0.0.0:10300}"
LANG_CODE="${LANG_CODE:-EN}"
DEFAULT_MOOD="${DEFAULT_MOOD-warm}"
EMO_STRENGTH="${EMO_STRENGTH:-1.0}"
SPEED="${SPEED:-1.0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
PREBUFFER_SECONDS="${PREBUFFER_SECONDS:-1}"
MIN_MERGED_SECONDS="${MIN_MERGED_SECONDS:-2.5}"

# A stale .env value would otherwise only show up as an argparse error.
for v in EMO_STRENGTH SPEED PREBUFFER_SECONDS MIN_MERGED_SECONDS; do
    [[ "${!v}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "ERROR: $v='${!v}' is not a number" >&2; exit 1; }
done
[[ "$NUM_BEAMS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: NUM_BEAMS='$NUM_BEAMS' is not a positive integer" >&2; exit 1; }

# Only needed for setup; systemd's PATH usually lacks ~/.local/bin.
ensure_uv() {
    export PATH="${HOME:-/root}/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo ">>> Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
}

if [ ! -d "$REPO_DIR/.git" ]; then
    echo ">>> Cloning index-tts..."
    git clone "$REPO_URL" "$REPO_DIR"
    git -C "$REPO_DIR" checkout -q "$REPO_COMMIT"
fi

STAMP="$REPO_DIR/.venv/.bridge-deps"
WANT="$REPO_COMMIT wyoming==1.10.2"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$WANT" ] || [ "${REINSTALL:-0}" = 1 ]; then
    ensure_uv
    echo ">>> Installing dependencies..."
    if ! { git -C "$REPO_DIR" checkout -q "$REPO_COMMIT" \
           && (cd "$REPO_DIR" && uv sync) \
           && uv pip install -p "$REPO_DIR/.venv" wyoming==1.10.2 soundfile; }; then
        # Don't crash-loop the service if an older install is still usable.
        if "$REPO_DIR/.venv/bin/python" -c "import indextts.infer_v2_5, wyoming" 2>/dev/null; then
            echo "WARNING: dependency install failed; starting with the existing venv" >&2
        else
            echo "ERROR: dependency install failed" >&2
            exit 1
        fi
    else
        echo "$WANT" > "$STAMP"
    fi
fi

if [ ! -f "$MODEL_DIR/gpt.pth" ] || [ ! -d "$MODEL_DIR/hf_cache/bigvgan" ]; then
    echo ">>> Downloading $MODEL_ID and auxiliary models (~10 GB)..."
    (cd "$REPO_DIR" && .venv/bin/python -c "
import sys
from huggingface_hub import snapshot_download
from indextts.utils.model_download import ensure_models_available
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
ensure_models_available(sys.argv[2])
" "$MODEL_ID" "$(realpath -m --relative-to="$REPO_DIR" "$MODEL_DIR")")
fi

if [ ! -f "$PROMPT_WAV" ]; then
    echo "ERROR: reference voice $PROMPT_WAV not found." >&2
    echo "Put a clean 10-15 s mono WAV of the voice to clone there, or set" >&2
    echo "PROMPT_WAV in .env." >&2
    exit 1
fi

if [ "${SETUP_ONLY:-0}" = 1 ]; then
    echo ">>> Setup complete."
    exit 0
fi

ARGS=(
    --repo "$REPO_DIR"
    --model-dir "$MODEL_DIR"
    --prompt-wav "$PROMPT_WAV"
    --voice "$VOICE"
    --uri "$URI"
    --lang "$LANG_CODE"
    --default-mood "$DEFAULT_MOOD"
    --emo-strength "$EMO_STRENGTH"
    --speed "$SPEED"
    --num-beams "$NUM_BEAMS"
    --prebuffer-seconds "$PREBUFFER_SECONDS"
    --min-merged-seconds "$MIN_MERGED_SECONDS"
)
if [[ "$(declare -p EXTRA_ARGS 2>/dev/null)" == "declare -a"* ]]; then
    ARGS+=("${EXTRA_ARGS[@]}")
elif [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a _extra <<< "$EXTRA_ARGS"
    ARGS+=("${_extra[@]}")
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CMD=("$REPO_DIR/.venv/bin/python" bridge_indextts.py "${ARGS[@]}" "$@")
echo ">>> cmd: ${CMD[*]}"
exec "${CMD[@]}"
