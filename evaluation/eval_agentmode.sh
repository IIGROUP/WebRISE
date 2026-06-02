#!/usr/bin/env bash
# Minimal example wrapper for running WebRISE agent-mode evaluation on one HTML
# artifact and one Interaction Contract Graph.
#
# Usage:
#   ./eval_agentmode.sh HTML_PATH ICG_PATH [OUTPUT_DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="${ENV_FILE:-$RELEASE_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
fi

export WEB_EVAL_API_KEY="${WEB_EVAL_API_KEY:-${OPENAI_API_KEY:-}}"
export WEB_EVAL_BASE_URL="${WEB_EVAL_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
export WEB_EVAL_MODEL_AGENT="${WEB_EVAL_MODEL_AGENT:-}"
export WEB_EVAL_MODEL_SCORER="${WEB_EVAL_MODEL_SCORER:-}"
export WEB_EVAL_REASONING_EFFORT="${WEB_EVAL_REASONING_EFFORT:-}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 HTML_PATH ICG_PATH [OUTPUT_DIR]" >&2
    exit 2
fi

HTML_PATH="$1"
ICG_PATH="$2"
OUTPUT_DIR="${3:-${OUTPUT_DIR:-$RELEASE_ROOT/eval_results}}"

python3 -u "$SCRIPT_DIR/eval_agentmode.py" \
    --html "$HTML_PATH" \
    --icg "$ICG_PATH" \
    --output "$OUTPUT_DIR" \
    --max-iter "${MAX_ITER:-15}" \
    --dom-settle-ms "${DOM_SETTLE_MS:-8000}" \
    --agent-timeout-s "${AGENT_TIMEOUT_S:-300}" \
    --action-timeout-s "${ACTION_TIMEOUT_S:-30}" \
    --record-operations
