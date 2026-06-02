"""
WebRISE evaluation configuration.
"""

import os

# Configure credentials through environment variables.

API_KEY = ""
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

def get_credentials() -> tuple[str, str]:
    key = (
        os.environ.get("WEB_EVAL_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or API_KEY
    )
    base_url = (
        os.environ.get("WEB_EVAL_BASE_URL", "").strip()
        or os.environ.get("OPENAI_BASE_URL", "").strip()
        or BASE_URL
        or "https://api.openai.com/v1"
    )
    if not key:
        raise ValueError(
            "API key not found.\n"
            "Set WEB_EVAL_API_KEY or OPENAI_API_KEY in the environment."
        )
    return key, base_url


# Model used by the contract-guided browser agent.
MODEL_AGENT = os.environ.get("WEB_EVAL_MODEL_AGENT", "").strip()

# Model used for postcondition scoring (vision with before/after screenshots)
MODEL_SCORER  = os.environ.get("WEB_EVAL_MODEL_SCORER", "").strip()

# Reasoning effort for models that support the OpenAI-compatible parameter.
# Set WEB_EVAL_REASONING_EFFORT=off to omit it, or use low/medium/high/etc.
REASONING_EFFORT = os.environ.get("WEB_EVAL_REASONING_EFFORT", "").strip()
REASONING_EFFORT_MODEL_PREFIXES = tuple(
    p.strip().lower()
    for p in os.environ.get("WEB_EVAL_REASONING_EFFORT_MODELS", "").split(",")
    if p.strip()
)


def reasoning_effort_kwargs(model: str) -> dict:
    effort = REASONING_EFFORT
    if not effort or effort.lower() in {"0", "false", "no", "none", "off"}:
        return {}
    model_l = (model or "").lower()
    if any(model_l.startswith(prefix) for prefix in REASONING_EFFORT_MODEL_PREFIXES):
        return {"reasoning_effort": effort}
    return {}


BROWSER_HEADLESS      = True
VIEWPORT_WIDTH        = 1280
VIEWPORT_HEIGHT       = 900
PAGE_LOAD_TIMEOUT_MS  = 15_000   # max ms to wait for page load
ACTION_SETTLE_MS      = 800      # ms to wait after each action for UI to settle
AGENT_CALL_TIMEOUT_S  = int(os.environ.get("WEB_EVAL_AGENT_TIMEOUT_S", "300"))
ACTION_TIMEOUT_S      = int(os.environ.get("WEB_EVAL_ACTION_TIMEOUT_S", "30"))


SCORER_MAX_TOKENS     = 4096
