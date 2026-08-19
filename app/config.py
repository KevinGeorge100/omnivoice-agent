"""Safe runtime configuration status for OmniVoice services."""

import os

from dotenv import load_dotenv


_PLACEHOLDER_PREFIXES = ("your_", "gsk_your_", "replace_", "change_me")
_SERVICE_ENVIRONMENT_KEYS = {
    "groq": "GROQ_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
    "sarvam": "SARVAM_API_KEY",
}


def is_configured(environment_key: str) -> bool:
    """Return whether a non-placeholder secret is available without exposing it."""
    load_dotenv()
    value = os.environ.get(environment_key, "").strip()
    return bool(value) and not value.lower().startswith(_PLACEHOLDER_PREFIXES)


def provider_status() -> dict[str, bool]:
    """Return public-safe provider readiness indicators for operational endpoints."""
    return {
        name: is_configured(environment_key)
        for name, environment_key in _SERVICE_ENVIRONMENT_KEYS.items()
    }
