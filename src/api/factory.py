"""Explicit backend selection; model choice never migrates user memory."""
import os
from src.api.claude_client import LLMClient, MODEL_ID


def create_client(provider=None, *, notifier=None, claude_class=LLMClient):
    provider = (provider or os.getenv("TOFU_PROVIDER", "claude")).strip().lower()
    if provider == "offline":
        # An explicit offline choice must not accidentally read an ambient key.
        return claude_class(api_key="", notifier=notifier)
    if provider == "claude":
        options = {"notifier": notifier}
        model = os.getenv("TOFU_MODEL")
        if model:
            options["model"] = model
        return claude_class(**options)
    if provider in {"openai", "openai-compatible"}:
        from src.api.openai_client import OpenAIClient
        return OpenAIClient(notifier=notifier)
    raise ValueError("TOFU_PROVIDER 必須是 claude、openai、openai-compatible 或 offline")
