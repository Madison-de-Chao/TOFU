"""OpenAI Chat Completions compatible transport for the shared Tofu workflow.

Modified by Codex for whitepaper §3.4. Prompt construction, JSON parsing and
fallbacks are inherited; only transport changes. No Anthropic client is created.
This adapter does not implement Anthropic's provider-specific Batch API.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener

from src.api.claude_client import LLMClient, LLMClientError, DEFAULT_MAX_TOKENS


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward an Authorization header to a redirected service.
        return None


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, notifier: Callable | None = None,
                 timeout: float = 30, token_parameter: str | None = None):
        self._key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self._model = model or os.getenv("TOFU_MODEL") or os.getenv("OPENAI_MODEL", "")
        base = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        parsed = urlsplit(base)
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query
                or parsed.fragment or parsed.scheme not in {"http", "https"}):
            raise ValueError("OPENAI_BASE_URL 必須是無帳密、query、fragment 的 HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("非本機模型端點必須使用 HTTPS")
        self._url = base + "/chat/completions"
        self._timeout = float(timeout)
        if self._timeout <= 0 or self._timeout > 120:
            raise ValueError("timeout 必須介於 0 與 120 秒之間")
        self._token_parameter = token_parameter or os.getenv("TOFU_OPENAI_TOKEN_PARAMETER", "max_completion_tokens")
        if self._token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("token_parameter 必須是 max_tokens 或 max_completion_tokens")
        self._notifier = notifier
        self._client = None
        self._opener = build_opener(_NoRedirect())
        self.fallback_mode = not bool(self._key)
        if not self.fallback_mode and not self._model:
            raise ValueError("使用 OpenAI 相容後端時請明確設定 TOFU_MODEL")
        self.last_usage: dict = {}

    def _call(self, system: str, user: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        if self.fallback_mode:
            raise LLMClientError("離線模式不能呼叫 API")
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self._token_parameter: max_tokens,
            "stream": False,
        }, ensure_ascii=False).encode("utf-8")
        request = Request(self._url, data=body, headers={
            "Authorization": "Bearer " + self._key, "Content-Type": "application/json",
        }, method="POST")
        for attempt in range(3):
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise LLMClientError("模型回應超過 4 MiB 限制")
                data = json.loads(raw)
                message = data["choices"][0]["message"]
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise LLMClientError("模型未回傳文字；可能拒絕、截斷或只回傳工具呼叫")
                self.last_usage = data.get("usage") or {}
                return content.strip()
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == 2:
                    # Do not echo response bodies, headers, credentials or user data.
                    raise LLMClientError(f"OpenAI 相容 API 回傳 HTTP {exc.code}") from None
                wait = min(2 ** attempt, 10)
                try:
                    wait = min(max(float(exc.headers.get("Retry-After", wait)), 0), 10)
                except (TypeError, ValueError):
                    pass
            except (URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise LLMClientError("OpenAI 相容 API 連線失敗或逾時（3 次嘗試）") from None
                wait = 2 ** attempt
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                raise LLMClientError("OpenAI 相容 API 回傳格式不符 Chat Completions 規格") from None
            self._notify(f"[逗福Tofu] API 暫時不可用，{wait:g} 秒後重試。")
            time.sleep(wait)
        raise LLMClientError("OpenAI 相容 API 未取得回應")

    def submit_batch(self, requests):
        raise LLMClientError("此後端不支援 Anthropic Batch API；請使用一般互動流程")

    def poll_batch(self, batch_id):
        raise LLMClientError("此後端不支援 Anthropic Batch API")

    def get_batch_results(self, batch_id):
        raise LLMClientError("此後端不支援 Anthropic Batch API")
