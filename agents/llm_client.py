"""
agents/llm_client.py

Thin synchronous wrapper around Claude (Anthropic) and GPT (OpenAI) APIs.
Both RCA Agent and Action Agent use this for inference.

Usage:
    from agents.llm_client import LLMClient

    client = LLMClient(provider="claude")          # default: claude-sonnet-4-6
    client = LLMClient(provider="openai")          # default: gpt-4o
    client = LLMClient(provider="claude", model="claude-opus-4-8")

    text = client.call(system_prompt, user_prompt)
    obj  = client.call_json(system_prompt, user_prompt)  # parses response as JSON
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any


class LLMClient:
    """
    Synchronous LLM client supporting Anthropic Claude and OpenAI GPT.

    API keys are read from environment variables:
      - ANTHROPIC_API_KEY  (for provider="claude")
      - OPENAI_API_KEY     (for provider="openai")
    """

    DEFAULTS = {
        "claude": "claude-sonnet-4-6",
        "openai": "gpt-4o",
    }

    def __init__(self, provider: str = "claude", model: str | None = None):
        if provider not in self.DEFAULTS:
            raise ValueError(f"provider must be 'claude' or 'openai', got '{provider}'")
        self.provider = provider
        self.model = model or self.DEFAULTS[provider]

    # ── Public API ────────────────────────────────────────────────────────────

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> str:
        """
        Send a chat request and return the response text.

        Args:
            system_prompt: system / persona instructions
            user_prompt:   the user turn (task description + data)
            max_tokens:    maximum response length
            temperature:   0.0 = deterministic, higher = more varied

        Returns:
            Raw response string from the model.
        """
        if self.provider == "claude":
            return self._call_claude(system_prompt, user_prompt, max_tokens, temperature)
        else:
            return self._call_openai(system_prompt, user_prompt, max_tokens, temperature)

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.0,
        retries: int = 2,
    ) -> Any:
        """
        Call the LLM and parse the response as JSON.
        Retries up to `retries` times if JSON extraction fails.
        Raises ValueError if all retries fail.
        """
        for attempt in range(retries + 1):
            raw = self.call(system_prompt, user_prompt, max_tokens, temperature)
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
            if attempt < retries:
                time.sleep(1)

        raise ValueError(
            f"LLM did not return valid JSON after {retries + 1} attempts.\n"
            f"Last response:\n{raw[:600]}"
        )

    # ── Provider implementations ──────────────────────────────────────────────

    def _call_claude(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return response.content[0].text

    def _call_openai(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


# ── JSON extraction helper ────────────────────────────────────────────────────


def _extract_json(text: str) -> Any | None:
    """
    Extract JSON from an LLM response that may contain markdown code fences,
    prose, or plain JSON.

    Tries in order:
      1. Direct parse of the entire response
      2. Extract from ```json ... ``` fence
      3. Extract from first { ... } or [ ... ] block
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. json code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. First {...} or [...] block
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    return None
