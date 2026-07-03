"""
agents/action_agent.py

Action Agent — fixed LLM (Claude / GPT) that takes an instruction prompt from
the Prompt Optimization Agent and outputs concrete kubectl/helm commands
to remediate the fault.

The agent outputs ONLY commands — no prose, no explanations.
Commands are parsed line by line, validated for syntax, and returned as a list.

Usage:
    from agents.action_agent import ActionAgent
    from agents.llm_client import LLMClient

    agent = ActionAgent()                               # default: Claude
    agent = ActionAgent(client=LLMClient("openai"))     # GPT variant

    commands = agent.get_commands(instruction_prompt, opt_context)
    # → ["kubectl rollout restart deployment/user -n test-social-network", ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.llm_client import LLMClient
from state_abstraction_full.agent_input_builder import build_action_agent_input

_SYSTEM_PROMPT = """\
You are a Kubernetes operations expert performing fault remediation.

You will receive a detailed instruction prompt specifying exactly what fix to apply.
Your response must be ONLY the kubectl/helm commands to execute, one per line.

Rules:
- Output only executable commands (kubectl, helm, mongosh)
- No explanations, no markdown, no code fences, no comments
- One command per line
- Commands must be syntactically valid
- Use the exact namespace specified in the instructions
- For multi-step fixes, order commands correctly (scale down → patch → scale up)
- Maximum 15 commands total

Example of correct output:
kubectl scale deployment/url-shorten-service --replicas=0 -n test-social-network
kubectl patch configmap url-shorten-config -n test-social-network --type=merge -p '{"data":{"auth":"enabled"}}'
kubectl scale deployment/url-shorten-service --replicas=1 -n test-social-network
kubectl rollout status deployment/url-shorten-service -n test-social-network --timeout=90s\
"""

# Commands we consider valid action agent outputs
_VALID_PREFIXES = ("kubectl", "helm", "mongosh")
# Pattern to strip markdown code fences and leading/trailing decorations
_FENCE_RE = re.compile(r"^```[a-z]*\s*$|^```\s*$", re.MULTILINE)
_LEADING_DOLLAR = re.compile(r"^\$\s+")


class ActionAgent:
    """
    Action Agent that converts instruction prompts into executable commands.

    Uses a fixed LLM — weights are NOT updated during training.
    Only the Prompt Optimization Agent is trained.
    """

    def __init__(self, client: LLMClient | None = None, max_commands: int = 15):
        self.client = client or LLMClient(provider="claude")
        self.max_commands = max_commands

    def get_commands(
        self,
        instruction_prompt: str,
        context: dict,
        max_tokens: int = 800,
    ) -> list[str]:
        """
        Generate remediation commands from an instruction prompt.

        Args:
            instruction_prompt: output of PromptOptimizerAgent.generate()
            context:            dict from build_prompt_optimizer_input()
            max_tokens:         LLM response budget

        Returns:
            List of kubectl/helm/mongosh command strings, ready to execute.
        """
        full_prompt = build_action_agent_input(instruction_prompt, context)

        print(
            f"[ActionAgent] Generating commands  "
            f"model={self.client.model}  "
            f"iter={context.get('iteration', 0)}"
        )

        raw = self.client.call(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        commands = self._parse_commands(raw)
        print(f"[ActionAgent] Extracted {len(commands)} commands")
        for cmd in commands:
            print(f"  $ {cmd}")
        return commands

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_commands(self, raw: str) -> list[str]:
        """
        Extract clean command lines from the LLM response.

        Handles:
        - Markdown code fences (```bash ... ```)
        - Lines prefixed with $ or >
        - Inline comments (# ...)
        - Blank lines between commands
        """
        # Strip code fences
        text = _FENCE_RE.sub("", raw).strip()

        commands: list[str] = []
        for line in text.splitlines():
            # Strip leading $ or > prompt symbols
            line = _LEADING_DOLLAR.sub("", line.strip())
            line = line.lstrip("> ").strip()

            # Drop empty lines and comment-only lines
            if not line or line.startswith("#"):
                continue

            # Drop prose lines that don't start with a valid command
            if not any(line.startswith(p) for p in _VALID_PREFIXES):
                # Check if it's a multi-line kubectl exec continuation
                # (begins with -- or similar shell continuation)
                if not (line.startswith("--") or line.startswith("'")):
                    continue

            commands.append(line)

            if len(commands) >= self.max_commands:
                break

        return commands
