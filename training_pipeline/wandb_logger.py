from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_WANDB_ENTITY = "drprofmjha-university-of-illinois-urbana-champaign"
DEFAULT_WANDB_PROJECT = "aiops-rl"


class WandbRunLogger:
    """Optional Weights & Biases logging for rollout generation.

    Design rules:
    - W&B is optional. If wandb is not installed or login is missing, local JSONL
      logging still remains the source of truth.
    - Every local output file is uploaded as an artifact at the end, so large
      data remains inspectable even when scalar dashboards are summarized.
    - Per-episode scalar metrics are logged during the run to produce plots.
    """

    def __init__(
        self,
        enabled: bool,
        project: str = DEFAULT_WANDB_PROJECT,
        entity: str | None = DEFAULT_WANDB_ENTITY,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        self.enabled = bool(enabled)
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config or {}
        self.tags = tags or []
        self._wandb = None
        self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except Exception as e:
            print(f"[W&B] disabled: could not import wandb ({e}). Install with: pip install wandb")
            self.enabled = False
            return

        try:
            self._wandb = wandb
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=self.config,
                tags=self.tags,
            )
            print(f"[W&B] logging enabled: project={self.project} entity={self.entity} run={self.run_name}")
        except Exception as e:
            print(f"[W&B] disabled: wandb.init failed ({e})")
            self.enabled = False
            self._run = None

    def log_episode(
        self,
        episode_index: int,
        result: dict[str, Any],
        samples: list[dict[str, Any]],
        passed_so_far: int,
        total_so_far: int,
    ) -> None:
        if not self.active:
            return
        attempts = result.get("attempts", []) or []
        final_attempt = attempts[-1] if attempts else {}
        sample_rewards = [float(s.get("reward", 0.0)) for s in samples]
        reward_components = final_attempt.get("reward_components", {}) or {}

        row = {
            "episode/index": episode_index,
            "episode/success": int(bool(result.get("success"))),
            "episode/success_rate_so_far": passed_so_far / max(total_so_far, 1),
            "episode/attempts": len(attempts),
            "episode/samples": len(samples),
            "episode/terminal_failure": int(bool(result.get("terminal"))),
            "episode/final_reward": float(final_attempt.get("reward", 0.0) or 0.0),
            "episode/sample_reward_mean": mean(sample_rewards) if sample_rewards else 0.0,
            "episode/sample_reward_std": pstdev(sample_rewards) if len(sample_rewards) > 1 else 0.0,
            "reward/pair_score": float(reward_components.get("pair_score", 0.0) or 0.0),
            "reward/twin_reproduction_score": float(reward_components.get("twin_reproduction_score", 0.0) or 0.0),
            "reward/count_mismatch": float(reward_components.get("count_mismatch", 0.0) or 0.0),
            "reward/invalid_format": int(bool(reward_components.get("invalid_format", False))),
            "reward/repeated_wrong_guess": int(bool(reward_components.get("repeated_wrong_guess", False))),
        }
        self._wandb.log(row, step=episode_index)

    def log_summary(self, summary: dict[str, Any], output_dir: str | Path) -> None:
        if not self.active:
            return
        output_path = Path(output_dir).expanduser()
        self._wandb.summary.update(summary)

        artifact = self._wandb.Artifact(
            name=f"{self._run.name or 'run'}-rollout-files",
            type="rollout",
            metadata=summary,
        )
        for fname in ["summary.json", "rollouts.jsonl", "grpo_samples.jsonl", "reward_audit.json"]:
            path = output_path / fname
            if path.exists():
                artifact.add_file(str(path), name=fname)
        self._run.log_artifact(artifact)

        # Also store a compact text summary for quick W&B preview.
        text_path = output_path / "wandb_text_summary.md"
        try:
            text_path.write_text(_summary_markdown(summary), encoding="utf-8")
            text_artifact = self._wandb.Artifact(
                name=f"{self._run.name or 'run'}-text-summary",
                type="report",
                metadata={"source": "training_pipeline.wandb_logger"},
            )
            text_artifact.add_file(str(text_path), name="wandb_text_summary.md")
            self._run.log_artifact(text_artifact)
        except Exception as e:
            print(f"[W&B] warning: could not write text summary artifact ({e})")

    def finish(self) -> None:
        if self.active:
            self._wandb.finish()
            self._run = None


def parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# RCA rollout summary",
            "",
            "```json",
            json.dumps(summary, indent=2, sort_keys=True, default=str),
            "```",
        ]
    )
