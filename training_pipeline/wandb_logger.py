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
        run_id: str | None = None,
    ):
        self.enabled = bool(enabled)
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config or {}
        self.tags = tags or []
        # When set, start() resumes this exact W&B run instead of creating a new
        # one — required for a checkpoint --resume to continue the same run's
        # history rather than fragmenting it across multiple run IDs. There is
        # no way to merge two already-created runs after the fact (confirmed
        # against W&B's own docs/issue tracker), so this must be set *before*
        # wandb.init() is ever called for a given training lineage.
        self.run_id = run_id
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
            init_kwargs: dict[str, Any] = dict(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=self.config,
                tags=self.tags,
            )
            if self.run_id:
                # resume="must" fails loudly if the run_id turns out not to
                # exist, rather than silently starting a fresh run under that
                # id — exactly the failure mode that produced this fragmentation
                # in the first place.
                init_kwargs["id"] = self.run_id
                init_kwargs["resume"] = "must"
            self._run = wandb.init(**init_kwargs)
            self.run_id = self._run.id
            print(
                f"[W&B] logging enabled: project={self.project} entity={self.entity} "
                f"run={self.run_name} run_id={self.run_id}"
            )
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
        for fname in [
            "summary.json", "run_manifest.json", "training_events.jsonl",
            "joint_trajectories.jsonl", "rca_policy_samples.jsonl",
            "action_policy_samples.jsonl", "openai_calls_worker_0.jsonl",
            "openai_calls_worker_1.jsonl",
        ]:
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

    def _log_role(self, row: dict[str, Any], prefix: str, role: dict[str, Any]) -> None:
        """Surface every diagnostic the synchronized trainer already computes.

        ``role`` is the per-role dict returned by
        ``StreamingSynchronizedFactorizedGRPOTrainer._update_role``: ``signal``
        carries the group-relative-advantage zero-signal gate, ``optimizer``
        carries the token-level PPO ratio/clip/KL/grad-norm diagnostics. Both
        are computed unconditionally; only the update itself is skipped when
        ``has_policy_gradient_signal`` is False.
        """
        row[f"train/{prefix}_updated"] = int(bool(role.get("updated")))
        skip_reason = role.get("skip_reason")
        # Clear a previous skip reason when the role updates again; W&B summary
        # otherwise retains the last nonempty value from an older update.
        row[f"train/{prefix}_skip_reason"] = str(skip_reason or "")
        signal = role.get("signal", {}) or {}
        for key in (
            "nonzero_advantage_groups", "zero_advantage_groups",
            "nonzero_advantage_trajectories",
        ):
            if key in signal:
                row[f"train/{prefix}_{key}"] = int(signal[key])
        if "has_policy_gradient_signal" in signal:
            row[f"train/{prefix}_has_policy_gradient_signal"] = int(bool(signal["has_policy_gradient_signal"]))
        optimizer = role.get("optimizer", {}) or {}
        for key, cast in (
            ("loss", float), ("grad_norm_before_clip", float),
            ("mean_clip_fraction", float), ("mean_ratio", float),
            ("mean_sampled_kl", float), ("ratio_min", float), ("ratio_max", float),
            ("num_rows", int), ("num_completion_tokens", float),
        ):
            if key in optimizer and optimizer[key] is not None:
                row[f"train/{prefix}_{key}"] = cast(optimizer[key])

    def log_training_update(self, step: int, update: dict[str, Any]) -> None:
        if not self.active:
            return
        rca = update.get("rca", {}) or {}
        action = update.get("action", {}) or {}
        row = {
            "train/bundle_update": int(update.get("bundle_update_step", step) or step),
            "train/scenarios_completed": int(update.get("scenarios_completed", 0) or 0),
            "train/scenarios_in_update": int(update.get("scenarios_in_update", 0) or 0),
            "train/epoch_index": int(update.get("epoch_index", 0) or 0),
            "train/scenario_cursor": int(update.get("scenario_cursor", 0) or 0),
            "train/twin_mode": str(update.get("twin_mode") or ""),
            "train/policy_version": str(update.get("published_policy_version") or update.get("policy_version") or ""),
            "train/rollout_policy_version": str(update.get("rollout_policy_version") or ""),
            "train/replica_adapter_tensors_copied": int(update.get("replica_adapter_tensors_copied", 0) or 0),
            "train/parallel_rollout_workers": int(update.get("parallel_rollout_workers", 0) or 0),
        }
        self._log_role(row, "rca", rca)
        self._log_role(row, "action", action)

        # Rollout/reward-route summary: computed once per batch in
        # _reward_route_summary and merged flat into `update`; surface it as-is
        # rather than re-deriving it here.
        for key, cast, wandb_key in (
            ("live_trajectory_count", int, "rollout/live_trajectory_count"),
            ("offline_trajectory_count", int, "rollout/offline_trajectory_count"),
            ("action_stage_invoked_count", int, "rollout/action_stage_invoked_count"),
            ("skipped_action_count", int, "rollout/skipped_action_count"),
            ("full_success_count", int, "rollout/full_success_count"),
            ("observable_improvement_count", int, "rollout/observable_improvement_count"),
            ("generation_examples_written", int, "rollout/generation_examples_written"),
        ):
            value = update.get(key)
            if value is not None:
                row[wandb_key] = cast(value)
        # Counts above are per trajectory. Each scenario can produce a group
        # of several trajectories, including unscorable/unknown routes.
        # Legacy update records can recover the total from all route counts.
        trajectory_count = update.get("trajectory_count")
        if trajectory_count is None and "reward_route_counts" in update:
            trajectory_count = sum(int(n) for n in (update["reward_route_counts"] or {}).values())
        if trajectory_count is not None:
            trajectory_count = int(trajectory_count)
            row["rollout/trajectory_count"] = trajectory_count
            if trajectory_count > 0:
                if "full_success_count" in update:
                    row["rollout/trajectory_success_rate"] = float(update["full_success_count"]) / trajectory_count
                if "skipped_action_count" in update:
                    row["rollout/skipped_action_rate"] = float(update["skipped_action_count"]) / trajectory_count
        for key, wandb_key in (
            ("live_twin_reproduction_mean", "twin/live_reproduction_score_mean"),
            ("offline_twin_reproduction_mean", "twin/offline_reproduction_score_mean"),
            ("rca_policy_return_mean", "train/rca_policy_return_mean"),
            ("action_policy_return_mean", "train/action_policy_return_mean"),
        ):
            value = update.get(key)
            if value is not None:
                row[wandb_key] = float(value)
        route_counts = update.get("reward_route_counts") or {}
        for route, count in route_counts.items():
            row[f"rollout/reward_route_count/{route}"] = int(count)

        # GPU memory: cheap, always-available signal for whether the current
        # config still fits (bounded-state/tail-logit/streaming-backward all
        # exist specifically to keep this under the device budget).
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    row[f"gpu/{index}_allocated_gib"] = torch.cuda.memory_allocated(index) / (1024 ** 3)
                    row[f"gpu/{index}_reserved_gib"] = torch.cuda.memory_reserved(index) / (1024 ** 3)
                    row[f"gpu/{index}_peak_allocated_gib"] = torch.cuda.max_memory_allocated(index) / (1024 ** 3)
        except Exception:
            pass

        # wandb defaults commit=False whenever an explicit step= is passed (only
        # step=None defaults to commit=True) — without this, a row sits as the
        # run's "pending" state and is only flushed once a LATER call advances
        # past it, or lost outright if the process dies first. This call is the
        # only log() per update, so there is no accumulation use case to defer.
        self._wandb.log(row, step=int(step), commit=True)

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
