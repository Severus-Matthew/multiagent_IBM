from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RolloutLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "rollouts.jsonl"
        self.grpo_jsonl_path = self.output_dir / "grpo_samples.jsonl"

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def log(self, row: dict[str, Any]) -> None:
        self._append_jsonl(self.jsonl_path, row)

    def log_grpo_sample(self, row: dict[str, Any]) -> None:
        self._append_jsonl(self.grpo_jsonl_path, row)

    def write_summary(self, summary: dict[str, Any]) -> None:
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True, default=str)
