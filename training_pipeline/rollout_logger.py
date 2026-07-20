from __future__ import annotations

import json
from pathlib import Path
from typing import Any

class RolloutLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "rollouts.jsonl"

    def log(self, row: dict[str, Any]) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def write_summary(self, summary: dict[str, Any]) -> None:
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True, default=str)
