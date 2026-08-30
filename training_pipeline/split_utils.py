from __future__ import annotations

from pathlib import Path
from typing import Iterable


def read_scenario_ids(path: str | Path | None) -> set[str] | None:
    """Read one scenario_id per line. Blank lines and # comments are ignored."""
    if not path:
        return None
    p = Path(path).expanduser()
    ids: set[str] = set()
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(line)
    return ids


def write_scenario_ids(path: str | Path, scenario_ids: Iterable[str]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    unique_sorted = sorted(set(str(x) for x in scenario_ids if str(x).strip()))
    p.write_text("\n".join(unique_sorted) + ("\n" if unique_sorted else ""))
