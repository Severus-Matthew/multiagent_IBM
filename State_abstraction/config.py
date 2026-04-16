from pathlib import Path

# Change this to your real run folder
RUN_DIR = Path("/Users/manvi/Downloads/multiagent_IBM/AIOpsLab/aiops_runs/misconfig_app_hotel_res-detection-1/run_20260415_203030/phase_before")

LOGS_DIR = RUN_DIR / "logs"
METRICS_DIR = RUN_DIR / "metrics"
SYSTEM_DIR = RUN_DIR / "system"
TRACES_DIR = RUN_DIR / "traces"

OUTPUT_DIR = RUN_DIR / "processed_state"
STATE_JSONL = OUTPUT_DIR / "states.jsonl"
STATE_SUMMARY_JSON = OUTPUT_DIR / "state_summary.json"
GRAPH_JSON = OUTPUT_DIR / "graph.json"
SNAPSHOT_INDEX_JSON = OUTPUT_DIR / "snapshot_index.json"

SERVICES = [
    "frontend",
    "profile",
    "recommendation",
    "reservation",
    "search",
    "user",
]

LOG_KEYWORDS = {
    "error": ["error", "exception", "fatal", "traceback", "failed"],
    "warn": ["warn", "warning"],
    "timeout": ["timeout", "timed out", "deadline exceeded"],
    "connection": ["connection refused", "connection reset", "broken pipe", "unavailable"],
    "config": ["config", "misconfig", "invalid configuration", "bad config"],
    "dns": ["dns", "name resolution", "no such host"],
    "retry": ["retry", "retrying"],
}

# If you later add action logs, you can fill history from those.
DEFAULT_HISTORY = {
    "previous_actions": [],
    "last_action": None,
    "action_count": 0,
}