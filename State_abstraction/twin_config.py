from pathlib import Path

# Change this to your real processed_state folder
PROCESSED_DIR = Path("/Users/manvi/Downloads/multiagent_IBM/State_abstraction/processed_state")

STATES_JSONL = PROCESSED_DIR / "states.jsonl"
GRAPH_JSON = PROCESSED_DIR / "graph.json"

TWIN_DIR = PROCESSED_DIR / "twin_inputs"

COMPLEXITY_REPORT_JSON = TWIN_DIR / "complexity_report.json"
TOPOLOGY_JSON = TWIN_DIR / "topology.json"
FAULT_LIBRARY_JSON = TWIN_DIR / "fault_library.json"
ACTION_LIBRARY_JSON = TWIN_DIR / "action_library.json"
OBSERVATION_MODEL_JSON = TWIN_DIR / "observation_model.json"
TWIN_SPEC_JSON = TWIN_DIR / "twin_spec.json"

DEFAULT_ACTIONS = [
    "restart_service",
    "scale_service",
    "rollback_config",
    "disable_edge",
    "wait",
]

DEFAULT_FAULT_TYPES = [
    "dependency_failure",
    "latency_degradation",
    "config_error",
    "infra_failure",
    "ghost_failure",
]