from pathlib import Path

# Change this to your real processed state directory
PROCESSED_DIR = Path("/Users/manvi/Downloads/multiagent_IBM/State_abstraction/processed_state")

STATES_JSONL = PROCESSED_DIR / "states.jsonl"
GRAPH_JSON = PROCESSED_DIR / "graph.json"

OUTPUT_DIR = PROCESSED_DIR / "twin_runtime_outputs"

SIMULATED_STATE_JSON = OUTPUT_DIR / "simulated_state.json"
RCA_VALIDATION_JSON = OUTPUT_DIR / "rca_validation.json"
MITIGATION_RESULT_JSON = OUTPUT_DIR / "mitigation_result.json"

# Twin thresholds
RCA_MATCH_HIGH = 0.75
RCA_MATCH_MEDIUM = 0.50

DEFAULT_FAILURE_TYPES = [
    "dependency_failure",
    "ghost_failure",
    "latency_degradation",
    "config_error",
    "infra_failure",
]