from pathlib import Path

# These are defaults. Override using CLI args in run_pipeline.py or build_state.py.
RUN_DIR = Path(".")
OUTPUT_DIR = RUN_DIR / "processed_state"

STATE_JSON = OUTPUT_DIR / "state_abstraction.json"
STATE_JSONL = OUTPUT_DIR / "states.jsonl"
STATE_SUMMARY_JSON = OUTPUT_DIR / "state_summary.json"
GRAPH_JSON = OUTPUT_DIR / "graph.json"
SLA_RESULTS_JSON = OUTPUT_DIR / "sla_results.json"

SIMULATOR_DIR = OUTPUT_DIR / "simulator_inputs"
SIMULATION_SPEC_JSON = SIMULATOR_DIR / "simulation_spec.json"
TOPOLOGY_JSON = SIMULATOR_DIR / "topology.json"
FAULT_LIBRARY_JSON = SIMULATOR_DIR / "fault_library.json"
ACTION_LIBRARY_JSON = SIMULATOR_DIR / "action_library.json"
OBSERVATION_MODEL_JSON = SIMULATOR_DIR / "observation_model.json"
SIMULATOR_SPEC_JSON = SIMULATOR_DIR / "simulator_spec.json"
COMPLEXITY_REPORT_JSON = SIMULATOR_DIR / "complexity_report.json"

SIMULATOR_RUNTIME_DIR = OUTPUT_DIR / "simulator_runtime_outputs"
SIMULATED_STATE_JSON = SIMULATOR_RUNTIME_DIR / "simulated_state.json"
RCA_VALIDATION_JSON = SIMULATOR_RUNTIME_DIR / "rca_validation.json"
MITIGATION_RESULT_JSON = SIMULATOR_RUNTIME_DIR / "mitigation_result.json"

LOG_KEYWORDS = {
    "error": ["error", "exception", "fatal", "traceback", "failed", "failure", "unauthorized", "authentication", "auth", "permission denied"],
    "warn": ["warn", "warning", "deprecated"],
    "timeout": ["timeout", "timed out", "deadline exceeded", "context deadline"],
    "connection": ["connection refused", "connection reset", "broken pipe", "unavailable", "connection error", "network is unreachable"],
    "config": ["config", "misconfig", "invalid configuration", "bad config", "missing", "not configured"],
    "dns": ["dns", "name resolution", "no such host", "lookup"],
    "retry": ["retry", "retrying"],
}

DEFAULT_ACTIONS = ["restart_service", "scale_service", "rollback_config", "disable_edge", "wait"]
DEFAULT_FAULT_TYPES = ["dependency_failure", "auth_failure", "latency_degradation", "config_error", "infra_failure", "ghost_failure"]
