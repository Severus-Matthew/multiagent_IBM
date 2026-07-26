from __future__ import annotations

# Node features produced by graph_state_encoder.encode_state_graph().
# Keep this list append-only when possible so saved checkpoints and W&B plots stay interpretable.
NODE_FEATURE_NAMES = [
    "bias",
    "system_present",
    "infra_issue_flag",
    "pods_unready_norm",
    "restart_count_norm",
    "crashloop_count_norm",
    "no_ready_endpoint_flag",
    "service_health_degraded_flag",
    "top_log_error_flag",
    "top_log_error_rank_score",
    "incoming_failed_trace_norm",
    "outgoing_failed_trace_norm",
    "max_incoming_error_ratio",
    "max_outgoing_error_ratio",
    "metric_latency_high_flag",
    "metric_latency_norm",
    "cluster_infra_flag",
    "cluster_log_or_dependency_flag",
    "previously_predicted_flag",
    "previous_failure_flag",
    "previous_success_flag",
    "previous_reward_mean",
]


EDGE_FEATURE_NAMES = [
    "bias",
    "static_graph_edge_flag",
    "trace_edge_flag",
    "failed_trace_edge_flag",
    "error_ratio",
    "latency_norm",
]


EVIDENCE_ORDERS = [
    ["system_health", "service_health", "logs", "traces", "metrics"],
    ["traces", "service_graph", "logs", "system_health", "metrics"],
    ["logs", "database_dependency_logs", "traces", "system_health", "metrics"],
    ["metrics", "system_health", "logs", "traces", "service_graph"],
]


EVIDENCE_ORDER_NAMES = [
    "system_first",
    "trace_first",
    "log_first",
    "metric_first",
]


RCA_OPERATOR_NAMES = [
    "ENFORCE_OUTPUT_SCHEMA",
    "PRIORITIZE_SYSTEM_HEALTH",
    "PRIORITIZE_TRACE_EDGES",
    "PRIORITIZE_LOG_ERRORS",
    "PRIORITIZE_METRICS",
    "FOCUS_ON_K8S_INFRA",
    "FOCUS_ON_TRACE_TARGETS",
    "FOCUS_ON_DATABASE_DEPENDENCIES",
    "ASK_FOR_MULTIFAULT",
    "USE_SMALLEST_EXPLANATORY_SET",
    "AVOID_DOWNSTREAM_VICTIMS",
    "AVOID_REPEATED_GUESSES",
]


CANONICAL_FAULT_TYPES = [
    "infra_failure",
    "auth_failure",
    "dependency_failure",
    "resource_exhaustion",
    "latency_degradation",
    "network_failure",
    "config_error",
    "unknown",
]
