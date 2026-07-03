"""
state_abstraction_full

Public API:

  Pipeline runner (processes one scenario folder):
    from state_abstraction_full.run_pipeline import main

  Batch runner (processes all completed scenarios):
    from state_abstraction_full.run_batch import main

  Core state builder:
    from state_abstraction_full.build_state import build_state, build_and_write

  Digital twin pod filter:
    from state_abstraction_full.digital_twin_filter import get_digital_twin_services

  LLM agent input builders:
    from state_abstraction_full.agent_input_builder import (
        build_rca_agent_input,
        build_prompt_optimizer_input,
        build_action_agent_input,
    )

  In-memory behavioral simulator (lightweight RCA verification):
    from state_abstraction_full.behavioral_simulator import AbstractBehavioralSimulator

  SLA evaluator:
    from state_abstraction_full.sla import evaluate_sla

  State compressor (LLM-ready compressed view):
    from state_abstraction_full.compress import compress_state
"""
