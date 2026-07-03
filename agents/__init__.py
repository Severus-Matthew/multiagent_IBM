"""
agents/

All agent implementations for the autonomous fault-remediation pipeline.

Public API:

  LLM client (wraps Claude / GPT):
    from agents.llm_client import LLMClient

  RCA Agent (fixed LLM):
    from agents.rca_agent import RCAAgent

  Prompt Optimization Agent (Qwen2.5-Coder-7B, trainable):
    from agents.prompt_optimizer import PromptOptimizerAgent

  Action Agent (fixed LLM, outputs kubectl/helm commands):
    from agents.action_agent import ActionAgent

  Full pipeline orchestration loop:
    from agents.pipeline import Pipeline, PipelineConfig, ScenarioResult

Typical setup:
    from agents.llm_client import LLMClient
    from agents.rca_agent import RCAAgent
    from agents.prompt_optimizer import PromptOptimizerAgent
    from agents.action_agent import ActionAgent
    from agents.pipeline import Pipeline
    from training.grpo_trainer import GRPOTrainer, GRPOConfig

    rca       = RCAAgent()
    optimizer = PromptOptimizerAgent()
    optimizer.load()
    action    = ActionAgent()
    trainer   = GRPOTrainer(optimizer.get_model(), optimizer.get_tokenizer())
    pipeline  = Pipeline(rca, optimizer, action, trainer=trainer)

    result = pipeline.run_scenario(state, compressed)
"""
