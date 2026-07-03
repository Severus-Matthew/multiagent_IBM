"""
training/

GRPO training for the Prompt Optimization Agent (Qwen2.5-Coder-7B).

Public API:

  GRPO trainer + config:
    from training.grpo_trainer import GRPOTrainer, GRPOConfig

Usage:
    from agents.prompt_optimizer import PromptOptimizerAgent
    from training.grpo_trainer import GRPOTrainer, GRPOConfig

    agent = PromptOptimizerAgent()
    agent.load()

    config  = GRPOConfig(learning_rate=1e-5, kl_coeff=0.01, min_batch_size=7)
    trainer = GRPOTrainer(agent.get_model(), agent.get_tokenizer(), config)

    # Per scenario — accumulate samples
    trainer.add_sample(prompt_text, completion_text, reward)

    # After scenario — do one update
    if trainer.should_update():
        metrics = trainer.step()

    # Save checkpoint
    trainer.save("checkpoints/step_100")
"""
