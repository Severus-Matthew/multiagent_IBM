"""
digital_twin/

K8s digital twin builder and solution executor.

Public API:

  Core twin lifecycle:
    from digital_twin.builder import DigitalTwin

  Execute solution commands and score them:
    from digital_twin.executor import SolutionExecutor, ExecutionResult

  Reward computation (also called internally by SolutionExecutor):
    from digital_twin.reward import compute_reward, reward_summary

Typical usage in the training pipeline:
    twin = DigitalTwin(state, compressed)
    twin.build()              # deploy full stack, trim bystanders
    twin.inject_fault()       # replicate scenario fault

    for iteration in range(max_iterations):
        commands = action_agent.get_commands(...)
        executor = SolutionExecutor(twin)
        result = executor.run(commands)
        # result.reward, result.passed, result.reward_breakdown
        if result.passed:
            break
        twin.reset_fault()    # recover + re-inject for next iteration

    twin.teardown()           # uninstall helm release
"""
