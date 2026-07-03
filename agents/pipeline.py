"""
agents/pipeline.py

Full autonomous fault-remediation pipeline.

Flow per scenario
-----------------
1. RCA Agent     → identifies root_cause_service + fault_type
2. DigitalTwin   → build (deploy + trim bystanders), inject fault
3. RCAVerifier   → check twin shows same symptoms as original (non-blocking)
4. For iteration in range(max_iterations):
     a. PromptOptimizer → generate instruction prompt (Qwen2.5-Coder-7B)
     b. ActionAgent     → convert instruction to kubectl/helm commands
     c. SolutionExecutor → execute on twin, compute reward
     d. If passed: break
     e. twin.reset_fault() → recover + re-inject for next iteration
5. GRPO update   → update PromptOptimizer weights with (prompt, completion, reward)
6. Return best solution; teardown twin

Usage:
    from agents.pipeline import Pipeline, PipelineConfig
    from agents.rca_agent import RCAAgent
    from agents.prompt_optimizer import PromptOptimizerAgent
    from agents.action_agent import ActionAgent
    from training.grpo_trainer import GRPOTrainer, GRPOConfig

    # Build agents
    rca       = RCAAgent()
    optimizer = PromptOptimizerAgent()
    optimizer.load()
    action    = ActionAgent()
    trainer   = GRPOTrainer(optimizer.get_model(), optimizer.get_tokenizer())

    pipeline  = Pipeline(rca, optimizer, action, trainer=trainer)

    # Run one scenario
    result = pipeline.run_scenario(state, compressed)
    print(result.passed, result.best_reward, result.best_commands)

    # Run a batch
    results = pipeline.run_batch(scenario_list, save_dir="results/")
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "AIOpsLab") not in sys.path:
    sys.path.insert(0, str(_ROOT / "AIOpsLab"))

from state_abstraction_full.agent_input_builder import (
    build_prompt_optimizer_input,
)

if TYPE_CHECKING:
    from agents.rca_agent import RCAAgent
    from agents.prompt_optimizer import PromptOptimizerAgent
    from agents.action_agent import ActionAgent
    from training.grpo_trainer import GRPOTrainer


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    max_iterations: int = 7          # iterations per scenario before giving up
    rca_verify: bool = True          # run RCA verifier (non-blocking — logs only)
    teardown_on_pass: bool = True    # uninstall twin after a passing scenario
    teardown_on_fail: bool = True    # uninstall twin even after failing scenarios
    save_results: bool = True        # write per-scenario JSON result files
    grpo_update: bool = True         # update PromptOptimizer after each scenario
    build_twin: bool = True          # actually build the K8s twin (set False for dry-run)
    stabilize_seconds: int = 15      # seconds to wait between reset and re-inject


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class IterationRecord:
    iteration: int
    instruction_prompt: str
    commands: list[str]
    reward: float
    passed: bool
    elapsed_seconds: float
    reward_breakdown: dict = field(default_factory=dict)
    command_results: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "commands": self.commands,
            "reward": round(self.reward, 4),
            "passed": self.passed,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "reward_breakdown": self.reward_breakdown,
            "command_results": self.command_results,
        }


@dataclass
class ScenarioResult:
    scenario_id: str
    rca_result: dict
    rca_verified: bool
    passed: bool
    best_reward: float
    best_commands: list[str]
    iterations: list[IterationRecord]
    elapsed_seconds: float
    error: str = ""

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "best_reward": round(self.best_reward, 4),
            "n_iterations": self.n_iterations,
            "rca_verified": self.rca_verified,
            "rca_result": self.rca_result,
            "best_commands": self.best_commands,
            "iterations": [it.as_dict() for it in self.iterations],
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error": self.error,
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────


class Pipeline:
    """
    Orchestrates the full RCA → DigitalTwin → PromptOptimizer → Action → Verify
    training loop for autonomous fault remediation.
    """

    def __init__(
        self,
        rca_agent: "RCAAgent",
        prompt_optimizer: "PromptOptimizerAgent",
        action_agent: "ActionAgent",
        trainer: "GRPOTrainer | None" = None,
        config: PipelineConfig | None = None,
    ):
        self.rca_agent = rca_agent
        self.prompt_optimizer = prompt_optimizer
        self.action_agent = action_agent
        self.trainer = trainer
        self.config = config or PipelineConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_scenario(
        self,
        state: dict,
        compressed: dict | None = None,
        save_dir: str | None = None,
    ) -> ScenarioResult:
        """
        Run the full pipeline for one scenario.

        Args:
            state:      state_abstraction.json dict
            compressed: state_abstraction_compressed.json dict
            save_dir:   if set, write result JSON here

        Returns:
            ScenarioResult with reward, commands, and GRPO training data
        """
        t0 = time.time()
        scenario_id = state.get("scenario_id", "unknown")
        print(f"\n{'='*60}")
        print(f"[Pipeline] Starting scenario: {scenario_id}")
        print(f"{'='*60}")

        # ── Step 1: RCA ───────────────────────────────────────────────────────
        try:
            rca_result = self.rca_agent.analyze(state, compressed)
        except Exception as e:
            return self._error_result(scenario_id, {}, str(e), time.time() - t0)

        # ── Step 2: Build digital twin ────────────────────────────────────────
        twin = None
        if self.config.build_twin:
            try:
                from digital_twin.builder import DigitalTwin
                twin = DigitalTwin(state, compressed)
                twin.build()
            except Exception as e:
                return self._error_result(scenario_id, rca_result, str(e), time.time() - t0)

        # ── Step 3: Inject fault ──────────────────────────────────────────────
        if twin is not None:
            try:
                twin.inject_fault()
            except Exception as e:
                twin_teardown(twin, force=True)
                return self._error_result(scenario_id, rca_result, str(e), time.time() - t0)

        # ── Step 4: RCA verification (non-blocking) ───────────────────────────
        rca_verified = False
        if twin is not None and self.config.rca_verify:
            try:
                from digital_twin.verifier import RCAVerifier
                ver = RCAVerifier()
                v_result = ver.verify(state, twin, rca_result)
                rca_verified = v_result.verified
                if not rca_verified:
                    print(
                        f"[Pipeline] RCA verification failed "
                        f"(match_score={v_result.match_score:.2f}) — continuing anyway"
                    )
            except Exception as e:
                print(f"[Pipeline] RCA verification error (non-fatal): {e}")

        # ── Step 5: Iteration loop ────────────────────────────────────────────
        iterations: list[IterationRecord] = []
        grpo_samples: list[dict] = []
        best_reward = 0.0
        best_commands: list[str] = []
        passed = False
        reward_history: list[dict] = []

        for iteration in range(self.config.max_iterations):
            iter_result = self._run_iteration(
                state=state,
                compressed=compressed,
                rca_result=rca_result,
                twin=twin,
                iteration=iteration,
                reward_history=reward_history,
                grpo_samples=grpo_samples,
            )
            iterations.append(iter_result)
            reward_history.append({
                "iteration": iteration,
                "commands": iter_result.commands,
                "result": "PASS" if iter_result.passed else "FAIL",
                "reward": iter_result.reward,
                "failure_reason": _failure_reason(iter_result),
            })

            if iter_result.reward > best_reward:
                best_reward = iter_result.reward
                best_commands = iter_result.commands

            if iter_result.passed:
                passed = True
                print(f"[Pipeline] PASSED on iteration {iteration + 1}")
                break

            # Reset twin for next iteration
            if twin is not None and iteration < self.config.max_iterations - 1:
                try:
                    twin.reset_fault(
                        stabilize_seconds=self.config.stabilize_seconds
                    )
                except Exception as e:
                    print(f"[Pipeline] reset_fault error: {e}")
                    break

        print(
            f"[Pipeline] Scenario {'PASSED' if passed else 'FAILED'}  "
            f"best_reward={best_reward:.3f}  "
            f"iterations={len(iterations)}"
        )

        # ── Step 6: GRPO update ───────────────────────────────────────────────
        if self.config.grpo_update and self.trainer and grpo_samples:
            self._grpo_update(grpo_samples)

        # ── Step 7: Teardown ──────────────────────────────────────────────────
        if twin is not None:
            should_teardown = (
                (passed and self.config.teardown_on_pass) or
                (not passed and self.config.teardown_on_fail)
            )
            if should_teardown:
                twin_teardown(twin)

        result = ScenarioResult(
            scenario_id=scenario_id,
            rca_result=rca_result,
            rca_verified=rca_verified,
            passed=passed,
            best_reward=best_reward,
            best_commands=best_commands,
            iterations=iterations,
            elapsed_seconds=time.time() - t0,
        )

        if self.config.save_results and save_dir:
            _save_result(result, save_dir)

        return result

    def run_batch(
        self,
        scenarios: list[dict],
        save_dir: str = "results/pipeline",
        stop_on_error: bool = False,
    ) -> list[ScenarioResult]:
        """
        Run pipeline on a list of (state, compressed) tuples.

        Args:
            scenarios:    list of {"state": dict, "compressed": dict | None} dicts
            save_dir:     directory to write per-scenario JSON results
            stop_on_error: if True, raise on first unhandled exception

        Returns:
            List of ScenarioResult (one per scenario)
        """
        results = []
        for i, entry in enumerate(scenarios):
            state = entry.get("state") if isinstance(entry, dict) else entry
            compressed = entry.get("compressed") if isinstance(entry, dict) else None
            scenario_id = state.get("scenario_id", f"scenario_{i}")

            print(f"\n[Pipeline] Batch {i + 1}/{len(scenarios)}: {scenario_id}")
            try:
                result = self.run_scenario(state, compressed, save_dir=save_dir)
                results.append(result)
            except Exception as e:
                print(f"[Pipeline] Error on {scenario_id}: {e}")
                traceback.print_exc()
                if stop_on_error:
                    raise
                results.append(self._error_result(scenario_id, {}, str(e), 0.0))

        self._print_batch_summary(results)
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_iteration(
        self,
        state: dict,
        compressed: dict | None,
        rca_result: dict,
        twin,
        iteration: int,
        reward_history: list,
        grpo_samples: list,
    ) -> IterationRecord:
        """Run one prompt → commands → execute → reward cycle."""
        from digital_twin.executor import SolutionExecutor

        t0 = time.time()

        # Build optimizer context
        opt_context = build_prompt_optimizer_input(
            state, compressed, rca_result,
            reward_history=reward_history,
            iteration=iteration,
        )

        # Generate instruction prompt (Qwen2.5-Coder-7B)
        instruction_prompt = self.prompt_optimizer.generate(opt_context)

        # Record for GRPO (reward filled in after execution)
        grpo_samples.append({
            "prompt": _build_grpo_prompt_text(opt_context),
            "completion": instruction_prompt,
            "reward": None,
            "group_id": state.get("scenario_id", "unknown"),
        })

        # Action agent: instruction prompt → commands
        commands = self.action_agent.get_commands(instruction_prompt, opt_context)

        # Execute on twin (or simulate if no twin)
        if twin is not None:
            executor = SolutionExecutor(twin)
            exec_result = executor.run(commands)
            reward = exec_result.reward
            passed = exec_result.passed
            reward_breakdown = exec_result.reward_breakdown
            cmd_results = [c.as_dict() for c in exec_result.commands]
        else:
            # Dry-run: no twin — reward is always 0
            print("[Pipeline] No twin — dry run, reward=0.0")
            reward, passed = 0.0, False
            reward_breakdown, cmd_results = {}, []

        # Fill reward back into GRPO sample
        grpo_samples[-1]["reward"] = reward

        return IterationRecord(
            iteration=iteration,
            instruction_prompt=instruction_prompt,
            commands=commands,
            reward=reward,
            passed=passed,
            elapsed_seconds=time.time() - t0,
            reward_breakdown=reward_breakdown,
            command_results=cmd_results,
        )

    def _grpo_update(self, grpo_samples: list[dict]) -> None:
        """Push samples to the GRPO trainer and trigger a step if ready."""
        for s in grpo_samples:
            if s.get("reward") is None:
                continue
            self.trainer.add_sample(
                prompt_text=s["prompt"],
                completion_text=s["completion"],
                reward=s["reward"],
                group_id=s.get("group_id"),
            )

        if self.trainer.should_update():
            self.trainer.step()

    @staticmethod
    def _error_result(
        scenario_id: str, rca_result: dict, error: str, elapsed: float
    ) -> "ScenarioResult":
        print(f"[Pipeline] ERROR on {scenario_id}: {error}")
        return ScenarioResult(
            scenario_id=scenario_id,
            rca_result=rca_result,
            rca_verified=False,
            passed=False,
            best_reward=0.0,
            best_commands=[],
            iterations=[],
            elapsed_seconds=elapsed,
            error=error,
        )

    @staticmethod
    def _print_batch_summary(results: list["ScenarioResult"]) -> None:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        mean_reward = sum(r.best_reward for r in results) / max(total, 1)
        print(
            f"\n[Pipeline] Batch complete: "
            f"{passed}/{total} passed  "
            f"mean_best_reward={mean_reward:.3f}"
        )


# ── Standalone helpers ────────────────────────────────────────────────────────


def twin_teardown(twin, force: bool = False) -> None:
    """Safely teardown a digital twin, ignoring errors."""
    try:
        twin.teardown()
    except Exception as e:
        if force:
            print(f"[Pipeline] teardown error (ignored): {e}")
        else:
            raise


def _failure_reason(iter_record: "IterationRecord") -> str:
    """Extract a short failure reason from an iteration record."""
    bd = iter_record.reward_breakdown
    post_ready = bd.get("counts", {}).get("post_ready", 0)
    total = bd.get("counts", {}).get("total_tracked", 0)
    if total > 0 and post_ready < total:
        return f"{post_ready}/{total} relevant pods ready after commands"
    if not iter_record.commands:
        return "no commands generated"
    failed_cmds = [
        c for c in iter_record.command_results if not c.get("ok", False)
    ]
    if failed_cmds:
        return f"{len(failed_cmds)} commands failed"
    return "pods not healthy after stabilization"


def _build_grpo_prompt_text(opt_context: dict) -> str:
    """
    Build the plain-text version of the optimizer input for GRPO training.
    Uses the same formatter as prompt_optimizer.py.
    """
    from agents.prompt_optimizer import _format_context_for_qwen
    return _format_context_for_qwen(opt_context)


def _save_result(result: "ScenarioResult", save_dir: str) -> None:
    """Write scenario result to a JSON file."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    safe_id = result.scenario_id.replace("/", "_").replace(" ", "_")
    path = Path(save_dir) / f"{safe_id}.json"
    with open(path, "w") as f:
        json.dump(result.as_dict(), f, indent=2)
    print(f"[Pipeline] Saved result → {path}")
