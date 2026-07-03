"""
digital_twin/verifier.py

RCA Verification — after inject_fault(), checks whether the digital twin
shows the same symptoms as the original scenario recording.

If the twin reproduces the fault correctly, the RCA hypothesis is validated.
If not, we flag it but still proceed (the training loop continues regardless —
this is primarily useful for analysis and future RCA agent improvement).

Verification logic:
  1. Extract unhealthy services from original state["service_health"]
  2. Take a pod health snapshot from the live twin (post fault injection)
  3. Map unhealthy pod names → service names via pod prefix stripping
  4. Compare unhealthy sets: original vs twin
  5. Check root_cause_service specifically — is it unhealthy in the twin?

Usage:
    from digital_twin.verifier import RCAVerifier

    verifier = RCAVerifier()
    result = verifier.verify(original_state, twin, rca_result)

    print(result.verified, result.match_score, result.reasoning)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digital_twin.builder import DigitalTwin

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from state_abstraction_full.utils import normalize_service_name


@dataclass
class VerificationResult:
    """Outcome of RCA symptom verification on the digital twin."""
    verified: bool
    match_score: float             # 0.0–1.0: overlap(original_unhealthy, twin_degraded)
    root_cause_degraded: bool      # is root_cause_service unhealthy in the twin?

    original_unhealthy: list[str]  # from state["service_health"]
    twin_degraded_services: list[str]  # derived from post-inject pod health
    overlap: list[str]             # intersection of above two sets

    wait_seconds: float            # how long we waited for fault to propagate
    reasoning: str

    def as_dict(self) -> dict:
        return {
            "verified": self.verified,
            "match_score": round(self.match_score, 3),
            "root_cause_degraded": self.root_cause_degraded,
            "original_unhealthy": self.original_unhealthy,
            "twin_degraded_services": self.twin_degraded_services,
            "overlap": self.overlap,
            "wait_seconds": self.wait_seconds,
            "reasoning": self.reasoning,
        }


class RCAVerifier:
    """
    Compares the live twin's post-fault symptom profile with the original
    scenario recording to validate the RCA hypothesis.

    A verification is considered PASSED if:
      - The root_cause_service identified by the RCA agent is degraded in the
        twin, OR
      - At least 50% of the originally-unhealthy services are degraded in the
        twin (symptom overlap threshold).
    """

    def __init__(
        self,
        symptom_overlap_threshold: float = 0.5,
        fault_propagation_wait: int = 20,
    ):
        """
        Args:
            symptom_overlap_threshold: fraction of orig unhealthy services that
                must appear degraded in the twin for a match (default 0.5).
            fault_propagation_wait: seconds to wait after inject_fault before
                taking the symptom snapshot (default 20s).
        """
        self.threshold = symptom_overlap_threshold
        self.propagation_wait = fault_propagation_wait

    def verify(
        self,
        original_state: dict,
        twin: "DigitalTwin",
        rca_result: dict,
    ) -> VerificationResult:
        """
        Verify that the twin reproduces the original scenario's fault symptoms.

        Should be called immediately after twin.inject_fault().

        Args:
            original_state: state_abstraction.json dict from the scenario run
            twin:           the live DigitalTwin (fault already injected)
            rca_result:     output from RCAAgent.analyze()

        Returns:
            VerificationResult
        """
        print(
            f"[Verifier] Waiting {self.propagation_wait}s for fault to propagate..."
        )
        time.sleep(self.propagation_wait)

        # ── Original unhealthy services ───────────────────────────────────────
        orig_unhealthy: set[str] = {
            svc
            for svc, h in original_state.get("service_health", {}).items()
            if h.get("status", "healthy") != "healthy"
        }

        # ── Twin degraded services ────────────────────────────────────────────
        pod_snapshot = twin.pod_health_snapshot()
        twin_degraded_svcs: set[str] = {
            _pod_to_service(pod)
            for pod, ready in pod_snapshot.items()
            if not ready
        }
        # Remove "unknown" entries
        twin_degraded_svcs.discard("unknown")
        twin_degraded_svcs.discard("")

        # ── Comparison ────────────────────────────────────────────────────────
        root_cause_svc = rca_result.get("root_cause_service", "")
        root_cause_degraded = (
            root_cause_svc in twin_degraded_svcs
            or any(
                root_cause_svc in svc or svc in root_cause_svc
                for svc in twin_degraded_svcs
            )
        )

        overlap: set[str] = set()
        if orig_unhealthy:
            # Fuzzy match: service names may have slight naming differences
            for orig_svc in orig_unhealthy:
                for twin_svc in twin_degraded_svcs:
                    if orig_svc == twin_svc or orig_svc in twin_svc or twin_svc in orig_svc:
                        overlap.add(orig_svc)
                        break

            match_score = len(overlap) / len(orig_unhealthy)
        else:
            # Original had no unhealthy services — symptom is absence of errors
            match_score = 1.0 if not twin_degraded_svcs else 0.5
            overlap = set()

        verified = root_cause_degraded or match_score >= self.threshold

        reasoning = _build_reasoning(
            verified, root_cause_svc, root_cause_degraded,
            orig_unhealthy, twin_degraded_svcs, overlap, match_score,
        )

        print(
            f"[Verifier] verified={verified}  "
            f"match_score={match_score:.2f}  "
            f"root_cause_degraded={root_cause_degraded}"
        )

        return VerificationResult(
            verified=verified,
            match_score=match_score,
            root_cause_degraded=root_cause_degraded,
            original_unhealthy=sorted(orig_unhealthy),
            twin_degraded_services=sorted(twin_degraded_svcs),
            overlap=sorted(overlap),
            wait_seconds=float(self.propagation_wait),
            reasoning=reasoning,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pod_to_service(pod_name: str) -> str:
    """
    Strip replica-set and pod hash suffixes to recover the service/deployment name.
    Delegates to the shared normalize_service_name from state_abstraction_full.

    Examples:
      user-service-9f655fc76-cvsfn  → user-service
      nginx-thrift-5d9b6c4f8-xkzlm  → nginx-thrift
    """
    return normalize_service_name(pod_name)


def _build_reasoning(
    verified: bool,
    root_cause_svc: str,
    root_cause_degraded: bool,
    orig_unhealthy: set[str],
    twin_degraded: set[str],
    overlap: set[str],
    match_score: float,
) -> str:
    lines = [
        f"Verified: {verified}",
        f"Root cause service '{root_cause_svc}' degraded in twin: {root_cause_degraded}",
        f"Symptom overlap: {len(overlap)}/{len(orig_unhealthy)} "
        f"originally-unhealthy services appear degraded in twin "
        f"(score={match_score:.2f})",
    ]
    if orig_unhealthy - overlap:
        lines.append(
            f"Missing from twin: {sorted(orig_unhealthy - overlap)}"
        )
    if twin_degraded - overlap:
        lines.append(
            f"Extra degraded in twin (not in original): "
            f"{sorted(twin_degraded - overlap)[:5]}"
        )
    return "  |  ".join(lines)
