"""Measure a healthy full application under a verified incident's same probes.

Creates temporary workload jobs in the explicit reference namespace. Application
controllers are never changed. Compare the resulting collection_metadata.json
with that incident's healthy sparse capture using compare_resources.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path
from .repair_transfer import bound_context, validate_plan, observation_verifier


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', required=True)
    ap.add_argument('--context', required=True)
    ap.add_argument('--namespace', required=True)
    ap.add_argument('--application_source_root', required=True)
    ap.add_argument('--state_abstraction_root', default='state_abstraction_full')
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text()); validate_plan(plan)
    observation = deepcopy(plan)
    observation['selected_services'] = list(plan['services'])
    root = Path(args.output_dir).resolve(); root.mkdir(parents=True, exist_ok=False)
    with bound_context(args.context):
        from .sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig
        reference = SparseLiveTwinVerifier(SparseLiveVerifierConfig(
            args.namespace, args.application_source_root, args.state_abstraction_root,
            workload_rate=plan['workload_rate'], workload_duration_seconds=plan['workload_duration_seconds']))
        reference.prepare_scenario({}, plan['expected_incident_state'])
        if (reference._incident_profile.source_namespace != args.namespace
                or reference.environment_sha256 != plan.get('environment_sha256')):
            raise RuntimeError('full reference configuration differs from the measured Twin reference')
        verify = observation_verifier(observation, args.namespace, args, root)
        capture = verify('real_after')
        if (not capture['ready'] or not capture['collection']['resources'].get('workload_healthy')
                or capture['state'].get('observed_deviations')):
            raise RuntimeError('full reference is not healthy under the matched workload')
        print(json.dumps({'collection': capture['collection']['run_dir'], 'resources': capture['collection']['resources']}))


if __name__ == '__main__':
    main()
