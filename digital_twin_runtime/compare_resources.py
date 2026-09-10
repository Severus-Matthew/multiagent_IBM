"""Compare measured application resources under matching healthy workloads.

Observer and cluster overhead are explicitly excluded. Service counts are not
CPU/memory estimates. Changed pod populations and unmatched workloads fail closed.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from .targeted_telemetry import MEASUREMENT_CONTRACT


def compare_captures(full: dict, twin: dict) -> dict:
    for capture in (full, twin):
        if capture.get('collection_mode') != MEASUREMENT_CONTRACT or capture.get('errors'):
            raise ValueError('both captures must use complete current telemetry')
        resources = capture.get('resources', {})
        if not resources.get('valid') or not resources.get('stable_pod_population') or not resources.get('workload_healthy'):
            raise ValueError('resource comparison requires healthy workloads and stable observed pod populations')
        if resources.get('includes_observer_overhead') is not False:
            raise ValueError('resource accounting scopes differ')
    a, b = full['resources'], twin['resources']
    if not a.get('reference_environment_sha256') or a['reference_environment_sha256'] != b.get('reference_environment_sha256'):
        raise ValueError('reference configuration differs between measurements')
    if not a.get('workload_contract_sha256') or a['workload_contract_sha256'] != b.get('workload_contract_sha256'):
        raise ValueError('workload payload, rate, duration, endpoint, or target differs')
    # Include probe startup/cleanup in both observed windows, but do not compare
    # heavily diluted load windows merely because their requested rates match.
    for key, tolerance in [('measurement_window_seconds', .05), ('effective_requests_per_second', .10)]:
        x, y = float(a.get(key, 0)), float(b.get(key, 0))
        if not math.isfinite(x) or not math.isfinite(y) or min(x, y) <= 0 or abs(y / x - 1) > tolerance:
            raise ValueError('observed workload/window comparability failed: ' + key)
    if not set(twin['selected_services']).issubset(full['selected_services']):
        raise ValueError('Twin application scope is not a subset of the full reference')
    result = {'accounting_scope': 'application only; observer and cluster overhead excluded',
              'maximum_window_difference_fraction': .05, 'maximum_observed_rate_difference_fraction': .10,
              'workload_contract_sha256': a['workload_contract_sha256'], 'measurements': {}}
    for key in ('application_cpu_cores_mean', 'application_memory_bytes_mean', 'application_running_pods'):
        x, y = float(a[key]), float(b[key])
        if not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y < 0:
            raise ValueError('a positive observed full baseline and finite Twin measurement are required')
        result['measurements'][key] = {'full': x, 'twin': y, 'reduction_fraction': 1 - y / x}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full_capture', required=True)
    ap.add_argument('--twin_capture', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    result = compare_captures(json.loads(Path(args.full_capture).read_text()), json.loads(Path(args.twin_capture).read_text()))
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
