"""Boundary regressions for the incident Twin training and repair contracts.

All cluster interactions are mocked; none of these results is live calibration.
"""
import copy
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'state_abstraction_full'))
from digital_twin_runtime.targeted_telemetry import (
    MEASUREMENT_CONTRACT, ObservationWindow, TelemetryCollectionError,
    TelemetryCollectionResult, _jaeger_rows, _phase_logs, _prometheus_rows,
    collect_targeted_telemetry,
)
from digital_twin_runtime.twin_spec_builder import build_incident_twin_spec
from digital_twin_runtime.sparse_live_manifest import _sanitize_for_clone
from digital_twin_runtime.telemetry_comparator import compare_symptoms_scoped, score_resolution
from digital_twin_runtime.incident_evidence import with_reference_deviations
from digital_twin_runtime.sparse_live_verifier import SparseLiveTwinVerifier, SparseLiveVerifierConfig
from digital_twin_runtime.reward_calibration import derive_entries, hypothesis_key, application_key, write_calibration, load_calibration
from digital_twin_runtime.compare_resources import compare_captures
from digital_twin_runtime.repair_transfer import digest, validate_plan, apply_bound_plan, export_verified_repair, FORMAT
from digital_twin_runtime.live_action_executor import execute_twin_commands
from training_pipeline.schemas import FaultLabel
from training_pipeline.dataset_integrity import ABSTRACTION_CONTRACT, validate_dataset, file_sha256
from state_abstraction_full.metrics_parser import parse_metrics_snapshot
from state_abstraction_full.workload_parser import parse_workload_from_traces
from state_abstraction_full.compress import compress_state
from training_pipeline.agent_input_safety import agent_input_safety_report


def state(faulty=()):
    names = ['front', 'a', 'b', 'db', 'bystander']
    return {'services': names, 'graph': {'edges': [['front', 'a'], ['a', 'db'], ['front', 'b']]},
            'system': {s: {'health': {'pods_total': 1, 'pods_ready': 0 if s in faulty else 1,
                                     'pods_unready': int(s in faulty)},
                           'deployment': {'replicas_desired': 1, 'replicas_ready': 0 if s in faulty else 1}}
                       for s in names}, 'sla': {'global_sla': {'healthy': not faulty}, 'violated': bool(faulty)}}


def fault(service='a', mechanism='scale_replicas_zero'):
    return FaultLabel(service=service, fault_type='infra_failure', fault_mechanism=mechanism)


class IncidentScopeTests(unittest.TestCase):
    def test_all_observed_services_retained_without_any_rca(self):
        spec = build_incident_twin_spec(state(['a', 'b']))
        self.assertTrue({'a', 'b'}.issubset(spec.services_to_keep))
        self.assertNotIn('bystander', spec.services_to_keep)
        self.assertFalse(spec.target_faults)
        self.assertFalse(spec.resource_summary['scope_depends_on_rca_prediction'])

    def test_healthy_incident_is_not_verifiable(self):
        s = state()
        self.assertEqual(compare_symptoms_scoped(s, s, s['services'])['reproduction_score'], 0)
        with self.assertRaisesRegex(ValueError, 'no observable'):
            build_incident_twin_spec(s)

    def test_excluding_degraded_bystander_fails_closed(self):
        s = state(['a', 'bystander'])
        result = compare_symptoms_scoped(s, s, ['front', 'a', 'db'])
        self.assertEqual(result['reproduction_score'], 0)
        self.assertFalse(result['incident_scope_coverage_complete'])

    def test_root_span_is_not_a_missing_service(self):
        s = state()
        s['traces'] = {'per_edge': {'ROOT->a': {'source': 'ROOT', 'target': 'a', 'count': 2, 'error_ratio': 1}}}
        result = compare_symptoms_scoped(s, s, s['services'])
        self.assertTrue(result['incident_scope_coverage_complete'])
        self.assertGreater(result['reproduction_score'], 0)

    def test_reference_replica_deviation_is_observable(self):
        s = state()
        reference = copy.deepcopy(s)
        s['system']['a']['deployment']['replicas_desired'] = 4
        observed = with_reference_deviations(s, reference)
        self.assertIn('a', observed['observed_deviations'])
        self.assertIn('a', build_incident_twin_spec(observed).services_to_keep)

    def test_renderer_preserves_reference_capacity(self):
        obj = {'kind': 'Deployment', 'metadata': {'name': 'a'}, 'spec': {'replicas': 4, 'template': {'spec': {'containers': []}}}}
        self.assertEqual(_sanitize_for_clone(obj, 'aiops-twin-test')['spec']['replicas'], 4)

    def test_cumulative_restart_counter_does_not_prevent_recovery(self):
        before, after = state(['a']), state()
        after['system']['a']['health'].update(restart_count=7, oomkilled_count=1, infra_issue_flag=True)
        self.assertTrue(score_resolution(before, after)['resolved'])

    def test_remaining_symptom_never_passes_95_percent_clearance(self):
        before = state()
        before['logs'] = {f's{i}': {'error_count': 1} for i in range(100)}
        after = state(); after['logs'] = {'s0': {'error_count': 1}}
        self.assertFalse(score_resolution(before, after)['resolved'])


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.window = ObservationWindow(100, 130, 'faulted')

    def test_jaeger_filters_phase_and_deduplicates(self):
        spans = [{'spanID': str(t), 'startTime': int(t * 1e6), 'duration': 10,
                  'processID': 'p', 'tags': [{'key': 'error', 'value': 'false'}]} for t in [99, 100, 120, 130]]
        data = {'data': [{'traceID': 'trace', 'processes': {'p': {'serviceName': 'a'}}, 'spans': spans}]}
        with patch('digital_twin_runtime.targeted_telemetry._read', return_value=json.dumps(data)) as query:
            rows = _jaeger_rows('app', ['a', 'b'], window=self.window)
        self.assertEqual([r['start_time_unix'] for r in rows], [100, 120])
        self.assertTrue(all(r['has_error'] == 'false' for r in rows))
        self.assertIn('start=100000000', query.call_args.args[0][-1])
        self.assertNotIn('lookback', query.call_args.args[0][-1])

    def test_trace_query_failure_is_not_observed_silence(self):
        with patch('digital_twin_runtime.targeted_telemetry._read', return_value='{"errors":["unavailable"],"data":[]}'):
            with self.assertRaises(RuntimeError):
                _jaeger_rows('app', ['a'], window=self.window)

    def test_complete_empty_query_is_distinct_from_failed_query(self):
        with patch('digital_twin_runtime.targeted_telemetry._read', return_value='{"data":[]}'):
            self.assertEqual(_jaeger_rows('app', ['a'], window=self.window), [])
        result = TelemetryCollectionResult('p', 'n', ['a'], channels={'traces': {'query_succeeded': True}})
        with self.assertRaises(TelemetryCollectionError):
            result.require_complete()

    def test_metric_query_pins_time_and_preserves_rate_unit(self):
        response = {'status': 'success', 'data': {'resultType': 'vector', 'result': [{'metric': {'pod': 'a-12345678-abcde'}, 'value': [130, '0.5']}]}}
        with patch('digital_twin_runtime.targeted_telemetry._read', return_value=json.dumps(response)) as query:
            rows = _prometheus_rows('app', window=self.window)
        self.assertEqual(rows[0]['kpi_name'], 'container_cpu_usage_cores')
        self.assertIn('time=130', query.call_args.args[0][-1])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'metrics.csv'
            with path.open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            metrics = parse_metrics_snapshot([path])['per_service']
        self.assertEqual(metrics['a']['cpu_usage_cores'], 0.5)
        self.assertTrue(metrics['a']['cpu_usage_rate_observed'])
        self.assertEqual(metrics['a']['cpu_usage_delta'], 0)

    def test_log_windows_have_no_phase_overlap(self):
        text = '1970-01-01T00:01:39Z old\n1970-01-01T00:01:40Z current\n1970-01-01T00:02:10Z next\n'
        self.assertEqual(_phase_logs(text, self.window), 'current\n')

    def test_collector_cannot_accept_partial_trace_failure(self):
        session = SimpleNamespace(namespace='app', bundle=SimpleNamespace(objects=[], object_refs=[{'kind': 'Deployment', 'name': 'a'}]))
        with tempfile.TemporaryDirectory() as temp, patch('digital_twin_runtime.targeted_telemetry._read', return_value='{"items":[]}'), patch('digital_twin_runtime.targeted_telemetry._prometheus_rows', return_value=[]), patch('digital_twin_runtime.targeted_telemetry._jaeger_rows', side_effect=RuntimeError('offline')):
            with self.assertRaises(TelemetryCollectionError) as caught:
                collect_targeted_telemetry(session, temp, window=self.window)
            self.assertTrue(any(e['channel'] == 'traces' for e in caught.exception.result.errors))

    def test_workload_rate_uses_root_requests_and_time(self):
        edges = {'ROOT->a': {'request_count': 10, 'source': 'ROOT', 'target': 'a'}, 'a->db': {'request_count': 20, 'source': 'a', 'target': 'db'}}
        result = parse_workload_from_traces(edges, observation_seconds=5)
        self.assertEqual(result['estimated_request_rate'], 2)
        self.assertIsNone(parse_workload_from_traces(edges)['estimated_request_rate'])


class CalibrationTests(unittest.TestCase):
    def controls(self, labels=None, positive=0.9, negative=0.2):
        labels = labels or [fault()]
        key = hypothesis_key(labels, application_key(state()))
        kinds = ['positive', 'no_fault', 'wrong_service', 'wrong_mechanism', 'wrong_variant', 'extra_root']
        if len(labels) > 1:
            kinds.append('missing_root')
        return [{'calibration_key': key, 'scenario_id': f'cal-{i}', 'control': c, 'score': positive if c == 'positive' else negative,
                 'measurement_contract': MEASUREMENT_CONTRACT, 'lifecycle_passed': True, 'telemetry_complete': True,
                 'environment_sha256': 'reference'} for i in range(3) for c in kinds]

    def test_current_single_and_joint_controls_can_qualify(self):
        for labels in ([fault()], [fault(), fault('b')]):
            entries = derive_entries(self.controls(labels))
            entry = next(iter(entries.values()))
            self.assertTrue(entry['eligible'])
            self.assertAlmostEqual(entry['threshold'], 0.55)

    def test_missing_joint_control_is_rejected(self):
        rows = [r for r in self.controls([fault(), fault('b')]) if r['control'] != 'missing_root']
        self.assertFalse(next(iter(derive_entries(rows).values()))['eligible'])

    def test_cross_mechanism_negative_prevents_false_qualification(self):
        a = self.controls(positive=0.6, negative=0.2)
        b = self.controls([fault(mechanism='assign_to_non_existent_node')], positive=0.95, negative=0.8)
        entries = derive_entries(a + b)
        self.assertFalse(entries[a[0]['calibration_key']]['eligible'])
        self.assertTrue(entries[b[0]['calibration_key']]['eligible'])

    def test_failed_telemetry_or_changed_environment_is_ineligible(self):
        for field, value in [('telemetry_complete', False), ('environment_sha256', 'other')]:
            rows = self.controls(); rows[0][field] = value
            self.assertFalse(next(iter(derive_entries(rows).values()))['eligible'])

    def test_tampered_threshold_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cal.json'
            manifest = write_calibration(self.controls(), path, dataset_sha256='frozen')
            next(iter(manifest['entries'].values()))['threshold'] = 0.01
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                load_calibration(path)


class DatasetTests(unittest.TestCase):
    def make_dataset(self, root):
        records = []
        for i, side in enumerate(['train', 'calibration', 'test']):
            sid = side + '-incident'; directory = root / 'processed_states' / sid; directory.mkdir(parents=True)
            public = state(['a']); public['abstraction_contract'] = ABSTRACTION_CONTRACT
            public['system']['a']['deployment']['replicas_desired'] = i + 2
            full = {'fault_context': {'faulty_service': f'service-{i}', 'fault_family': 'scale_pod'}}
            files = []
            for name, content in [('state_abstraction.json', full), ('state_abstraction_compressed.json', public)]:
                path = directory / name; path.write_text(json.dumps(content))
                files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': file_sha256(path)})
            records.append({'scenario_id': sid, 'split': side, 'files': files})
        manifest = {'format': 'frozen_aiops_dataset_v1', 'counts': {'total': 3}, 'records': records}
        path = root / 'manifest.json'; path.write_text(json.dumps(manifest)); (root / 'manifest.sha256').write_text(file_sha256(path))
        return path

    def test_explicit_train_selection_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = self.make_dataset(root)
            result = validate_dataset(path, root / 'processed_states', {'train-incident'})
            self.assertIn('selection_sha256', result)
            with self.assertRaisesRegex(ValueError, 'subset of the train'):
                validate_dataset(path, root / 'processed_states', {'test-incident'})
            public = root / 'processed_states/train-incident/state_abstraction_compressed.json'
            public.write_text(public.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                validate_dataset(path, root / 'processed_states', {'train-incident'})

    def test_standalone_compressed_view_redacts_identity(self):
        full = state(['a'])
        full.update(scenario_id='scale_pod-private', namespace='private-app', timestamp='2026-09-09',
                    fault_context={'faulty_service': 'a', 'fault_family': 'scale_pod'})
        public = compress_state(full)
        self.assertTrue(agent_input_safety_report(public)['safe_for_training_agent'])
        self.assertNotIn('scenario_id', public); self.assertNotIn('timestamp', public); self.assertNotIn('namespace', public)


class RepairTests(unittest.TestCase):
    def plan(self):
        p = {'format': FORMAT, 'measurement_contract': MEASUREMENT_CONTRACT,
             'operations': [{'verb': 'scale', 'kind': 'deployment', 'name': 'a', 'argv': ['scale', 'deployment/a', '--replicas=1']}],
             'services': state()['services'], 'selected_services': state()['services'],
             'expected_incident_state': state(['a']), 'decision_threshold': 0.8,
             'evidence': {'rca_verified': True, 'calibrated': True, 'recovery_verified': True, 'independent_fault_state': True, 'calibration_sha256': 'cal'}}
        return {**p, 'plan_sha256': digest(p)}

    def test_portable_target_metadata_cannot_disagree_with_command(self):
        p = self.plan(); p['operations'][0]['argv'][1] = 'deployment/b'
        p['plan_sha256'] = digest({k: v for k, v in p.items() if k != 'plan_sha256'})
        with self.assertRaisesRegex(ValueError, 'executable target'):
            validate_plan(p)

    def test_unqualified_repair_cannot_be_exported(self):
        with self.assertRaisesRegex(ValueError, 'verification evidence'):
            export_verified_repair(SimpleNamespace(last_rca_result={}), [], {})

    def test_new_action_attempt_revalidates_from_frozen_fault_state(self):
        verifier = SparseLiveTwinVerifier(SparseLiveVerifierConfig('reference', '.', 'state_abstraction_full'))
        verifier._incident_state = state(['a'])
        verifier._action_attempt_count = 1
        with patch.object(verifier, 'validate_rca_prediction', return_value={'rca_twin_verified': True}) as replay, patch.object(verifier, 'current_rca_gate', return_value={'rca_twin_verified': True}):
            verifier.prepare_action_attempt([fault()])
            replay.assert_called_once_with({}, verifier._incident_state, [fault()])

    def test_executor_rejects_namespace_and_cluster_escape_before_execution(self):
        session = SimpleNamespace(namespace='aiops-twin-test', bundle=SimpleNamespace(objects=[], object_refs=[{'kind': 'Deployment', 'name': 'a'}]))
        for extra in ['--context=other', '-n production', '-nproduction', '--server=https://other']:
            with patch('digital_twin_runtime.live_action_executor.subprocess.run') as run:
                result = execute_twin_commands(session, [f'kubectl scale deployment/a --replicas=1 -n aiops-twin-test {extra}'])
                self.assertFalse(result.executed); run.assert_not_called()

    def test_executor_rejects_second_target_and_cluster_kind(self):
        session = SimpleNamespace(namespace='aiops-twin-test', bundle=SimpleNamespace(objects=[], object_refs=[{'kind': 'Deployment', 'name': 'a'}]))
        for cmd in ['kubectl scale deployment a b --replicas=1', 'kubectl scale clusterrole/a --replicas=1']:
            with patch('digital_twin_runtime.live_action_executor.subprocess.run') as run:
                result = execute_twin_commands(session, [cmd + ' -n aiops-twin-test'])
                self.assertFalse(result.executed); run.assert_not_called()

    def test_real_postcheck_failure_rolls_back_with_concurrency_preconditions(self):
        p = self.plan()
        original = {'metadata': {'uid': 'obj', 'resourceVersion': '1'}, 'spec': {'replicas': 0}}
        current = copy.deepcopy(original)
        bound = {'namespace': 'production', 'namespace_uid': 'ns', 'cluster_sha256': 'cluster',
                 'plan_sha256': p['plan_sha256'], 'snapshots': {'deployment/a': original}, 'operations': p['operations']}
        calls = []
        def api(args, **kwargs):
            calls.append(args)
            if args[:2] == ['get', 'namespace']:
                return {'metadata': {'uid': 'ns'}}
            if args[0] == 'get':
                return copy.deepcopy(current)
            if args[0] == 'scale':
                return {**copy.deepcopy(current), 'spec': {'replicas': 1}}
            if args[0] == 'patch':
                payload = json.loads(args[args.index('-p') + 1])
                self.assertTrue(any(x['op'] == 'test' and x['path'] == '/metadata/resourceVersion' for x in payload))
                current['spec'] = next(x['value'] for x in payload if x['path'] == '/spec')
                current['metadata']['resourceVersion'] = str(int(current['metadata']['resourceVersion']) + 1)
                return copy.deepcopy(current)
            raise AssertionError(args)
        def verify(phase):
            return {'state': state(['a']), 'workload': SimpleNamespace(completed=True, failed=False, application_failures=0), 'ready': True}
        with patch('digital_twin_runtime.repair_transfer._json', side_effect=api):
            result = apply_bound_plan(p, bound, verify=verify)
        self.assertFalse(result['success']); self.assertTrue(result['rollback'][0]['restored'])
        self.assertEqual(current['spec']['replicas'], 0)
        self.assertEqual(sum(a[0] == 'patch' for a in calls), 2)


class ResourceAndIdentityTests(unittest.TestCase):
    def capture(self, cpu, memory, pods):
        return {'collection_mode': MEASUREMENT_CONTRACT, 'selected_services': ['a', 'b'], 'errors': [],
                'resources': {'valid': True, 'stable_pod_population': True, 'workload_healthy': True,
                              'includes_observer_overhead': False, 'workload_contract_sha256': 'same',
                              'measurement_window_seconds': 30, 'effective_requests_per_second': 10, 'reference_environment_sha256': 'same',
                              'application_cpu_cores_mean': cpu, 'application_memory_bytes_mean': memory, 'application_running_pods': pods}}

    def test_savings_use_measurements_and_reject_unmatched_workload(self):
        full, twin = self.capture(2, 1000, 5), self.capture(1, 800, 3)
        result = compare_captures(full, twin)
        self.assertAlmostEqual(result['measurements']['application_memory_bytes_mean']['reduction_fraction'], .2)
        twin['resources']['workload_contract_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'workload'):
            compare_captures(full, twin)

    def test_multifault_variant_changes_identity_and_execution_order_matters(self):
        spec = importlib.util.spec_from_file_location('scenario_identity', ROOT / 'dataset_generation/scenario_identity.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        a = {'problem_id': 'multi', 'is_multifault': True, 'subproblems': [{'faulty_service': 'a', 'variant': {'replicas': 2}}, {'faulty_service': 'b'}]}
        b = copy.deepcopy(a); b['subproblems'][0]['variant']['replicas'] = 4
        self.assertNotEqual(module.attach_scenario_identity(a)['problem_id'], module.attach_scenario_identity(b)['problem_id'])
        self.assertNotEqual(module.scenario_spec_hash(a), module.scenario_spec_hash({**a, 'subproblems': a['subproblems'][::-1]}))
        with self.assertRaisesRegex(ValueError, 'distinct scenario'):
            module.unique_scenarios([a, b])
        self.assertEqual(len(module.unique_scenarios([a, a])), 1)

class AdditionalContractTests(unittest.TestCase):
    def test_multifault_has_one_physical_line_and_round_trips(self):
        from training_pipeline.frozen_qwen_agents import _normalize_rca_lines
        from training_pipeline.schemas import parse_fault_lines
        text = _normalize_rca_lines('a::infra_failure::scale_replicas_zero\nb::infra_failure::assign_to_non_existent_node')
        self.assertNotIn('\n', text)
        self.assertEqual([f.service for f in parse_fault_lines(text)], ['a', 'b'])

    def test_service_rates_sum_replicas_without_making_counter_deltas(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'm.csv'
            path.write_text('timestamp,cmdb_id,kpi_name,value\n'
                            '10,a-12345678-abcde,container_cpu_usage_cores,0.4\n'
                            '10,a-12345678-abcdf,container_cpu_usage_cores,0.6\n')
            service = parse_metrics_snapshot([path])['per_service']['a']
            self.assertAlmostEqual(service['cpu_usage_cores'], 1)
            self.assertEqual(service['cpu_usage_delta'], 0)

    def test_counter_resets_are_per_pod_and_duplicate_exports_do_not_double_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'm.csv'
            path.write_text('timestamp,cmdb_id,kpi_name,value\n'
                            '10,a-12345678-abcde,container_cpu_usage_seconds_total,100\n'
                            '20,a-12345678-abcde,container_cpu_usage_seconds_total,2\n'
                            '10,a-12345678-abcdf,container_cpu_usage_seconds_total,10\n'
                            '20,a-12345678-abcdf,container_cpu_usage_seconds_total,13\n')
            service = parse_metrics_snapshot([path, path])['per_service']['a']
            self.assertEqual(service['cpu_usage_delta'], 5)

    def test_reference_environment_change_blocks_qualified_hypothesis(self):
        verifier = SparseLiveTwinVerifier(SparseLiveVerifierConfig('reference', '.', 'state_abstraction_full'))
        s = state(['a']); verifier._incident_state = s; verifier.environment_sha256 = 'changed'
        with patch('digital_twin_runtime.sparse_live_verifier.assess_live_reward_calibration',
                   return_value={'eligible': True, 'threshold': .6, 'environment_sha256': 'old'}):
            result = verifier.validate_rca_prediction({}, s, [fault()])
        self.assertFalse(result['rca_twin_verified'])
        self.assertFalse(result['live_reward_calibrated'])

    def test_invalid_sla_definition_is_rejected(self):
        from state_abstraction_full.sla import configure_sla
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'bad.json'; p.write_text('{"MAX_ERROR_RATIO":-1}')
            with self.assertRaises(ValueError):
                configure_sla(p)

class FrozenRuntimeWorkflowTests(unittest.TestCase):
    def test_scope_baseline_injection_independent_repair_and_export_are_connected(self):
        from contextlib import ExitStack
        from digital_twin_runtime.sparse_live_manifest import SparseManifestBundle
        incident = state(['a', 'b']); sessions = []; phases = []
        healthy = SimpleNamespace(ready=True, to_dict=lambda: {'ready': True})
        workload = SimpleNamespace(completed=True, failed=False, application_failures=0,
                                   total_requests=100, required_ready_endpoints=1, execution_started=True,
                                   to_dict=lambda: {'completed': True, 'total_requests': 100})
        def session_factory(bundle):
            session = SimpleNamespace(bundle=bundle, namespace=bundle.target_namespace, created=False, applied=False,
                                      wait_for_clean_baseline=lambda **kw: healthy)
            session.create_namespace = lambda: setattr(session, 'created', True)
            session.apply_manifests = lambda: setattr(session, 'applied', True)
            session.destroy = lambda: setattr(session, 'created', False)
            sessions.append(session)
            return session
        def plan(namespace, selected):
            return SimpleNamespace(selected_services=selected, configmaps=[],
                controllers=[{'kind': 'Deployment', 'metadata': {'name': name}, 'spec': {'replicas': 1}} for name in selected],
                service_objects=[])
        def render(p, namespace, **kw):
            return SparseManifestBundle('reference', namespace, objects=p.controllers,
                                        object_refs=[{'kind': 'Deployment', 'name': n} for n in p.selected_services])
        def capture(phase, **kw):
            phases.append(phase)
            observed = copy.deepcopy(incident if phase == 'post_injection' else state())
            return {'state': observed, 'workload': workload, 'workloads': [workload], 'channels': {},
                    'coverage': {}, 'collection': {'resources': {'valid': False}}}
        manifestation = SimpleNamespace(manifested=True, to_dict=lambda: {'manifested': True})
        handle = SimpleNamespace(restore_mode='scale_replicas', injected_details={}, wait_for_manifestation=lambda **kw: manifestation)
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            verifier = SparseLiveTwinVerifier(SparseLiveVerifierConfig('reference', temp, 'state_abstraction_full'))
            profile = SimpleNamespace(source_namespace='reference', source_root=Path(temp))
            stack.enter_context(patch.object(verifier, '_profile', return_value=profile))
            stack.enter_context(patch.object(verifier, '_planner_state', side_effect=lambda s, p: copy.deepcopy(s)))
            discover = stack.enter_context(patch('digital_twin_runtime.sparse_live_verifier.discover_sparse_manifest_plan', side_effect=plan))
            stack.enter_context(patch('digital_twin_runtime.sparse_live_verifier.render_sparse_manifest_bundle', side_effect=render))
            stack.enter_context(patch('digital_twin_runtime.sparse_live_verifier.SparseLiveTwinSession', side_effect=session_factory))
            stack.enter_context(patch.object(verifier, '_capture_phase', side_effect=capture))
            inject = stack.enter_context(patch('digital_twin_runtime.sparse_live_verifier.inject_predicted_fault', return_value=handle))
            verifier.prepare_incident_twin(incident)
            inject.assert_not_called()
            self.assertEqual(phases, ['clean_baseline'])
            self.assertTrue({'a', 'b'}.issubset(verifier.selected_services))
            stack.enter_context(patch('digital_twin_runtime.sparse_live_verifier.assess_live_reward_calibration',
                return_value={'eligible': True, 'threshold': .8, 'environment_sha256': verifier.environment_sha256,
                              'manifest_sha256': 'synthetic-test-only'}))
            labels = [fault(), fault('b')]
            result = verifier.validate_rca_prediction({}, incident, labels)
            self.assertTrue(result['rca_twin_verified'], result)
            verifier.prepare_action_attempt(labels)
            command = 'kubectl scale deployment/a --replicas=1 -n ' + verifier.session.namespace
            execution = SimpleNamespace(executed=True, to_dict=lambda: {'executed': True})
            stack.enter_context(patch('digital_twin_runtime.sparse_live_verifier.execute_twin_commands', return_value=execution))
            repaired = verifier.apply_commands_and_score({}, labels, {'service': 'a'}, [command], compressed_state=incident)
            self.assertTrue(repaired['resolved'], repaired)
            self.assertIn('verified_repair_plan', repaired)
            validate_plan(repaired['verified_repair_plan'])
            verifier.prepare_action_attempt(labels)
            self.assertEqual(len(sessions), 2)
            self.assertEqual(phases.count('clean_baseline'), 2)
            self.assertFalse(sessions[0].created)
            self.assertEqual(discover.call_count, 2)  # full reference and selected plan frozen once
            verifier.end_trajectory()


class ObserverAccountingTests(unittest.TestCase):
    def test_telemetry_controller_is_excluded_by_capability_not_its_name(self):
        from digital_twin_runtime.targeted_telemetry import application_services
        from digital_twin_runtime.sparse_live_manifest import SparseManifestBundle
        objects = [
            {"kind": "Deployment", "metadata": {"name": "backend"}, "spec": {"template": {"metadata": {"labels": {"app": "tracing"}}}}},
            {"kind": "Service", "spec": {"selector": {"app": "tracing"}, "ports": [{"port": 16686}]}},
            {"kind": "Deployment", "metadata": {"name": "api"}, "spec": {"template": {"metadata": {"labels": {"app": "api"}}}}},
        ]
        bundle = SparseManifestBundle("source", "aiops-twin-test", objects=objects,
            object_refs=[{"kind": "Deployment", "name": name} for name in ["api", "backend"]])
        self.assertEqual(application_services(SimpleNamespace(bundle=bundle)), ["api"])


if __name__ == '__main__':
    unittest.main()
