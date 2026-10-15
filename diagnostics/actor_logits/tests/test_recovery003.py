"""CPU-only recovery boundary tests; no model, simulator, Episode or live network."""
import os
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile
import ssl
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from diagnostics.actor_logits import recovery003 as recovery


class RecoveryBoundaryTests(unittest.TestCase):
    def test_cross_session_evidence_accepts_only_fresh_distinct_sessions(self):
        claim = 'a'*40
        with tempfile.TemporaryDirectory() as td, patch.object(recovery,'NETWORK_WITNESSES',Path(td)):
            root=Path(td)/claim;root.mkdir()
            report={'status':'PASS','transport':recovery.TRANSPORT,'proxy_url':recovery.PROXY_URL,'head_sha':claim,
                    'ci':{'head_sha':claim,'conclusion':'success'},
                    'observed_utc':datetime.now(timezone.utc).isoformat(),'ssh_session_sha256':'session-a'}
            (root/'session_a.json').write_text(json.dumps(report))
            report['ssh_session_sha256']='session-b'
            (root/'session_b.json').write_text(json.dumps(report))
            self.assertEqual(len(recovery.require_disconnect_witnesses(claim)),2)
            report['ssh_session_sha256']='session-a'
            (root/'session_b.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'distinct'):
                recovery.require_disconnect_witnesses(claim)

    def test_cross_session_evidence_rejects_old_or_wrong_claim(self):
        claim='a'*40
        with tempfile.TemporaryDirectory() as td, patch.object(recovery,'NETWORK_WITNESSES',Path(td)):
            root=Path(td)/claim;root.mkdir()
            report={'status':'PASS','transport':recovery.TRANSPORT,'proxy_url':recovery.PROXY_URL,'head_sha':claim,
                    'ci':{'head_sha':claim,'conclusion':'success'},
                    'observed_utc':'2000-01-01T00:00:00+00:00','ssh_session_sha256':'session-a'}
            for name in ['session_a.json','session_b.json']:(root/name).write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'fresh'):
                recovery.require_disconnect_witnesses(claim)
            report['observed_utc']=datetime.now(timezone.utc).isoformat()
            report['head_sha']='b'*40
            (root/'session_a.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'Invalid cross-session'):
                recovery.require_disconnect_witnesses(claim)

    def test_transport_environment_uses_approved_proxy_and_tls(self):
        with patch.dict(os.environ, {'HTTPS_PROXY':'secret-proxy','http_proxy':'other',
                                     'ALL_PROXY':'third','NO_PROXY':'*',
                                     'GIT_SSL_NO_VERIFY':'true','KEEP_TEST_VALUE':'kept'}, clear=True):
            env = recovery.transport_environment()
        self.assertEqual(env['NO_PROXY'], '')
        self.assertEqual(env['no_proxy'], '')
        self.assertEqual(env['KEEP_TEST_VALUE'], 'kept')
        self.assertNotIn('GIT_SSL_NO_VERIFY', env)
        self.assertNotIn('ALL_PROXY', env)
        for key in ['HTTPS_PROXY','HTTP_PROXY','https_proxy','http_proxy']:
            self.assertEqual(env[key], 'http://127.0.0.1:7899')

    def test_git_uses_proxy_and_enforces_tls(self):
        with patch.object(recovery.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='ok')) as run:
            self.assertEqual(recovery.git(Path('/unused'),'rev-parse','HEAD'),'ok')
        argv=run.call_args.args[0]
        self.assertIn('http.proxy=http://127.0.0.1:7899',argv)
        self.assertIn('http.sslVerify=true',argv)
        self.assertEqual(run.call_args.kwargs['env']['HTTPS_PROXY'], recovery.PROXY_URL)

    def test_api_uses_proxy_with_certificate_and_hostname_verification(self):
        with patch.dict(os.environ, {'NO_PROXY':'*','no_proxy':'*'}), \
             patch.object(recovery.urllib.request,'build_opener') as build:
            recovery.github_opener()
            self.assertEqual(os.environ['NO_PROXY'],'*')
            self.assertEqual(os.environ['no_proxy'],'*')
        proxy,https=build.call_args.args
        self.assertEqual(proxy.proxies, {'http':recovery.PROXY_URL,'https':recovery.PROXY_URL})
        self.assertEqual(https._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(https._context.check_hostname)

    def test_explicit_proxy_ignores_bypass_without_global_mutation(self):
        request = recovery.urllib.request.Request('https://api.github.com/repos/example')
        with patch.dict(os.environ, {'NO_PROXY':'*','no_proxy':'*'}), \
             patch.object(recovery.urllib.request,'proxy_bypass',side_effect=AssertionError('bypass called')):
            handler = recovery.GitHubProxyHandler({'https':recovery.PROXY_URL})
            self.assertIsNone(handler.proxy_open(request,recovery.PROXY_URL,'https'))
            self.assertEqual(request.host,'127.0.0.1:7899')
            self.assertEqual(request._tunnel_host,'api.github.com')
            self.assertEqual(os.environ['NO_PROXY'],'*')

    def test_004_bindings_and_immutable_off(self):
        self.assertEqual(recovery.EXPERIMENT,'EXP-ACTOR-RESET-ON200-004')
        self.assertEqual(recovery.CYCLE,'actor-reset-on200-off002-reuse-004-20261015')
        self.assertEqual(str(recovery.OUTPUT),'/nvme2/user/qyy/SafeVLA_p2_004_on200_results_20261015')
        self.assertEqual(str(recovery.EVIDENCE),'/nvme2/user/qyy/SafeVLA_p2_004_launch_20261015')
        self.assertEqual(str(recovery.NETWORK_WITNESSES),'/nvme2/user/qyy/SafeVLA_p2_004_network_claim_20261015')
        self.assertEqual(str(recovery.OFF),'/nvme2/user/qyy/SafeVLA_p2_002_results_20261014/OFF200')

    def test_failed_network_probe_is_not_authorization(self):
        with patch.object(recovery, 'git', side_effect=subprocess.TimeoutExpired('secret', 60)):
            result = recovery.network_probe(Path('/unused'))
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse(result['authorizes_execution'])
        self.assertEqual(result['error_type'], 'TimeoutExpired')
        self.assertNotIn('secret', str(result))

    def test_remote_drift_fails_before_ci_lookup(self):
        with patch.object(recovery,'git',side_effect=['a'*40, 'b'*40+' refs/heads/research-loop']), \
             patch.object(recovery,'ci_success') as ci:
            result = recovery.network_probe(Path('/unused'))
        self.assertEqual(result['status'], 'FAIL')
        ci.assert_not_called()

    def test_scientific_source_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Scientific source drift'):
            recovery.science_parity({'architecture/model.py':'old'}, {'architecture/model.py':'new'})

    def test_removed_source_rejected(self):
        with self.assertRaisesRegex(ValueError, 'removed'):
            recovery.science_parity({'model.py':'old'}, {})

    def test_unreviewed_new_source_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unreviewed new'):
            recovery.science_parity({'model.py':'old'}, {'model.py':'old','extra.py':'new'})

    def test_only_explicit_orchestration_files_may_change(self):
        before = {'model.py':'same','diagnostics/actor_logits/evaluate.py':'old'}
        after = {'model.py':'same','diagnostics/actor_logits/evaluate.py':'new',
                 'diagnostics/actor_logits/recovery003.py':'new'}
        recovery.science_parity(before, after)

    def test_off_execution_rejected_before_any_control_or_data_read(self):
        with patch.object(recovery,'strict_control') as gate:
            with self.assertRaisesRegex(ValueError, 'forbids OFF'):
                recovery.before_model(SimpleNamespace(actor_reset='off', approval_reference='approval',
                    claim_reference='claim',control_root=Path('/control')), [], b'')
        gate.assert_not_called()

    def test_wrong_output_rejected_before_model_gate(self):
        with patch.object(recovery,'strict_control') as gate:
            with self.assertRaisesRegex(ValueError, 'Wrong fresh ON'):
                recovery.before_model(SimpleNamespace(actor_reset='on',output=Path('/wrong'),
                    approval_reference='approval',claim_reference='claim',control_root=Path('/control')), [], b'')
        gate.assert_not_called()

    def test_report_cannot_write_old_off_tree(self):
        with tempfile.TemporaryDirectory() as td, patch.object(recovery, 'OFF', Path(td)/'old'/'OFF200'):
            with self.assertRaisesRegex(ValueError, 'forbidden'):
                recovery.write_report(Path(td)/'old'/'report.json', {'x':1})
            self.assertFalse((Path(td)/'old').exists())

    def test_report_cannot_follow_symlink_into_old_tree(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'old').mkdir();(root/'link').symlink_to(root/'old',target_is_directory=True)
            with patch.object(recovery,'OFF',root/'old'/'OFF200'):
                with self.assertRaisesRegex(ValueError,'forbidden'):
                    recovery.write_report(root/'link'/'report.json', {})
            self.assertFalse((root/'old'/'report.json').exists())

    def test_off_identity_mismatch_stops_before_full_validation(self):
        with patch.object(recovery,'sha256',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'Immutable OFF identity mismatch'):
                recovery.verify_off()


if __name__ == '__main__':
    unittest.main()
