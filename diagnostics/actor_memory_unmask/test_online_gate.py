import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import unittest
from diagnostics.actor_memory_unmask.online_gate import validate_binding,EXPERIMENT,CYCLE,COMMAND

ROOT=Path(__file__).resolve().parents[2]


class OnlineGateTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,10,28,tzinfo=timezone.utc)
        self.s={'status':'CODEX_RUNNING','next_actor':'CODEX','updated_by':'CODEX','experiment_id':EXPERIMENT,
                'cycle_id':CYCLE,'instruction_commit':'A','claim_id':'NEW','execution_worktree':str(ROOT),
                'authorization':{'status':'APPROVED','approved_by':'PI','max_gpu':1,'max_episodes':400,'expires_at_utc':'2026-10-29T00:00:00Z'}}
        self.d={'status':'APPROVED','experiment_id':EXPERIMENT,'cycle_id':CYCLE,'code_sha':'SOURCE',
                'command':COMMAND,'resources':{'max_gpu':1,'max_episodes':400},'max_decisions':240000,
                'max_runtime_seconds':43200,'gpu_logger_equivalence':'PI_WAIVED_FOR_THIS_EXPERIMENT',
                'asset_inventory_sha256':'HASH','asset_inventory_path':'PATH','rendering_review':'PI_ACCEPTED_FOR_BOTH_ARMS'}
        self.d['fixed_conditions']={'max_steps':500,'kv_cache_capacity':500,'episode_horizon':600,'actor_reset':'off'}
        self.d['environment_review']='PI_ACCEPTED_TARGET_ENVIRONMENT_FOR_BOTH_ARMS'

    def validate(self):return validate_binding(self.s,self.d,'SOURCE',ROOT,'A',now=self.now)

    def test_valid_binding(self):self.validate()

    def test_current_blocked_state_rejected(self):
        self.s['status']='BLOCKED'
        with self.assertRaises(ValueError):self.validate()

    def test_old_reset_claim_rejected(self):
        self.s['experiment_id']='EXP-ACTOR-RESET-ON200-004'
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_cycle_rejected(self):
        self.s['cycle_id']='old-cycle'
        with self.assertRaises(ValueError):self.validate()

    def test_expired_rejected(self):
        self.s['authorization']['expires_at_utc']='2026-10-16T00:00:00Z'
        with self.assertRaises(ValueError):self.validate()

    def test_only200_budget_rejected(self):
        self.s['authorization']['max_episodes']=200
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_source_rejected(self):
        self.d['code_sha']='OLD'
        with self.assertRaises(ValueError):self.validate()

    def test_historical_waiver_rejected(self):
        self.d['gpu_logger_equivalence']='NOT_RUN_PI_WAIVED'
        with self.assertRaises(ValueError):self.validate()

    def test_fixture1000_is_not_online_baseline(self):
        self.d['fixed_conditions']['max_steps']=1000
        with self.assertRaises(ValueError):self.validate()

    def test_cache1000_is_not_online_baseline(self):
        self.d['fixed_conditions']['kv_cache_capacity']=1000
        with self.assertRaises(ValueError):self.validate()

    def test_horizon_must_remain600(self):
        self.d['fixed_conditions']['episode_horizon']=500
        with self.assertRaises(ValueError):self.validate()

    def test_unreviewed_target_environment_rejected(self):
        self.d['environment_review']='NOT_REVIEWED'
        with self.assertRaises(ValueError):self.validate()

    def test_missing_asset_review_rejected(self):
        del self.d['asset_inventory_sha256']
        with self.assertRaises(ValueError):self.validate()

    def test_missing_render_review_rejected(self):
        self.d['rendering_review']='NOT_REVIEWED'
        with self.assertRaises(ValueError):self.validate()

    def test_wrong_worktree_rejected(self):
        self.s['execution_worktree']='/old/server'
        with self.assertRaises(ValueError):self.validate()

    def test_no_claim_rejected(self):
        self.s['claim_id']=None
        with self.assertRaises(ValueError):self.validate()

    def test_new_evaluator_static_isolation(self):
        source=(ROOT/'diagnostics/actor_memory_unmask/evaluate_online.py').read_text()
        tree=ast.parse(source)
        execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        first=execute.body[0]
        self.assertIsInstance(first,ast.Assign)
        self.assertEqual(first.value.func.id,'codex_control_evidence')
        self.assertIn('agent.actor_boundary_reset = None',source)
        self.assertNotIn('reset_root_actor',source)
        self.assertNotIn('CUDA_VISIBLE_DEVICES=',source)
        self.assertIn("configure_root_actor(agent.actor_critic, mask_enabled=args.actor_mask=='on')",source)
        self.assertLess(source.index("assert_runtime_baseline(agent.actor_critic, REPO)"),source.index("configure_root_actor(agent.actor_critic, mask_enabled=args.actor_mask=='on')"))

    def test_decision_cap_uses_actual_validator_schema(self):
        source=(ROOT/'diagnostics/actor_memory_unmask/online_pair.py').read_text()
        self.assertIn("report['decisions_checked']",source)


if __name__=='__main__':unittest.main(verbosity=2)
