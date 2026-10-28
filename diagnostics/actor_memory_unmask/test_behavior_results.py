from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .online_results import behavior,aggregate_behavior,summarize,analyze
from diagnostics.actor_logits.logger import COST_KEYS


class BehaviorResultsTests(unittest.TestCase):
    def test_failed_early_end_separate_from_successful_end(self):
        rows=[{'executed_action_name':'end'}]
        self.assertTrue(behavior(rows,{'success':0})['failed_early_end'])
        self.assertFalse(behavior(rows,{'success':1})['failed_early_end'])

    def test_horizon600_not_early_end_and_short_non_end_not_early_end(self):
        self.assertFalse(behavior([{'executed_action_name':'end'}]*600,{'success':0})['failed_early_end'])
        self.assertTrue(behavior([{'executed_action_name':'end'}]*600,{'success':0})['failed_horizon_exhaustion'])
        self.assertFalse(behavior([{'executed_action_name':'move'}],{'success':0})['failed_early_end'])

    def test_missing_exploration_metrics_stay_unavailable(self):
        item=behavior([{'executed_action_name':'end'}],{'success':0})
        self.assertIsNone(item['official_percentage_rooms_visited'])
        summary=aggregate_behavior([item])
        self.assertEqual(summary['failure_room_coverage_n'],0)
        self.assertIsNone(summary['failure_room_coverage_mean'])
        self.assertIsNone(summary['failed_without_target_room_visit_proxy'])

    def test_official_exploration_fields_are_preserved_as_proxies(self):
        item=behavior([{'executed_action_name':'end'}],{'success':0,'percentage_rooms_visited':.25,'total_rooms_visited':1,'for_video_table/has_agent_been_in_room':False})
        self.assertEqual(item['official_percentage_rooms_visited'],.25)
        self.assertTrue(item['failed_without_target_room_visit_proxy'])
        self.assertEqual(aggregate_behavior([item])['insufficient_exploration_causal_conclusion'],'NOT_IDENTIFIED_FROM_THESE_PROXIES')

    def test_200_pair_direction_and_five_cost_components(self):
        baseline={'source_baseline':{'max_steps':500,'episode_horizon':600},'loaded_branches':{name:{'max_steps':500,'decoder_max_seq_len':500,'layers':[{'cache_k_shape':[1,500,4,4],'cache_v_shape':[1,500,4,4]}]} for name in ['actor','reward_critic','cost_critic']}}
        common={k:'same' for k in ['git_commit','source_sha256','checkpoint_sha256','dino_sha256','manifest_sha256','seed','num_workers','shuffle','greedy','test_augmentation','physical_gpu','control_evidence']}
        common.update(actor_reset='off',runtime_baseline=baseline)
        tasks=[{'sample_id':str(i)} for i in range(200)]
        manifests={name:{'tasks':tasks,'action_names':['move','end'],'provenance':dict(common,actor_attention_mask=mode)} for name,mode in [('ON','on'),('OFF','off')]}
        outcomes={}
        for name in ['ON','OFF']:
            cost=2 if name=='ON' else 1
            outcomes[name]=[{'success':name=='OFF','cost':cost,**{k:(cost if k=='corner' else 0) for k in COST_KEYS}} for _ in range(200)]
        def validation(directory,expected):return {'status':'PASS','outcomes':outcomes[Path(directory).name]}
        def read(path):
            path=Path(path)
            if path.name=='RUN_MANIFEST.json':return manifests[path.parent.name]
            name=path.parent.parent.parent.name
            return {'official':{'success':1 if name=='OFF' else 0,'percentage_rooms_visited':.5 if name=='OFF' else .1,'for_video_table/has_agent_been_in_room':name=='OFF'}}
        with tempfile.TemporaryDirectory() as tmp:
            on,off=Path(tmp)/'ON',Path(tmp)/'OFF'
            for group in [on,off]:
                for i in range(200):(group/'episodes'/f'{i:04d}.complete').mkdir(parents=True)
            with patch('diagnostics.actor_memory_unmask.online_results.validate_run',side_effect=validation),patch('diagnostics.actor_memory_unmask.online_results.read_json',side_effect=read),patch('diagnostics.actor_memory_unmask.online_results.read_jsonl',return_value=[{'executed_action_name':'end'}]):
                result=analyze(on,off)
        self.assertEqual(result['delta_SR'],1)
        self.assertEqual(result['delta_mean_cost'],-1)
        self.assertEqual(result['paired_bootstrap_95_SR'],[1.,1.])
        self.assertEqual(result['paired_bootstrap_95_cost'],[-1.,-1.])
        self.assertTrue(result['joint_directional_criterion'])
        self.assertEqual(result['behavior']['mask_on']['failed_early_end'],200)
        self.assertEqual(result['behavior']['mask_off']['failed_early_end'],0)
        self.assertEqual(set(result['paired_tasks'][0]['delta_cost_components']),set(COST_KEYS))
        self.assertEqual(result['paired_tasks'][0]['delta_cost_components']['corner'],-1)


if __name__=='__main__':unittest.main(verbosity=2)
