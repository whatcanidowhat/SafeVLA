import copy
import unittest
import torch
from .baseline_contract import source_baseline,assert_runtime_baseline
from .cpu_model import ROOT,make_model,fixed_input,cache

torch.set_num_threads(1)


class BaselineContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.root=make_model()
        self.root.critic_tsfm=make_model()
        self.root.c_critic_tsfm=make_model()
        for m in self.branches():
            with torch.no_grad():m(**fixed_input(torch.ones(1,1,16),0))

    def branches(self):return [self.root,self.root.critic_tsfm,self.root.c_critic_tsfm]

    def test_source_configuration_chain(self):
        result=source_baseline(ROOT)
        self.assertEqual(result['max_steps'],500)
        self.assertEqual(result['kv_cache_capacity'],500)
        self.assertEqual(result['episode_horizon'],600)
        self.assertEqual(result['model_class_default_not_effective_config'],1000)

    def test_actual_three_branches_pass_without_mutation(self):
        before=[cache(m) for m in self.branches()]
        counters=[m.time_step_counter for m in self.branches()]
        identities=[(id(l.attention.cache_k),id(l.attention.cache_v)) for m in self.branches() for l in m.decoder.layers]
        rng=torch.get_rng_state().clone();calls=[];handles=[]
        for m in self.branches():handles.append(m.decoder.register_forward_pre_hook(lambda *args:calls.append(1)))
        result=assert_runtime_baseline(self.root,ROOT)
        self.assertEqual(set(result['loaded_branches']),{'actor','reward_critic','cost_critic'})
        self.assertFalse(calls)
        self.assertEqual(counters,[m.time_step_counter for m in self.branches()])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(identities,[(id(l.attention.cache_k),id(l.attention.cache_v)) for m in self.branches() for l in m.decoder.layers])
        for previous,m in zip(before,self.branches()):
            for (ka,va),(kb,vb) in zip(previous,cache(m)):
                self.assertTrue(torch.equal(ka,kb));self.assertTrue(torch.equal(va,vb))
        for h in handles:h.remove()

    def test_wrong_actor_configuration_rejected_not_repaired(self):
        self.root.max_steps=1000
        with self.assertRaises(AssertionError):assert_runtime_baseline(self.root,ROOT)
        self.assertEqual(self.root.max_steps,1000)

    def test_empty_batch_cache_retains_valid_capacity500(self):
        root=make_model();root.critic_tsfm=make_model();root.c_critic_tsfm=make_model()
        for branch in [root,root.critic_tsfm,root.c_critic_tsfm]:branch.decoder.sampler_select([])
        result=assert_runtime_baseline(root,ROOT)
        for branch in result['loaded_branches'].values():
            self.assertEqual(branch['counter'],0)
            self.assertTrue(all(layer['cache_k_shape'][0]==0 and layer['cache_k_shape'][1]==500 for layer in branch['layers']))

    def test_reward_critic_mismatch_rejected_not_repaired(self):
        self.root.critic_tsfm.decoder.params.max_seq_len=1000
        with self.assertRaises(AssertionError):assert_runtime_baseline(self.root,ROOT)
        self.assertEqual(self.root.critic_tsfm.decoder.params.max_seq_len,1000)

    def test_cost_critic_cache_mismatch_rejected_not_repaired(self):
        attention=self.root.c_critic_tsfm.decoder.layers[0].attention
        attention.cache_v=attention.cache_v[:,:499]
        with self.assertRaises(AssertionError):assert_runtime_baseline(self.root,ROOT)
        self.assertEqual(attention.cache_v.shape[1],499)

    def test_rollover_matches_legacy_at500(self):
        new=make_model();old=make_model(legacy=True)
        old.load_state_dict(new.state_dict(),strict=True)
        x=torch.ones(1,1,16)
        for step in range(501):
            with torch.no_grad():
                a=new(**fixed_input(x,step,0))[0].distributions.logits
                b=old(**fixed_input(x,step,0))[0].distributions.logits
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(new.time_step_counter,old.time_step_counter)
            if step in [498,499,500]:
                for (ka,va),(kb,vb) in zip(cache(new),cache(old)):
                    self.assertTrue(torch.equal(ka,kb));self.assertTrue(torch.equal(va,vb))
        self.assertEqual(new.time_step_counter,1)


if __name__=='__main__':unittest.main(verbosity=2)
