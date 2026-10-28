import copy
import unittest
from types import SimpleNamespace as NS
import torch
from diagnostics.actor_memory_unmask.cpu_model import make_model, fixed_input, cache, extract, ROOT
from diagnostics.actor_memory_unmask.configure import configure_root_actor

torch.set_num_threads(1)


class MemoryMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = make_model(max_steps=500)
        self.x = torch.randn(1, 1, 16)

    def call(self, model, t, e=0, x=None):
        with torch.no_grad():
            return model(**fixed_input(self.x if x is None else x, t, e))[0]

    def warm(self):
        for t in range(3):
            self.call(self.model, t, x=self.x + t)

    def test_default_matches_original_forward_and_cache(self):
        old = make_model(max_steps=500, legacy=True)
        old.load_state_dict(self.model.state_dict(), strict=True)
        for e,t in [(0,0),(0,1),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(2,0)]:
            a,b = self.call(self.model,t,e),self.call(old,t,e)
            self.assertTrue(torch.equal(a.distributions.logits,b.distributions.logits))
            self.assertEqual(self.model.time_step_counter,old.time_step_counter)
            for aa,bb in zip(cache(self.model),cache(old)):
                for aaa,bbb in zip(aa,bb): self.assertTrue(torch.equal(aaa,bbb))

    def test_history_mutation_blocked_on_accessible_off(self):
        self.warm()
        for enabled in [True,False]:
            a,b=copy.deepcopy(self.model),copy.deepcopy(self.model)
            a.actor_cross_episode_attention_mask=b.actor_cross_episode_attention_mask=enabled
            b.decoder.layers[-1].attention.cache_v[:,:3] += 7
            la,lb=self.call(a,0,1).distributions.logits,self.call(b,0,1).distributions.logits
            self.assertEqual(torch.equal(la,lb),enabled)

    def test_switch_preserves_all_cache_objects_counter_rng_and_critics(self):
        self.warm()
        self.model.critic_tsfm=make_model(max_steps=500)
        self.model.c_critic_tsfm=make_model(max_steps=500)
        before=cache(self.model); counter=self.model.time_step_counter; rng=torch.get_rng_state().clone()
        ids=[id(l.attention.cache_k) for l in self.model.decoder.layers]
        configure_root_actor(self.model,mask_enabled=False)
        self.assertEqual(counter,self.model.time_step_counter)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(ids,[id(l.attention.cache_k) for l in self.model.decoder.layers])
        for a,b in zip(before,cache(self.model)):
            for aa,bb in zip(a,b): self.assertTrue(torch.equal(aa,bb))
        self.assertTrue(self.model.critic_tsfm.actor_cross_episode_attention_mask)
        self.assertTrue(self.model.c_critic_tsfm.actor_cross_episode_attention_mask)

    def test_prev_actions_null_token_unchanged(self):
        self.warm()
        a,b=copy.deepcopy(self.model),copy.deepcopy(self.model)
        b.actor_cross_episode_attention_mask=False
        seen=[]
        for m in [a,b]:
            h=m.last_actions_embed.register_forward_pre_hook(lambda _,args: seen.append(args[0].clone()))
            self.call(m,0,1); h.remove()
        self.assertTrue(all(torch.equal(x,torch.tensor([[20]])) for x in seen))

    def test_batch_unmask_is_causal(self):
        m=self.model; m.actor_cross_episode_attention_mask=False
        inp=fixed_input(self.x.repeat(3,1,1),0)
        inp['observations']['time_step']=torch.tensor([[0],[1],[0]])
        inp['observations']['traj_index']=torch.tensor([[0],[0],[1]])
        inp['prev_actions']=torch.tensor([[3],[3],[3]])
        inp['masks']=torch.tensor([[[0.]],[[1.]],[[0.]]])
        with torch.no_grad():
            a=m(**inp)[0].distributions.logits.clone()
            inp['observations']['fixed_embedding'][2] += 100
            b=m(**inp)[0].distributions.logits
        self.assertTrue(torch.equal(a[:2],b[:2]))
        self.assertEqual(m.time_step_counter,0)

    def test_real_max_steps_rollover(self):
        m=make_model(dim=16, max_steps=500); m.actor_cross_episode_attention_mask=False
        for i in range(501): self.call(m,i%600,i//600)
        self.assertEqual(m.time_step_counter,1)
        # Old allocated slots remain, but access prefix after rollover is length 1.
        a,b=copy.deepcopy(m),copy.deepcopy(m)
        for l in b.decoder.layers: l.attention.cache_v[:,2:] += 100
        self.assertTrue(torch.equal(self.call(a,501,0).distributions.logits,self.call(b,501,0).distributions.logits))

    def test_separate_critic_outputs_and_forward_counts_unchanged(self):
        source=(ROOT/'architecture/models/allenact_transformer_models/separate_actor_critic.py').read_text()
        base=type(self.model)
        reward=extract(source,'DinoLLAMATxNavActorCriticSeparate',namespace={'DinoLLAMATxNavActorCritic':base})
        safe=extract(source,'SafeDinoLLAMATxNavActorCriticSeparate',namespace={'DinoLLAMATxNavActorCriticSeparate':reward,'SafeActorCriticOutput':NS})
        root=safe.__new__(safe); root.__dict__=copy.deepcopy(self.model.__dict__)
        root.critic_tsfm=make_model(max_steps=500); root.c_critic_tsfm=make_model(max_steps=500)
        for t in range(3): self.call(root,t)
        a,b=copy.deepcopy(root),copy.deepcopy(root); configure_root_actor(b,mask_enabled=False)
        counts=[0,0,0]
        handles=[]
        for i,m in enumerate([b,b.critic_tsfm,b.c_critic_tsfm]):
            def hook(_,args,i=i): counts[i]+=1
            handles.append(m.decoder.register_forward_pre_hook(hook))
        oa,ob=self.call(a,0,1),self.call(b,0,1)
        self.assertTrue(torch.equal(oa.values,ob.values)); self.assertTrue(torch.equal(oa.c_values,ob.c_values))
        self.assertEqual(counts,[1,1,1])
        for ca,cb in [(a.critic_tsfm,b.critic_tsfm),(a.c_critic_tsfm,b.c_critic_tsfm)]:
            for aa,bb in zip(cache(ca),cache(cb)):
                for aaa,bbb in zip(aa,bb): self.assertTrue(torch.equal(aaa,bbb))
        for h in handles: h.remove()

    def test_config_rejects_invalid_or_critic_drift(self):
        with self.assertRaises(TypeError): configure_root_actor(self.model,mask_enabled='off')
        with self.assertRaises(ValueError): configure_root_actor(self.model,mask_enabled=False)


if __name__=='__main__': unittest.main(verbosity=2)
