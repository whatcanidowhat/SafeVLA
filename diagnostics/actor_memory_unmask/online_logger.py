"""Passive inherited logger, with root mask mode assertions only."""
from diagnostics.actor_logits.logger import ActorLogitsLogger, guarded
from diagnostics.actor_logits.storage import require


class ActorMemoryLogger(ActorLogitsLogger):
    def __init__(self,root,output,tasks,action_names,provenance):
        provenance['group']='MASK_ON200' if provenance['actor_attention_mask']=='on' else 'MASK_OFF200'
        super().__init__(root,output,tasks,action_names,provenance)

    @guarded
    def _root_before(self,module,args):
        require(module.actor_cross_episode_attention_mask==(self.manifest['provenance']['actor_attention_mask']=='on'),'Root mask mode drift')
        require(module.critic_tsfm.actor_cross_episode_attention_mask and module.c_critic_tsfm.actor_cross_episode_attention_mask,'Critic mask changed')
        super()._root_before(module,args)

    @guarded
    def _decoder(self,module,args):
        if self.manifest['provenance']['actor_attention_mask']=='off':require(args[2] is None,'Unmask root still passed episode mask')
        else:require(args[2] is not None,'Mask ON did not produce episode mask')
        super()._decoder(module,args)
