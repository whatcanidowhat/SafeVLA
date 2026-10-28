"""Explicit root-only configuration. No environment variables or reset hooks."""


def configure_root_actor(root, *, mask_enabled):
    if type(mask_enabled) is not bool:
        raise TypeError("mask_enabled must be bool")
    for name in ("critic_tsfm", "c_critic_tsfm"):
        critic = getattr(root, name, None)
        if critic is None or not getattr(critic, "actor_cross_episode_attention_mask", True):
            raise ValueError("Expected separate unmodified reward and cost Critic: " + name)
    if not hasattr(root, "decoder") or not hasattr(root, "time_step_counter"):
        raise ValueError("Expected root Actor")
    root.actor_cross_episode_attention_mask = mask_enabled
