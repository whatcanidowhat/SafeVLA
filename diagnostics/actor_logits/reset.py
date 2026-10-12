"""Separate ON treatment. Logger never calls this function."""


def reset_root_actor(root):
    root.time_step_counter = 0
    for layer in root.decoder.layers:
        layer.attention.cache_k.zero_()
        layer.attention.cache_v.zero_()
