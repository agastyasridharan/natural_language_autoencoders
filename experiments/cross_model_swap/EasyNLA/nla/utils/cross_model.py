"""Explicit source/NLA contracts for equal-width cross-model training."""


def resolve_critic_depth(source_layer, requested, declared=None):
    """A declared reconstructor depth is independent of the extraction layer."""
    expected = declared if declared is not None else (
        source_layer + 1 if source_layer is not None else None
    )
    if expected is None:
        raise ValueError("Declare nla.reconstructor_num_layers or extraction.layer_index")
    if not isinstance(expected, int) or expected <= 0:
        raise ValueError("Reconstructor depth must be a positive integer")
    if requested is not None and requested != expected:
        raise ValueError(f"Requested AR depth {requested} differs from sidecar depth {expected}")
    return expected


def validate_backbone(cfg, model_config):
    text = getattr(model_config, "text_config", model_config)
    if text.hidden_size != cfg.d_model:
        raise ValueError(f"Source width {cfg.d_model} != NLA width {text.hidden_size}; "
                         "this recipe deliberately supports equal-width swaps only")
    if cfg.nla_model_type and text.model_type != cfg.nla_model_type:
        raise ValueError(f"Sidecar NLA family {cfg.nla_model_type} != {text.model_type}")
    depth = cfg.reconstructor_num_layers
    if depth is not None and depth > text.num_hidden_layers:
        raise ValueError(f"Reconstructor depth {depth} exceeds backbone depth {text.num_hidden_layers}")
