from second.core.box_coders import GroundBox3dCoderTorch


def build(box_coder_config):
    """Create optimizer based on config.

    Args:
        optimizer_config: A Optimizer proto message.

    Returns:
        An optimizer and a list of variables for summary.

    Raises:
        ValueError: when using an unsupported input data type.
    """
    box_coder_type = box_coder_config.WhichOneof('box_coder')
    if box_coder_type == 'ground_box3d_coder':
        cfg = box_coder_config.ground_box3d_coder
        return GroundBox3dCoderTorch(cfg.linear_dim, cfg.encode_angle_vector)
    else:
        raise ValueError("unknown box_coder type")
