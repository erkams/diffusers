def interpolate(im, depth, alphas):
    """
    Interpolates between two image tensors with a magnitude based on the alphas.
    im: (batch_size, 3, H, W)
    depth: (batch_size, 1, H, W)
    alphas: (batch_size,)

    Returns:
    interpolated: (batch_size, 3, H, W) tensor
    """

    interpolated = im * alphas + depth * (1 - alphas)

    return interpolated
