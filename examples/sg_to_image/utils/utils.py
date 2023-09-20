import torch


def interpolate(im, depth, alphas):
    """
    Interpolates between two image tensors with a magnitude based on the alphas.
    im: (batch_size, 3, H, W)
    depth: (batch_size, 1, H, W)
    alphas: (batch_size,)

    Returns:
    interpolated: (batch_size, 3, H, W) tensor
    """
    a = torch.clone(alphas)
    # a[a >= 0.9] = 1.0
    a = a.view(im.shape[0], 1, 1, 1)
    interpolated = depth * a + im * (1 - a)

    return interpolated
