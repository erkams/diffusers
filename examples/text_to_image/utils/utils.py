import torch

def difference(source, target):
    """Returns the difference between two set of tensors.

    Args:
        source: The source triplets.
        target: The target triplets.

    Returns:
        neg: set difference source \ target
        pos: set difference target \ source
    """
    combined = torch.cat((source, target, target))
    uniques, counts = combined.unique(return_counts=True, dim=0)
    neg = uniques[counts == 1]
    pos = uniques[counts == 2]
    # intersection = uniques[counts > 2]
    return neg, pos

def triplet_to_text(triplets, objects, vocab):
    """Converts the triplets to a text.

    Args:
        triplet: The list of triplets to convert.
        vocab: The vocabulary to use.

    Returns:
        The text.
    """
    text = []

    for tri in triplets:
        s,p,o = tri.chunk(3)
        if p == 0:
            continue
        text.append(f"{vocab['object_idx_to_name'][objects[s]]} {vocab['pred_idx_to_name'][p]} {vocab['object_idx_to_name'][objects[o]]}")
    
    return ', '.join(text)
        