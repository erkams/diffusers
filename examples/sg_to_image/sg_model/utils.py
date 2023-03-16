import torch
MAX_OBJ_LEN = 7
MAX_PRED_LEN = 21

def prepare_sg_embedding(obj_vecs, pred_vecs):
    """Prepare the object and predicate embeddings for the image model.
    Args:
        obj_vecs: (num_objs, obj_dim) object embeddings
        pred_vecs: (num_preds, pred_dim) predicate embeddings
    Returns:
        embedding: (N, dim) embedding of objects and predicates
    """
    
    # Pad the (num_objs, obj_dim) object embeddings with zeros to (MAX_OBJ_LEN, obj_dim) shape
    obj_vecs = torch.nn.functional.pad(obj_vecs, (0, 0, 0, MAX_OBJ_LEN - obj_vecs.shape[0]))

    # Pad the (num_preds, pred_dim) predicate embeddings with zeros to (MAX_PRED_LEN, pred_dim) shape
    pred_vecs = torch.nn.functional.pad(pred_vecs, (0, 0, 0, MAX_PRED_LEN - pred_vecs.shape[0]))

    # Concatenate the object and predicate embeddings
    embedding = torch.cat([obj_vecs, pred_vecs], dim=0)

    return embedding

