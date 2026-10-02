"""SELF-ONLY cosine distances to known truthful responses from the same participant."""
import numpy as np

def _validate_reference_inputs(query, truth_embeddings, *, batch):
    x = np.asarray(query, dtype=np.float64)
    refs = np.asarray(truth_embeddings, dtype=np.float64)
    if x.ndim != (2 if batch else 1) or refs.ndim != 2 or refs.shape[1] != x.shape[-1] or (refs.shape[1] < 1):
        raise ValueError('invalid query or truthful-reference shape')
    if len(refs) < 1:
        raise ValueError('truthful references must be nonempty')
    if batch and len(x) < 1:
        raise ValueError('queries must be nonempty')
    if not np.isfinite(x).all():
        raise ValueError('query embeddings must be finite')
    if not np.isfinite(refs).all():
        raise ValueError('reference embeddings must be finite')
    query_norm = np.linalg.norm(x, axis=1) if batch else np.linalg.norm(x)
    if np.any(abs(query_norm - 1) > 0.001) or np.any(abs(np.linalg.norm(refs, axis=1) - 1) > 0.001):
        raise ValueError('query and reference embeddings must be L2 normalized')
    if len(refs) > 1 and np.linalg.norm(refs.mean(axis=0)) <= 1e-12:
        raise ValueError('Truthful-reference centroid has zero norm; cosine relation is undefined.')
    return (x, refs)

def build_self_reference_features(query, truth_embeddings):
    """Return float64 q for one query [d] and L2-normalized references [K,d]. K=1 gives d_self; K>1 gives normalized-centroid distance and mean individual distance. No query label is used."""
    x, refs = _validate_reference_inputs(query, truth_embeddings, batch=False)
    if len(refs) == 1:
        return np.array([1.0 - float(x @ refs[0])], dtype=np.float64)
    mean = refs.mean(axis=0)
    norm = np.linalg.norm(mean)
    return np.array([1.0 - float(x @ (mean / norm)), float(np.mean(1.0 - refs @ x))], dtype=np.float64)

def build_batch_self_reference_features(queries, truth_embeddings):
    """Return reference features for queries [N,d], preserving scalar feature definitions."""
    x, refs = _validate_reference_inputs(queries, truth_embeddings, batch=True)
    if len(refs) == 1:
        return (1.0 - x @ refs[0]).reshape(-1, 1)
    mean = refs.mean(axis=0)
    mean /= np.linalg.norm(mean)
    return np.column_stack((1.0 - x @ mean, 1.0 - (x @ refs.T).mean(axis=1)))
