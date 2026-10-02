"""Participant-balanced fusion of two probability vectors.

Visual probabilities must be generated externally.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tr_deception.common import SEED, fusion_participant_weights as participant_weights

class FusionState:

    def __init__(self, scaler, model):
        self.scaler, self.model = (scaler, model)

    def logits(self, p_s, p_f):
        x = np.column_stack((p_s, p_f)).astype(np.float64)
        return self.model.decision_function(self.scaler.transform(x))

def fit_fusion(rows, p_s, p_f):
    ids = [r['participant_id'] for r in rows]
    x = np.column_stack(([p_s[r['sample_id']] for r in rows], [p_f[r['sample_id']] for r in rows])).astype(np.float64)
    y = np.array([r['y'] for r in rows], dtype=np.int8)
    w = participant_weights(ids)
    assert x.shape == (len(rows), 2) and set(y.tolist()) == {0, 1}
    scaler = StandardScaler().fit(x, sample_weight=w)
    model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', fit_intercept=True, class_weight=None, max_iter=10000, random_state=SEED).fit(scaler.transform(x), y, sample_weight=w)
    assert model.classes_.tolist() == [0, 1]
    return FusionState(scaler, model)

def fit_score_fusion(semantic_prob, visual_prob, y, participant_ids):
    """Fit weighted StandardScaler and C=1 liblinear LR on semantic/visual probabilities [N]. Visual probabilities must be generated externally. y uses integer 0=truthful/1=deceptive; participant_ids define training-side weights."""
    from tr_deception.data import validate_training_inputs
    if np.asarray(semantic_prob).ndim != 1 or np.asarray(visual_prob).ndim != 1:
        raise ValueError('fusion scores must be vectors')
    validate_training_inputs(np.column_stack((semantic_prob, visual_prob)), y, participant_ids)
    ps, pf = (np.asarray(semantic_prob), np.asarray(visual_prob))
    labels, ids = (np.asarray(y), np.asarray(participant_ids))
    if not len(ps) == len(pf) == len(labels) == len(ids):
        raise ValueError('score/label/participant lengths differ')
    if not np.isfinite(ps).all() or not np.isfinite(pf).all():
        raise ValueError('scores must be finite')
    if np.any((ps < 0) | (ps > 1) | (pf < 0) | (pf > 1)):
        raise ValueError('inputs must be probabilities')
    rows = [dict(sample_id=str(i), participant_id=str(p), y=int(v)) for i, (p, v) in enumerate(zip(ids, labels))]
    return fit_fusion(rows, {str(i): float(v) for i, v in enumerate(ps)}, {str(i): float(v) for i, v in enumerate(pf)})
