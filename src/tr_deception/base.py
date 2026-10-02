"""Participant-balanced semantic logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tr_deception.common import BASE_C, SEED, participant_weights

class BaseModel:

    def __init__(self, scaler, model):
        self.scaler, self.model = (scaler, model)

    def logits(self, x):
        return self.model.decision_function(self.scaler.transform(np.asarray(x, dtype=np.float64)))

def _fit_base(x, y, participant_ids):
    from tr_deception.data import validate_training_inputs
    validate_training_inputs(x, y, participant_ids)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    w = participant_weights(participant_ids)
    assert x.ndim == 2 and x.shape[1] >= 1 and (len(set(y)) == 2)
    scaler = StandardScaler().fit(x, sample_weight=w)
    model = LogisticRegression(penalty='l2', C=BASE_C, solver='liblinear', fit_intercept=True, class_weight=None, max_iter=10000, random_state=SEED).fit(scaler.transform(x), y, sample_weight=w)
    assert model.classes_.tolist() == [0, 1]
    return BaseModel(scaler, model)

def fit_base_model(X, y, participant_ids, C=BASE_C):
    """Fit weighted StandardScaler and L2/liblinear LR. X is [N,d], y is integer 0=truthful/1=deceptive; participant IDs define inverse-count weights. Formal d=768 and C=0.001."""
    if C != BASE_C:
        raise ValueError('C must equal the frozen paper value')
    return _fit_base(X, y, participant_ids)

def predict_base_logits(model, embeddings):
    """Return one base logit per supplied embedding row."""
    return model.logits(embeddings)
