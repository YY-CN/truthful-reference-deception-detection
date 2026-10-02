"""Paper constants, participant folds and balanced sample weights."""
import numpy as np
from sklearn.model_selection import KFold

SEED = 20260722
BASE_C = 0.001
RHO = 0.01
BOUND = 0.6931471805599453
BOOTSTRAP_DRAWS = 5000
SVM_C = 0.001
CALIBRATION_C = 1.0
CALIBRATION_FOLDS = 4
RESIDUAL_FOLDS = 5
PROBABILITY_CLIP = 1e-7
MLP_SEEDS = (20260722, 20260723, 20260724)
MLP_EPOCHS = 30
MLP_BATCH_SIZE = 64
MLP_LR = 3e-4
MLP_WEIGHT_DECAY = 1e-4
def require_disjoint(train_ids, test_ids):
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f'participant leakage: {sorted(overlap)[:5]}')

def participant_folds(participants, count):
    ids = sorted(set(participants))
    if len(ids) < count:
        raise ValueError('insufficient participants')
    splitter = KFold(n_splits=count, shuffle=True, random_state=SEED)
    for fold, (train, valid) in enumerate(splitter.split(ids)):
        a = {ids[i] for i in train}
        b = {ids[i] for i in valid}
        assert not a & b and a | b == set(ids)
        yield (fold, a, b)

def participant_weights(participant_ids):
    ids = np.asarray(participant_ids)
    unique, counts = np.unique(ids, return_counts=True)
    count = dict(zip(unique, counts))
    w = np.array([1.0 / count[p] for p in ids], dtype=np.float64)
    return w * (len(w) / w.sum())

def head_participant_weights(ids):
    ids = np.asarray(ids)
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    raw = 1.0 / counts[inverse]
    return raw * (len(ids) / raw.sum())

def fusion_participant_weights(ids):
    ids = np.asarray(ids)
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w * len(w) / w.sum()
