"""Weighted SVM calibration and three-seed MLP robustness heads."""
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from tr_deception.common import (SEED, SVM_C, CALIBRATION_C, CALIBRATION_FOLDS,
    PROBABILITY_CLIP, participant_folds, head_participant_weights as participant_weights,
    MLP_SEEDS, MLP_EPOCHS, MLP_BATCH_SIZE, MLP_LR, MLP_WEIGHT_DECAY)

def fit_margin(rows):
    from tr_deception.data import validate_training_inputs
    validate_training_inputs(np.stack([r['x'] for r in rows]), [r['y'] for r in rows], [r['participant_id'] for r in rows])
    x = np.stack([r['x'] for r in rows])
    y = np.array([r['y'] for r in rows], dtype=np.int8)
    ids = [r['participant_id'] for r in rows]
    w = participant_weights(ids)
    scaler = StandardScaler().fit(x, sample_weight=w)
    head = LinearSVC(C=SVM_C, penalty='l2', loss='squared_hinge', dual='auto', tol=0.0001, max_iter=100000, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=SEED)
    head.fit(scaler.transform(x), y, sample_weight=w)
    assert head.classes_.tolist() == [0, 1]
    return (scaler, head)

class CalibratedSVM:

    def __init__(self, scaler, head, calibrator):
        self.scaler, self.head, self.calibrator = scaler, head, calibrator

    def logits(self, x):
        margin = self.head.decision_function(self.scaler.transform(np.asarray(x, dtype=np.float64)))
        probability = self.calibrator.predict_proba(np.asarray(margin).reshape(-1, 1))[:, 1]
        probability = np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
        return np.log(probability) - np.log1p(-probability)

def fit_calibrated_svm(rows, participants):
    """Fit a weighted LinearSVC with four-fold held-participant sigmoid calibration. Rows contain sample_id, participant_id, integer y (0 truthful, 1 deceptive), and numeric x [768]."""
    participants = set(participants)
    rows = sorted((r for r in rows if r['participant_id'] in participants), key=lambda r: r['sample_id'])
    if not rows or {r['participant_id'] for r in rows} != participants:
        raise ValueError('training rows must cover the declared participants')
    from tr_deception.data import validate_training_inputs
    validate_training_inputs(np.stack([r['x'] for r in rows]), [r['y'] for r in rows], [r['participant_id'] for r in rows])
    margin_by_sample = {}
    for fold, train, held in participant_folds(participants, CALIBRATION_FOLDS):
        fit_rows = [r for r in rows if r['participant_id'] in train]
        held_rows = [r for r in rows if r['participant_id'] in held]
        scaler, head = fit_margin(fit_rows)
        margin = head.decision_function(scaler.transform(np.stack([r['x'] for r in held_rows])))
        margin_by_sample.update({r['sample_id']: float(v) for r, v in zip(held_rows, margin)})
    assert set(margin_by_sample) == {r['sample_id'] for r in rows}
    xcal = np.array([margin_by_sample[r['sample_id']] for r in rows], dtype=np.float64).reshape(-1, 1)
    ycal = np.array([r['y'] for r in rows], dtype=np.int8)
    w = participant_weights([r['participant_id'] for r in rows])
    calibrator = LogisticRegression(penalty='l2', C=CALIBRATION_C, solver='lbfgs', fit_intercept=True, max_iter=10000, random_state=SEED)
    calibrator.fit(xcal, ycal, sample_weight=w)
    scaler, head = fit_margin(rows)
    return CalibratedSVM(scaler, head, calibrator)

def architecture():
    model = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Linear(128, 1))
    assert sum((p.numel() for p in model.parameters())) == 98561
    return model

class MLPEnsemble:

    def __init__(self, scaler, models):
        self.scaler, self.models = scaler, models

    def logits(self, x):
        x = self.scaler.transform(np.asarray(x, dtype=np.float64)).astype(np.float32)
        with torch.no_grad():
            tensor = torch.from_numpy(x)
            return np.column_stack([model(tensor).squeeze(1).numpy().astype(np.float64) for model in self.models])

def fit_mlp(rows, participants):
    """Fit weighted 768->128->1 ReLU MLPs with three fixed seeds, 30 epochs and AdamW. Rows contain integer y (0 truthful, 1 deceptive), x [768], sample_id and training participant_id. logits returns [N,3]."""
    participants = set(participants)
    rows = sorted((r for r in rows if r['participant_id'] in participants), key=lambda r: r['sample_id'])
    if not rows or {r['participant_id'] for r in rows} != participants:
        raise ValueError('training rows must cover the declared participants')
    from tr_deception.data import validate_training_inputs
    validate_training_inputs(np.stack([r['x'] for r in rows]), [r['y'] for r in rows], [r['participant_id'] for r in rows])
    x = np.stack([r['x'] for r in rows])
    if x.shape[1] != 768:
        raise ValueError('frozen MLP architecture requires 768-dimensional embeddings')
    y = np.array([r['y'] for r in rows], dtype=np.float32)
    w = participant_weights([r['participant_id'] for r in rows]).astype(np.float32)
    scaler = StandardScaler().fit(x, sample_weight=w)
    xt = torch.from_numpy(scaler.transform(x).astype(np.float32))
    yt = torch.from_numpy(y)
    wt = torch.from_numpy(w)
    models = []
    for seed in MLP_SEEDS:
        torch.manual_seed(seed)
        model = architecture()
        optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
        rng = torch.Generator().manual_seed(seed)
        model.train()
        for _ in range(MLP_EPOCHS):
            order = torch.randperm(len(rows), generator=rng)
            for indices in order.split(MLP_BATCH_SIZE):
                logits = model(xt[indices]).squeeze(1)
                loss_rows = nn.functional.binary_cross_entropy_with_logits(logits, yt[indices], reduction='none')
                loss = (wt[indices] * loss_rows).sum() / wt[indices].sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        models.append(model)
    return MLPEnsemble(scaler, models)
