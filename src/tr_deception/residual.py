"""Shared no-intercept bounded correction with fixed base coefficient one."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from tr_deception.common import BOUND, RHO, head_participant_weights as participant_weights

class ResidualState:

    def __init__(self, scaler, beta, message):
        self.scaler, self.beta, self.message = (scaler, beta, message)

    def logits(self, z0, q):
        a = self.scaler.transform(np.asarray(q, dtype=np.float64)) @ self.beta
        return np.asarray(z0, dtype=np.float64) + BOUND * np.tanh(a / BOUND)

    def correction(self, q):
        a = self.scaler.transform(np.asarray(q, dtype=np.float64)) @ self.beta
        return BOUND * np.tanh(a / BOUND)

def fit_residual(q, z0_oof, y, participant_ids):
    """Fit delta*tanh(beta.T@q_hat/delta) from training-side OOF logits. q: [N,1] for K=1 or [N,2] for K>1; labels are integer 0/1. Inverse query-count weights give equal participant weight. delta=ln2, rho=0.01, weighted scaling, L-BFGS-B, no intercept."""
    from tr_deception.data import validate_training_inputs
    validate_training_inputs(q, y, participant_ids)
    if np.asarray(z0_oof).ndim != 1 or not np.isfinite(z0_oof).all():
        raise ValueError('OOF logits must be a finite vector')
    q = np.asarray(q, dtype=np.float64)
    z0_oof = np.asarray(z0_oof, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    participants = np.asarray(participant_ids)
    if q.ndim != 2 or len(q) != len(z0_oof) or len(q) != len(y):
        raise ValueError('residual dimensions')
    counts = {p: (participants == p).sum() for p in np.unique(participants)}
    w = np.array([1 / counts[p] for p in participants], dtype=np.float64)
    w /= w.sum()
    scaler = StandardScaler().fit(q, sample_weight=w)
    qhat = scaler.transform(q)

    def objective(beta):
        u = qhat @ beta / BOUND
        t = np.tanh(u)
        z = z0_oof + BOUND * t
        loss = np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * RHO * (beta @ beta)
        grad = np.sum((w * (expit(z) - y) * (1 - t * t))[:, None] * qhat, axis=0) + RHO * beta
        return (float(loss), grad)
    result = minimize(objective, np.zeros(q.shape[1], dtype=np.float64), jac=True, method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-08, 'ftol': 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'residual fit failed: {result.message}')
    return ResidualState(scaler, np.asarray(result.x, dtype=np.float64), str(result.message))

class SharedMLPResidual:

    def __init__(self, scaler, beta, message):
        self.scaler, self.beta, self.message = (scaler, beta, message)

    def correction(self, q):
        a = self.scaler.transform(np.asarray(q, dtype=np.float64)) @ self.beta
        return BOUND * np.tanh(a / BOUND)

def fit_shared_residual(q, z, y, participant_ids):
    """Fit one shared correction against the mean BCE of three MLP seed logits [N,3]."""
    q = np.asarray(q, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert z.ndim == 2 and z.shape == (len(q), 3)
    w = participant_weights(participant_ids)
    w /= w.sum()
    scaler = StandardScaler().fit(q, sample_weight=w)
    qhat = scaler.transform(q)

    def objective(beta):
        u = qhat @ beta / BOUND
        t = np.tanh(u)
        zz = z + BOUND * t[:, None]
        loss_row = (np.logaddexp(0, zz) - y[:, None] * zz).mean(axis=1)
        loss = float(np.dot(w, loss_row) + 0.5 * RHO * (beta @ beta))
        grad_row = (expit(zz) - y[:, None]).mean(axis=1) * (1 - t * t)
        grad = np.sum((w * grad_row)[:, None] * qhat, axis=0) + RHO * beta
        return (loss, grad)
    result = minimize(objective, np.zeros(q.shape[1], dtype=np.float64), jac=True, method='L-BFGS-B', options={'maxiter': 1000, 'gtol': 1e-08, 'ftol': 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('shared residual failed: ' + str(result.message))
    return SharedMLPResidual(scaler, np.asarray(result.x, dtype=np.float64), str(result.message))

def apply_residual(state, base_logits, q):
    """Add the shared correction to logits [N] or [N,3] without changing the base coefficient. q has shape [N,1] or [N,2]. Floating-point tanh may reach an endpoint for extreme arguments."""
    z = np.asarray(base_logits, dtype=np.float64)
    if z.ndim == 1:
        return z + state.correction(q)
    return z + state.correction(q)[:, None]
