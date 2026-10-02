"""Draw averaging, equal-weight participant AP/AUROC and paired bootstrap."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tr_deception.common import BOOTSTRAP_DRAWS, SEED

def draw_metrics(y, base_probability, self_probability):
    y = np.asarray(y, dtype=np.int8)
    if set(y.tolist()) != {0, 1}:
        raise ValueError('draw needs both classes')
    return dict(base_ap=float(average_precision_score(y, base_probability)), self_ap=float(average_precision_score(y, self_probability)), base_auroc=float(roc_auc_score(y, base_probability)), self_auroc=float(roc_auc_score(y, self_probability)))

def participant_metric(draws):
    keys = ('base_ap', 'self_ap', 'base_auroc', 'self_auroc')
    return {k: float(np.mean([d[k] for d in draws])) for k in keys}

def participant_macro_metrics(participants):
    keys = ('base_ap', 'self_ap', 'base_auroc', 'self_auroc')
    result = {k: float(np.mean([p[k] for p in participants.values()])) for k in keys}
    result['delta_ap'] = result['self_ap'] - result['base_ap']
    result['delta_auroc'] = result['self_auroc'] - result['base_auroc']
    return result

def paired_participant_bootstrap(participants, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """Resample whole participant metric vectors, keeping Base/TR paired. Values contain base_ap, self_ap, base_auroc and self_auroc after within-participant draw averaging. Returns bootstrap draws and percentile 95% intervals; default B=5000, seed=20260722."""
    ids = sorted(participants)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(ids), size=(draws, len(ids)))
    out = {}
    for metric in ('ap', 'auroc'):
        vector = np.array([participants[p]['self_' + metric] - participants[p]['base_' + metric] for p in ids])
        draw = vector[index].mean(axis=1)
        out[metric] = dict(draws=draw, ci=np.quantile(draw, [0.025, 0.975]).tolist())
    return out
