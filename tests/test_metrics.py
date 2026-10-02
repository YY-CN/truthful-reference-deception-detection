import numpy as np
from tr_deception.evaluation import draw_metrics,participant_metric,participant_macro_metrics,paired_participant_bootstrap
import tr_deception.evaluation as evaluation

def test_equal_participant_macro():
    y_large=np.array([0,1]*4);p_large=np.array([.1,.9]*4)
    y_small=np.array([0,1]);p_small=np.array([.8,.2])
    parts={'toy-large':participant_metric([draw_metrics(y_large,p_large,p_large)]),
           'toy-small':participant_metric([draw_metrics(y_small,p_small,p_small[::-1])])}
    result=participant_macro_metrics(parts)
    assert result['base_ap']==.75 and result['self_ap']==1. and result['base_auroc']==.5

def test_paired_participant_bootstrap_known_resamples(monkeypatch):
    indices=np.array([[0,0],[0,1],[1,1]])
    class Generator:
        def integers(self,low,high,size):
            assert (low,high,size)==(0,2,(3,2))
            return indices.copy()
    monkeypatch.setattr(evaluation.np.random,'default_rng',lambda seed:Generator())
    parts={'toy-b':dict(base_ap=.6,self_ap=.5,base_auroc=.4,self_auroc=.7),
           'toy-a':dict(base_ap=.5,self_ap=.7,base_auroc=.5,self_auroc=.6)}
    result=paired_participant_bootstrap(parts,draws=3)
    np.testing.assert_allclose(result['ap']['draws'],[.2,.05,-.1],rtol=0,atol=1e-15)
    np.testing.assert_allclose(result['ap']['ci'],[-.0925,.1925],rtol=0,atol=1e-15)
    np.testing.assert_allclose(result['auroc']['draws'],[.1,.2,.3],rtol=0,atol=1e-15)
