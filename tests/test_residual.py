import numpy as np
from sklearn.preprocessing import StandardScaler
from tr_deception.common import BOUND
from tr_deception.residual import ResidualState,fit_residual,apply_residual

def test_fitted_bound_center_and_base_coefficient():
    rng=np.random.default_rng(123)
    q=rng.normal(size=(40,2));y=(q[:,0]+.2*q[:,1]>0).astype(int)
    base=rng.normal(scale=.1,size=40);ids=np.repeat(['toy-a','toy-b','toy-c','toy-d'],10)
    state=fit_residual(q,base,y,ids)
    np.testing.assert_array_equal(state.correction(state.scaler.mean_[None,:]),[0.])
    assert np.all(np.abs(state.correction(q))<BOUND)
    np.testing.assert_allclose(apply_residual(state,base+1,q)-apply_residual(state,base,q),1.,rtol=0,atol=1e-15)

def test_residual_analytical_formula():
    scaler=StandardScaler();scaler.mean_=np.array([1.,-2.]);scaler.scale_=np.array([2.,4.]);scaler.n_features_in_=2
    state=ResidualState(scaler,np.array([.3,-.7]),'synthetic fixture')
    q=np.array([[3.,2.],[-1.,-6.],[1.,-2.]]);base=np.array([.5,-.2,1.])
    expected=base+BOUND*np.tanh(np.array([-.4,.4,0.])/BOUND)
    np.testing.assert_allclose(apply_residual(state,base,q),expected,rtol=0,atol=1e-15)
