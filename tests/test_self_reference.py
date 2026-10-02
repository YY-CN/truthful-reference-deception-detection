import numpy as np
from tr_deception.self_reference import build_self_reference_features

def test_single_reference_distance():
    query=np.array([1.,0.,0.]);reference=np.array([[0.,1.,0.]])
    np.testing.assert_array_equal(build_self_reference_features(query,reference),[1.])

def test_three_reference_distances():
    query=np.array([1.,0.,0.]);references=np.eye(3)
    expected=np.array([1.-1./np.sqrt(3.),2./3.])
    np.testing.assert_allclose(build_self_reference_features(query,references),expected,rtol=0,atol=1e-15)
