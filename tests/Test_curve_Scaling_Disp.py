import numpy as np
from curveCoupling import ndcurve


def test_ndims(ndims: int = 2, idx: int = 0):
    data = np.array(
        [[0.0] * ndims, np.random.uniform(0.0, 1.0, ndims), [1.0] * ndims])
    curve = ndcurve(data)
    factor = 2.0
    cte = 1.0
    scaled_curve = curve.scale(factor, idx)
    added_curve = curve.add_cte(cte, idx)
    t = np.linspace(0.0, 1.0, 20)
    val = curve(t)
    val_scaled = val.copy()
    val_scaled[:, idx] *= factor
    val_added = val.copy()
    val_added[:, idx] += cte
    assert np.allclose(val_scaled, scaled_curve(t)), "Error in scaling"
    assert np.allclose(val_added, added_curve(t)), "Error in displacements"


def test_no_dim():
    data = np.array([0.0, np.random.uniform(0.0, 1.0), 1.0])
    curve = ndcurve(data)
    factor = 2.0
    cte = 1.0
    scaled_curve = curve.scale(factor)
    added_curve = curve.add_cte(cte)
    t = np.linspace(0.0, 1.0, 20)
    assert np.allclose(curve(t)*factor, scaled_curve(t)), "Error in scaling"
    assert np.allclose(curve(t)+cte, added_curve(t)), "Error in displacements"


if __name__ == "__main__":
    test_ndims(2,0)
    test_no_dim()
