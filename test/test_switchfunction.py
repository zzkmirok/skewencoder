import numpy as np

from skewencoder.switchfunction import SwitchFun


def rational(d, r0, d0, n, m):
    shifted = (d - d0) / r0
    return (1. - shifted ** n) / (1. - shifted ** m)


def test_default_d0_is_zero():
    sf = SwitchFun(r0=1.7)
    assert sf.d0 == 0.


def test_d0_from_options():
    sf = SwitchFun(r0=1.7, options={"m": 8, "n": 6, "d0": 0.5})
    assert sf.d0 == 0.5


def test_call_without_d0_matches_plain_rational():
    x = np.array([[1.1, 1.8], [1.2, 2.0]])
    sf = SwitchFun(r0=1.7, options={"m": 8, "n": 6})
    expected = rational(x, r0=1.7, d0=0., n=6, m=8)
    np.testing.assert_allclose(sf(x), expected)


def test_call_uses_d0_shift():
    x = np.array([[1.1, 1.8], [1.2, 2.0]])
    d0 = 0.5
    sf = SwitchFun(r0=1.7, options={"m": 8, "n": 6, "d0": d0})
    expected = rational(x, r0=1.7, d0=d0, n=6, m=8)
    np.testing.assert_allclose(sf(x), expected)


def test_d0_changes_result():
    x = np.array([[1.1, 1.8], [1.2, 2.0]])
    without = SwitchFun(r0=1.7, options={"m": 8, "n": 6})(x)
    with_d0 = SwitchFun(r0=1.7, options={"m": 8, "n": 6, "d0": 0.5})(x)
    assert not np.allclose(without, with_d0)


def test_value_one_at_d0():
    # At distance == d0 the shifted argument is 0, giving s = 1.
    d0 = 0.5
    sf = SwitchFun(r0=1.7, options={"m": 8, "n": 6, "d0": d0})
    np.testing.assert_allclose(sf(np.array([d0])), np.array([1.]))


def test_one_below_d0():
    # PLUMED convention: s = 1 for r <= d0, also for even powers of a negative shift
    sf = SwitchFun(r0=0.375, options={"m": 12, "n": 6, "d0": 1.5})
    np.testing.assert_allclose(sf(np.array([0.0, 1.2, 1.5])), 1.)


def test_limit_at_d0_plus_r0():
    # (1 - x^n) / (1 - x^m) is 0/0 at x = 1; its limit is n/m
    sf = SwitchFun(r0=0.375, options={"m": 12, "n": 6, "d0": 1.5})
    assert sf(1.875) == 0.5
    assert np.isfinite(sf(np.array([1.875, 2.0]))).all()


def test_m_twice_n_is_one_over_one_plus_x_n():
    r = np.linspace(1.5, 3.0, 50)
    x = (r - 1.5) / 0.375
    sf = SwitchFun(r0=0.375, options={"m": 12, "n": 6, "d0": 1.5})
    np.testing.assert_allclose(sf(r), 1. / (1. + x ** 6))
