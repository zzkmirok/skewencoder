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
