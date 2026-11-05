"""Tests for skewencoder.contactwall without PLUMED, through stand-ins for the PyCV objects."""
import sys
import types

import numpy as np
import pytest

# stand-in for the module PLUMED embeds into its Python interpreter
stub = types.ModuleType("plumedCommunications")
stub.defaults = types.SimpleNamespace(COMPONENT={"derivative": True, "period": None})
stub.PythonCVInterface = object
sys.modules.setdefault("plumedCommunications", stub)

from skewencoder import contactwall  # noqa: E402

MASS = {s: v[0] for s, v in contactwall.ELEMENTS.items()}

# CH3NH2 + COCl2 complex, N...C = 2.90 A
SYMBOLS = ["C", "Cl", "Cl", "O", "C", "H", "H", "H", "N", "H", "H"]
X = np.array([
    [0.000000, 0.000000, 0.500665],
    [0.000000, 1.453075, -0.482240],
    [0.000000, -1.453075, -0.482240],
    [0.000000, 0.000000, 1.674022],
    [3.494058, -0.899252, 0.553079],
    [4.569315, -1.000497, 0.350579],
    [3.378156, -0.548182, 1.579783],
    [3.048098, -1.893788, 0.499972],
    [2.773284, 0.000000, -0.347213],
    [2.855630, -0.311865, -1.307352],
    [3.161441, 0.934683, -0.307081],
])


class NoPbc:
    def apply(self, d):
        return d


class OrthoPbc:
    def __init__(self, box):
        self.box = np.asarray(box, dtype=float)

    def apply(self, d):
        return d - self.box * np.round(d / self.box)


class FakeAction:
    def __init__(self, x, symbols, pbc=None):
        self.x = x
        self._masses = np.array([MASS[s] for s in symbols])
        self._pbc = pbc if pbc is not None else NoPbc()
        self.data = {}

    def getPositions(self):
        return self.x

    def masses(self):
        return self._masses

    def getPbc(self):
        return self._pbc


def calc(x, symbols, pbc=None):
    return contactwall.plumedCalculate(FakeAction(x, symbols, pbc))


def numerical_gradient(x, symbols, pbc=None, h=1e-6):
    fd = np.zeros_like(x)
    for k in range(len(x)):
        for c in range(3):
            xp, xm = x.copy(), x.copy()
            xp[k, c] += h
            xm[k, c] -= h
            fd[k, c] = (calc(xp, symbols, pbc)["gap"][0] - calc(xm, symbols, pbc)["gap"][0]) / (2 * h)
    return fd


def true_min_gap(x, symbols, ia, ib):
    rvdw = np.array([contactwall.ELEMENTS[s][2] for s in symbols])
    r = np.linalg.norm(x[ia][:, None] - x[ib][None], axis=-1)
    return (r - rvdw[ia][:, None] - rvdw[ib][None]).min()


def test_two_fragments():
    out = calc(X, SYMBOLS)
    gap, grad, box = out["gap"]
    assert out["nfrag"][0] == 2
    # closest contact is N...C: 2.90 - (1.55 + 1.70)
    expected = true_min_gap(X, SYMBOLS, np.arange(4), np.arange(4, 11))
    assert expected == pytest.approx(-0.35, abs=1e-3)
    assert expected - np.log(4 * 7) / contactwall.SOFTMIN_LAMBDA <= gap <= expected
    np.testing.assert_allclose(grad, numerical_gradient(X, SYMBOLS), atol=1e-8)
    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(box, -np.einsum("ij,ik->jk", X, grad), atol=1e-12)


def test_distant_fragment_is_the_bottleneck():
    symbols = SYMBOLS + ["H", "Cl"]
    far = X.max(axis=0) + np.array([6.0, 0.0, 0.0])
    x = np.vstack([X, far, far + [1.275, 0.0, 0.0]])
    out = calc(x, symbols)
    assert out["nfrag"][0] == 3
    expected = true_min_gap(x, symbols, np.arange(11), np.array([11, 12]))
    assert out["gap"][0] == pytest.approx(expected, abs=0.05)
    assert out["gap"][0] > 2.0
    np.testing.assert_allclose(out["gap"][1], numerical_gradient(x, symbols), atol=1e-8)


def test_one_molecule_switches_off():
    x = X.copy()
    c, n = x[0], x[8]
    x[4:] += (c + 1.47 * (n - c) / np.linalg.norm(n - c)) - n
    out = calc(x, SYMBOLS)
    assert out["nfrag"][0] == 1
    assert out["gap"][0] == 0.0
    assert not out["gap"][1].any()


def test_minimum_image():
    box = np.array([12.0, 12.0, 12.0])
    pbc = OrthoPbc(box)
    reference = calc(X, SYMBOLS, pbc)
    # move the methylamine by one box vector: same system under periodic boundaries
    x = X.copy()
    x[4:] += [box[0], 0.0, 0.0]
    out = calc(x, SYMBOLS, pbc)
    assert out["nfrag"][0] == 2
    assert out["gap"][0] == pytest.approx(reference["gap"][0], abs=1e-12)
    np.testing.assert_allclose(out["gap"][1], reference["gap"][1], atol=1e-12)
    np.testing.assert_allclose(out["gap"][2], reference["gap"][2], atol=1e-12)
    np.testing.assert_allclose(out["gap"][1], numerical_gradient(x, SYMBOLS, pbc), atol=1e-8)


def test_element_assignment():
    masses = np.array([12.011, 35.45, 2.014, 15.999])  # common rounded masses and a deuterium
    assert contactwall.element_symbols(masses) == ["C", "Cl", "H", "O"]
    with pytest.raises(ValueError, match="mass 101.07"):
        contactwall.element_symbols(np.array([12.011, 101.07]))  # Ru has no vdW radius here


def test_element_symbols_atomic_units():
    # DFTB+ passes masses in electron masses; values from a real DFTB+/PLUMED run
    amu = np.array([12.0107, 35.453, 1.00794, 15.9994])
    assert contactwall.element_symbols(amu * contactwall.AMU_IN_ELECTRON_MASSES) == ["C", "Cl", "H", "O"]
    assert contactwall.element_symbols(np.array([21894.23216763776]))[0] == "C"
