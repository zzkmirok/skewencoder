"""PyCV collective variable that keeps all molecular fragments in van der Waals contact.

Load it in PLUMED with
    PYCVINTERFACE IMPORT=skewencoder.contactwall INIT=plumedInit CALCULATE=plumedCalculate
and put an UPPER_WALLS on the ``gap`` component (see PlumedInput.gen_plumed_vdw_constr_snippet).

Fragments are re-detected at every step from a covalent-bond graph, so the CV follows bond
formation and breaking. The quantity measured between two atoms is their distance beyond van der
Waals contact, r_ij - (R_vdw_i + R_vdw_j).

Components
  gap    bottleneck gap of the fragment graph: the longest edge of the minimum spanning tree whose
         edge weights are smooth minimum gaps between fragments. With two fragments it is simply
         their smooth minimum gap. gap <= 0 means every fragment touches the rest through a chain of
         van der Waals contacts. 0 when everything is one molecule.
  nfrag  number of fragments (zero gradient, for monitoring only)

Why the wall stays smooth although fragments change discretely: when a bond forms or breaks, the
atoms involved are within the bond cutoff (about 2 A), far inside van der Waals contact (about 3 A),
so gap < 0 and a wall at AT >= 0 exerts no force on either side of the switch.

Lengths are in Angstrom, so the PLUMED input needs UNITS LENGTH=A. Elements are assigned from atomic
masses, so the MD engine (or ``plumed driver --mc``) must provide standard masses.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

import plumedCommunications as PLMD

from .elements import ATOMIC_WEIGHTS, BOND_SCALE, COVALENT_RADII, VDW_RADII

SOFTMIN_LAMBDA = 50.0  # 1/A; the smooth minimum underestimates the true one by at most ln(npairs)/SOFTMIN_LAMBDA

# symbol: (standard atomic weight in amu, covalent radius in A, van der Waals radius in A), see elements.py.
# Only elements with a van der Waals radius are listed, so mass lookup cannot hit an unsupported one.
ELEMENTS = {s: (ATOMIC_WEIGHTS[s], COVALENT_RADII[s], r) for s, r in VDW_RADII.items()}
MASS_TOLERANCE = 0.5  # amu
H_ISOTOPE_MAX_MASS = 3.1  # amu; D and T are treated as H


def element_symbols(masses):
    """Assign element symbols from atomic masses (amu)."""
    names = list(ELEMENTS)
    ref = np.array([ELEMENTS[e][0] for e in names])
    idx = np.abs(masses[:, None] - ref[None, :]).argmin(axis=1)
    symbols = [names[k] for k in idx]
    for i, m in enumerate(masses):
        if m <= H_ISOTOPE_MAX_MASS:
            symbols[i] = "H"
        elif abs(m - ref[idx[i]]) > MASS_TOLERANCE:
            raise ValueError(
                f"atom {i + 1} has mass {m} amu, which matches no element in "
                f"skewencoder.contactwall.ELEMENTS ({', '.join(names)}); add it there"
            )
    return symbols


def atom_radii(symbols):
    """Covalent and van der Waals radii (A) of each atom."""
    rcov = np.array([ELEMENTS[s][1] for s in symbols])
    rvdw = np.array([ELEMENTS[s][2] for s in symbols])
    return rcov, rvdw


def pair_vectors(x, pbc=None):
    """d[i, j] = x_i - x_j under the minimum image convention (if pbc is given) and r[i, j] = |d[i, j]|."""
    d = x[:, None, :] - x[None, :, :]
    if pbc is not None:
        n = len(x)
        d = np.asarray(pbc.apply(d.reshape(n * n, 3))).reshape(n, n, 3)
    return d, np.linalg.norm(d, axis=-1)


def fragments(r, rcov):
    """Number of fragments and fragment label of each atom."""
    bonded = r < BOND_SCALE * (rcov[:, None] + rcov[None, :])
    np.fill_diagonal(bonded, False)
    return connected_components(bonded, directed=False)


def smooth_min_gap(d, r, rvdw, ia, ib):
    """Smooth minimum of r_ij - (R_vdw_i + R_vdw_j) over i in ia, j in ib, and its derivative.

    s = -ln(sum_ij exp(-lambda g_ij)) / lambda. Returns s and ds/dd_ij for the selected pairs,
    where d_ij = x_i - x_j.
    """
    dsel = d[np.ix_(ia, ib)]
    rsel = r[np.ix_(ia, ib)]
    gap = rsel - (rvdw[ia][:, None] + rvdw[ib][None, :])
    a = -SOFTMIN_LAMBDA * gap
    amax = a.max()
    lnZ = amax + np.log(np.exp(a - amax).sum())  # log-sum-exp, no overflow
    s = -lnZ / SOFTMIN_LAMBDA
    w = np.exp(a - lnZ)  # ds/dr_ij
    ds_dd = (w / rsel)[..., None] * dsel  # dr_ij/dd_ij = d_ij / r_ij
    return s, ds_dd


def fragment_bottleneck(x, rcov, rvdw, pbc=None):
    """Return (gap, dgap/dx, box derivative, number of fragments)."""
    d, r = pair_vectors(x, pbc)
    nfrag, labels = fragments(r, rcov)
    if nfrag == 1:
        return 0.0, np.zeros_like(x), np.zeros((3, 3)), 1
    members = [np.flatnonzero(labels == f) for f in range(nfrag)]
    pairs = [(a, b) for a in range(nfrag) for b in range(a + 1, nfrag)]
    gaps = {p: smooth_min_gap(d, r, rvdw, members[p[0]], members[p[1]])[0] for p in pairs}
    # scipy treats zero weights as missing edges, so shift all weights to be positive
    shift = 1.0 - min(gaps.values())
    W = np.zeros((nfrag, nfrag))
    for (a, b), g in gaps.items():
        W[a, b] = g + shift
    mst = minimum_spanning_tree(W).tocoo()
    k = mst.data.argmax()
    ia, ib = members[mst.row[k]], members[mst.col[k]]
    s, ds_dd = smooth_min_gap(d, r, rvdw, ia, ib)
    grad = np.zeros_like(x)
    grad[ia] += ds_dd.sum(axis=1)
    grad[ib] -= ds_dd.sum(axis=0)
    # PLUMED convention, as in DISTANCE: box derivative = -sum_ij d_ij (x) ds/dd_ij
    box = -np.einsum("ija,ijb->ab", d[np.ix_(ia, ib)], ds_dd)
    return s, grad, box, nfrag


def plumedInit(action: PLMD.PythonCVInterface):
    return {
        "COMPONENTS": {
            "gap": PLMD.defaults.COMPONENT,
            "nfrag": PLMD.defaults.COMPONENT,
        }
    }


def plumedCalculate(action: PLMD.PythonCVInterface):
    if "radii" not in action.data:
        symbols = element_symbols(np.asarray(action.masses()))
        action.data["radii"] = atom_radii(symbols)
    rcov, rvdw = action.data["radii"]
    x = np.asarray(action.getPositions())
    gap, grad, box, nfrag = fragment_bottleneck(x, rcov, rvdw, action.getPbc())
    return {
        "gap": (gap, grad, box),
        "nfrag": (float(nfrag), np.zeros_like(x), np.zeros((3, 3))),
    }
