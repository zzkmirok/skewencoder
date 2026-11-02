"""Element data shared by the geometry parser and the PyCV contact wall.

Covalent radii and standard atomic weights are copied from chemtrayzer.core.periodic_table;
the bond criterion is chemtrayzer's default (reaction_sampling.atomutils.default_r0_func with
StepSwitchingFunction): atoms i, j are bonded if r_ij < 1.2 * (Rcov_i + Rcov_j).
chemtrayzer has no van der Waals radii, so VDW_RADII holds Bondi's values.
"""

__all__ = ["COVALENT_RADII", "ATOMIC_WEIGHTS", "VDW_RADII", "BOND_SCALE", "normalize_symbol", "bond_cutoff"]

BOND_SCALE = 1.2

# Angstrom; Pyykko & Atsumi, Chem. Eur. J. 2009, 15, 12770 (via chemtrayzer)
COVALENT_RADII: dict[str, float] = {
    'H': 0.32, 'He': 0.46, 'Li': 1.33, 'Be': 1.02, 'B': 0.85, 'C': 0.75, 'N': 0.71, 'O': 0.63,
    'F': 0.64, 'Ne': 0.67, 'Na': 1.55, 'Mg': 1.39, 'Al': 1.26, 'Si': 1.16, 'P': 1.11, 'S': 1.03,
    'Cl': 0.99, 'Ar': 0.96, 'K': 1.96, 'Ca': 1.71, 'Sc': 1.48, 'Ti': 1.36, 'V': 1.34, 'Cr': 1.22,
    'Mn': 1.19, 'Fe': 1.16, 'Co': 1.11, 'Ni': 1.1, 'Cu': 1.12, 'Zn': 1.18, 'Ga': 1.24, 'Ge': 1.21,
    'As': 1.21, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.17, 'Rb': 2.1, 'Sr': 1.85, 'Y': 1.63, 'Zr': 1.54,
    'Nb': 1.47, 'Mo': 1.38, 'Tc': 1.28, 'Ru': 1.25, 'Rh': 1.25, 'Pd': 1.2, 'Ag': 1.28, 'Cd': 1.36,
    'In': 1.42, 'Sn': 1.4, 'Sb': 1.4, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31, 'Cs': 2.32, 'Ba': 1.96,
    'La': 1.8, 'Ce': 1.63, 'Pr': 1.76, 'Nd': 1.74, 'Pm': 1.73, 'Sm': 1.72, 'Eu': 1.68, 'Gd': 1.69,
    'Tb': 1.68, 'Dy': 1.67, 'Ho': 1.66, 'Er': 1.65, 'Tm': 1.64, 'Yb': 1.7, 'Lu': 1.62, 'Hf': 1.52,
    'Ta': 1.46, 'W': 1.37, 'Re': 1.31, 'Os': 1.29, 'Ir': 1.22, 'Pt': 1.23, 'Au': 1.24, 'Hg': 1.33,
    'Tl': 1.44, 'Pb': 1.44, 'Bi': 1.51, 'Po': 1.45, 'At': 1.47, 'Rn': 1.42, 'Fr': 2.23, 'Ra': 2.01,
    'Ac': 1.86, 'Th': 1.75, 'Pa': 1.69, 'U': 1.7, 'Np': 1.71, 'Pu': 1.72, 'Am': 1.66, 'Cm': 1.66,
    'Bk': 1.68, 'Cf': 1.68, 'Es': 1.65, 'Fm': 1.67, 'Md': 1.73, 'No': 1.76, 'Lr': 1.61, 'Rf': 1.57,
    'Db': 1.49, 'Sg': 1.43, 'Bh': 1.41, 'Hs': 1.34, 'Mt': 1.29, 'Ds': 1.28, 'Rg': 1.21, 'Cn': 1.22,
    'Nh': 1.36, 'Fl': 1.43, 'Mc': 1.62, 'Lv': 1.75, 'Ts': 1.65, 'Og': 1.57,
}

# amu (via chemtrayzer)
ATOMIC_WEIGHTS: dict[str, float] = {
    'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107,
    'N': 14.0067, 'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305,
    'Al': 26.9815386, 'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
    'K': 39.0983, 'Ca': 40.078, 'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961,
    'Mn': 54.938045, 'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409,
    'Ga': 69.723, 'Ge': 72.64, 'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798,
    'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585, 'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94,
    'Tc': 98.9063, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411,
    'In': 114.818, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6, 'I': 126.90447, 'Xe': 131.293,
    'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116, 'Pr': 140.90465, 'Nd': 144.242,
    'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.92535, 'Dy': 162.5,
    'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04, 'Lu': 174.967, 'Hf': 178.49,
    'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.084,
    'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804, 'Po': 208.9824,
    'At': 209.9871, 'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278, 'Th': 232.03806,
    'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614, 'Cm': 247.0703,
    'Bk': 247.0703, 'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951, 'No': 259.1009,
    'Lr': 262, 'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269,
    'Mt': 278, 'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289,
    'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294,
}

# Angstrom; Bondi, J. Phys. Chem. 1964, 68, 441
VDW_RADII: dict[str, float] = {
    "H": 1.20, "He": 1.40, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54, "Si": 2.10,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88, "Br": 1.85, "Kr": 2.02, "I": 1.98, "Xe": 2.16,
}


def normalize_symbol(symbol: str) -> str:
    """'CL', 'cl' -> 'Cl'."""
    symbol = symbol.strip()
    normalized = symbol[:1].upper() + symbol[1:].lower()
    if normalized not in COVALENT_RADII:
        raise ValueError(f"unknown element symbol {symbol!r}")
    return normalized


def bond_cutoff(symbol_a: str, symbol_b: str, bond_scale: float = BOND_SCALE) -> float:
    """Distance (A) below which two atoms count as bonded."""
    return bond_scale * (COVALENT_RADII[normalize_symbol(symbol_a)] + COVALENT_RADII[normalize_symbol(symbol_b)])
