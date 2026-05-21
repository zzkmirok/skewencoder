"""ASE-based bias calculator and MACE descriptor extraction for skewencoder.

Provides tools to:
1. Extract MACE descriptors (simple and differentiable modes)
2. Aggregate per-atom descriptors via configurable weights
3. Apply harmonic wall bias based on a trained skewencoder CV
"""

import numpy as np
import torch
from typing import Optional, Union

from ase.calculators.calculator import Calculator, all_changes

__all__ = [
    "DescriptorWeights",
    "MACEDescriptorExtractor",
    "SkewencoderBiasCalculator",
    "create_bias_calculator",
]


class DescriptorWeights:
    """Template class for generating per-atom weights for descriptor aggregation.

    Controls how per-atom MACE descriptors are weighted before summation into
    a system-level descriptor vector. Supports uniform, element-based, and
    custom weighting schemes.

    Parameters
    ----------
    n_atoms : int
        Total number of atoms in the system.
    atom_indices : list[int], optional
        Indices of atoms to include. None means all atoms.
    weight_mode : str
        Weighting scheme: "uniform", "element_based", or "custom".
    custom_weights : np.ndarray, optional
        Explicit per-atom weights array of shape (n_atoms,). Required when
        weight_mode is "custom".
    element_weights : dict, optional
        Mapping from atomic number to weight, e.g. {7: 2.0, 1: 1.0}.
        Required when weight_mode is "element_based".
    """

    def __init__(
        self,
        n_atoms: int,
        atom_indices: Optional[list] = None,
        weight_mode: str = "uniform",
        custom_weights: Optional[np.ndarray] = None,
        element_weights: Optional[dict] = None,
    ):
        self.n_atoms = n_atoms
        self.atom_indices = atom_indices
        self.weight_mode = weight_mode
        self.custom_weights = custom_weights
        self.element_weights = element_weights

        if weight_mode == "custom" and custom_weights is None:
            raise ValueError("custom_weights required when weight_mode='custom'")
        if weight_mode == "element_based" and element_weights is None:
            raise ValueError("element_weights required when weight_mode='element_based'")

    def __call__(self, atomic_numbers: np.ndarray) -> torch.Tensor:
        """Compute normalized per-atom weights.

        Parameters
        ----------
        atomic_numbers : np.ndarray
            Array of atomic numbers for each atom, shape (n_atoms,).

        Returns
        -------
        torch.Tensor
            Normalized weights of shape (n_atoms,) that sum to 1.0.
            Atoms not in atom_indices (if specified) get weight 0.
        """
        n = len(atomic_numbers)
        weights = np.zeros(n, dtype=np.float64)

        if self.weight_mode == "uniform":
            if self.atom_indices is not None:
                weights[self.atom_indices] = 1.0
            else:
                weights[:] = 1.0

        elif self.weight_mode == "element_based":
            for i, z in enumerate(atomic_numbers):
                if self.atom_indices is not None and i not in self.atom_indices:
                    continue
                weights[i] = self.element_weights.get(int(z), 0.0)

        elif self.weight_mode == "custom":
            weights = self.custom_weights.copy()
            if self.atom_indices is not None:
                mask = np.zeros(n, dtype=bool)
                mask[self.atom_indices] = True
                weights[~mask] = 0.0

        total = weights.sum()
        if total > 0:
            weights /= total

        return torch.tensor(weights, dtype=torch.float64)

    @classmethod
    def from_element_weights(cls, element_weights: dict, n_atoms: int, atom_indices=None):
        """Create weights from element-to-weight mapping.

        Parameters
        ----------
        element_weights : dict
            Mapping from atomic number to raw weight, e.g. {7: 2.0, 1: 1.0, 44: 0.0}.
        n_atoms : int
            Total number of atoms.
        atom_indices : list[int], optional
            Restrict to these atom indices.

        Returns
        -------
        DescriptorWeights
            Configured instance.
        """
        return cls(
            n_atoms=n_atoms,
            atom_indices=atom_indices,
            weight_mode="element_based",
            element_weights=element_weights,
        )

    @classmethod
    def from_indices(cls, atom_indices: list, n_atoms: int):
        """Create uniform weights restricted to specific atom indices.

        Parameters
        ----------
        atom_indices : list[int]
            Atom indices to include (uniform weight among them).
        n_atoms : int
            Total number of atoms.

        Returns
        -------
        DescriptorWeights
            Configured instance with uniform weights on selected atoms.
        """
        return cls(n_atoms=n_atoms, atom_indices=atom_indices, weight_mode="uniform")


def _parse_irreps_invariant_indices(irreps_out):
    """Parse irreps_out to find indices of invariant (0e) channels.

    Parameters
    ----------
    irreps_out : o3.Irreps or similar
        The irreps specification from model.products[0].linear.irreps_out.

    Returns
    -------
    invariant_indices : list[int]
        Indices into the feature dimension that correspond to l=0 even parity.
    total_dim : int
        Total feature dimension.
    """
    invariant_indices = []
    idx = 0
    for mul, ir in irreps_out:
        dim = mul * ir.dim
        if ir.l == 0 and ir.p == 1:
            invariant_indices.extend(range(idx, idx + dim))
        idx += dim
    return invariant_indices, idx


class MACEDescriptorExtractor:
    """Extract MACE descriptors from atomic configurations.

    Supports two extraction modes:

    1. **Simple mode** (``get_descriptors``): Uses the MACECalculator's
       built-in ``get_descriptors(invariants_only=...)`` method. Fast, returns
       numpy arrays. Suitable for collecting training data during MD.

    2. **Differentiable mode** (``get_descriptors_differentiable``): Runs the
       internal MACE torch model with positions as a differentiable tensor.
       Gradients flow back to positions via autograd. Used by the bias
       calculator for force computation.

    Parameters
    ----------
    mace_calc : MACECalculator
        The MACE ASE calculator (from ``mace_mp()`` or ``MACECalculator()``).
    weights : DescriptorWeights, optional
        Aggregation weights. If None, uses uniform weights over all atoms.
    invariants_only : bool
        Whether to extract only invariant (l=0, even parity) features.
        Applies to both simple and differentiable modes.
    device : str
        Torch device for differentiable mode. Default: auto-detect from model.
    """

    def __init__(
        self,
        mace_calc,
        weights: Optional[DescriptorWeights] = None,
        invariants_only: bool = True,
        device: str = "",
    ):
        self.mace_calc = mace_calc
        self.weights = weights
        self.invariants_only = invariants_only

        self.model = mace_calc.models[0]
        if device:
            self.device = torch.device(device)
        else:
            self.device = next(self.model.parameters()).device

        self._invariant_indices = None
        self._total_dim = None
        self._setup_irreps()

    def _setup_irreps(self):
        """Inspect model irreps to determine invariant channel indices."""
        try:
            irreps_out = self.model.products[0].linear.__dict__.get("irreps_out", None)
            if irreps_out is None:
                irreps_out = self.model.products[0].linear.irreps_out
            self._invariant_indices, self._total_dim = _parse_irreps_invariant_indices(irreps_out)
        except (AttributeError, IndexError):
            self._invariant_indices = None
            self._total_dim = None

    def get_descriptors(self, atoms) -> np.ndarray:
        """Extract descriptors via the ASE calculator interface (simple mode).

        Calls ``atoms.calc.get_descriptors(atoms, invariants_only=...)`` and
        applies weighted aggregation over atoms.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic configuration. Must have the MACE calculator attached.

        Returns
        -------
        np.ndarray
            Aggregated descriptor vector of shape (n_features,).
        """
        per_atom_desc = self.mace_calc.get_descriptors(
            atoms, invariants_only=self.invariants_only
        )

        atomic_numbers = atoms.get_atomic_numbers()
        if self.weights is not None:
            w = self.weights(atomic_numbers).numpy()
        else:
            w = np.ones(len(atoms)) / len(atoms)

        aggregated = np.einsum("i,ij->j", w, per_atom_desc)
        return aggregated

    def get_descriptors_differentiable(self, atoms, positions_tensor: torch.Tensor) -> torch.Tensor:
        """Extract descriptors via internal MACE model (differentiable mode).

        Runs the MACE model's forward pass with ``positions_tensor`` as input,
        preserving the computational graph for autograd. Applies weighted
        aggregation and optional invariant channel selection.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic configuration (used for cell, pbc, atomic numbers).
        positions_tensor : torch.Tensor
            Positions tensor with ``requires_grad=True``, shape (n_atoms, 3).

        Returns
        -------
        torch.Tensor
            Aggregated descriptor vector of shape (n_features,) with gradient
            graph intact for backpropagation to positions.
        """
        from mace.tools.utils import get_atomic_number_table_from_zs
        from mace.data import AtomicData
        from mace.data.utils import config_from_atoms
        from copy import deepcopy

        atomic_numbers = atoms.get_atomic_numbers()

        if hasattr(self.model, "atomic_numbers") and self.model.atomic_numbers is not None:
            zs_list = self.model.atomic_numbers.tolist()
        else:
            zs_list = list(set(atomic_numbers.tolist()))
        z_table = get_atomic_number_table_from_zs(zs_list)

        r_max = float(self.model.r_max)

        atoms_copy = deepcopy(atoms)
        config = config_from_atoms(atoms=atoms_copy)
        data = AtomicData.from_config(config, z_table=z_table, cutoff=r_max)

        batch_dict = data.to_dict()

        if "head" in batch_dict and torch.is_tensor(batch_dict["head"]):
            if batch_dict["head"].dim() == 0:
                batch_dict["head"] = batch_dict["head"].unsqueeze(0)

        num_atoms = len(atoms)
        batch_dict["batch"] = torch.zeros(num_atoms, dtype=torch.long, device=self.device)
        batch_dict["ptr"] = torch.tensor([0, num_atoms], dtype=torch.long, device=self.device)

        model_dtype = next(self.model.parameters()).dtype
        for k, v in batch_dict.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    batch_dict[k] = v.to(device=self.device, dtype=model_dtype)
                else:
                    batch_dict[k] = v.to(device=self.device)

        batch_dict["positions"] = positions_tensor.to(dtype=model_dtype)

        output = self.model(batch_dict, training=False, compute_force=False)

        node_feats = output["node_feats"]

        if self.invariants_only and self._invariant_indices is not None:
            node_feats = node_feats[:, self._invariant_indices]

        if self.weights is not None:
            w = self.weights(atomic_numbers).to(device=self.device, dtype=node_feats.dtype)
        else:
            w = torch.ones(len(atoms), device=self.device, dtype=node_feats.dtype) / len(atoms)

        aggregated = torch.einsum("i,ij->j", w, node_feats)
        return aggregated


class SkewencoderBiasCalculator(Calculator):
    """ASE Calculator applying harmonic wall bias from a trained skewencoder CV.

    Computes energy and forces via torch autograd through the full chain:
    atomic positions -> MACE descriptors -> CV model -> harmonic wall energy.

    The harmonic wall energy is::

        E_wall = kappa * (cv - AT)^2    when the wall condition is violated
        E_wall = 0                       otherwise

    For a lower wall (penalizes cv < AT): active when cv < AT.
    For an upper wall (penalizes cv > AT): active when cv > AT.

    Parameters
    ----------
    cv_model : torch.nn.Module
        Trained skewencoder model (e.g., MultiTaskCV). Must accept input of
        shape (1, n_features) and return a scalar CV value.
    descriptor_extractor : MACEDescriptorExtractor
        Extracts differentiable MACE descriptors from atoms.
    wall_position : float
        The AT value (wall threshold position).
    kappa : float
        Spring constant (stiffness) for the harmonic wall.
    is_lower_wall : bool
        If True, wall penalizes cv < AT (pushes CV upward).
        If False, wall penalizes cv > AT (pushes CV downward).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        cv_model,
        descriptor_extractor: MACEDescriptorExtractor,
        wall_position: float,
        kappa: float,
        is_lower_wall: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cv_model = cv_model
        self.descriptor_extractor = descriptor_extractor
        self.wall_position = wall_position
        self.kappa = kappa
        self.is_lower_wall = is_lower_wall

        self.cv_model.eval()

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        """Compute bias energy and forces.

        Parameters
        ----------
        atoms : ase.Atoms
            Current atomic configuration.
        properties : list[str]
            Properties to compute (always ["energy", "forces"]).
        system_changes : list
            What changed since last calculation.
        """
        super().calculate(atoms, properties, system_changes)

        positions = atoms.get_positions()
        positions_tensor = torch.tensor(
            positions,
            dtype=torch.float64,
            device=self.descriptor_extractor.device,
            requires_grad=True,
        )

        descriptors = self.descriptor_extractor.get_descriptors_differentiable(
            atoms, positions_tensor
        )

        desc_input = descriptors.unsqueeze(0).float()
        with torch.enable_grad():
            cv_value = self.cv_model(desc_input)

        cv_scalar = cv_value.squeeze()
        at = self.wall_position

        if self.is_lower_wall:
            wall_active = cv_scalar < at
        else:
            wall_active = cv_scalar > at

        if wall_active:
            energy = self.kappa * (cv_scalar - at) ** 2
            energy.backward()
            if positions_tensor.grad is not None:
                forces = -positions_tensor.grad.detach().cpu().numpy()
            else:
                forces = np.zeros_like(positions)
            self.results["energy"] = energy.item()
            self.results["forces"] = forces
        else:
            self.results["energy"] = 0.0
            self.results["forces"] = np.zeros_like(positions)


def create_bias_calculator(
    cv_model,
    descriptor_extractor: MACEDescriptorExtractor,
    cv_mean: float,
    cv_variance: float,
    cv_skewness: float,
    kappa: float = 300.0,
    offset: float = 1.0,
) -> SkewencoderBiasCalculator:
    """Create a bias calculator from CV statistics.

    Determines the wall type and position from the CV mean, variance, and
    skewness, following the logic from the iterative enhanced sampling workflow:

    1. ``offset += sqrt(cv_variance)``
    2. If ``skewness < 0``: upper wall at ``AT = cv_mean - offset``
    3. If ``skewness >= 0``: lower wall at ``AT = cv_mean + offset``

    A lower wall penalizes ``cv < AT`` (pushes CV upward toward unexplored
    regions). An upper wall penalizes ``cv > AT`` (pushes CV downward).

    Parameters
    ----------
    cv_model : torch.nn.Module
        Trained and normalized skewencoder model.
    descriptor_extractor : MACEDescriptorExtractor
        Extracts differentiable MACE descriptors.
    cv_mean : float
        Mean of the CV evaluated on training data.
    cv_variance : float
        Variance of the CV evaluated on training data.
    cv_skewness : float
        Skewness of the CV evaluated on training data.
    kappa : float
        Wall spring constant. Default: 300.0.
    offset : float
        Base offset from the mean. Default: 1.0. The final offset is
        ``offset + sqrt(cv_variance)``.

    Returns
    -------
    SkewencoderBiasCalculator
        Configured bias calculator ready to be combined with a MACE calculator
        via ASE's SumCalculator.
    """
    offset += np.sqrt(cv_variance)

    if cv_skewness < 0:
        is_lower_wall = False
        wall_position = cv_mean - offset
    else:
        is_lower_wall = True
        wall_position = cv_mean + offset

    return SkewencoderBiasCalculator(
        cv_model=cv_model,
        descriptor_extractor=descriptor_extractor,
        wall_position=wall_position,
        kappa=kappa,
        is_lower_wall=is_lower_wall,
    )
