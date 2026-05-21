import numpy as np
import torch
import pytest

from skewencoder.gen_ASE import (
    DescriptorWeights,
    MACEDescriptorExtractor,
    SkewencoderBiasCalculator,
    create_bias_calculator,
    _parse_irreps_invariant_indices,
)


def _can_import_mace():
    try:
        import mace
        return True
    except ImportError:
        return False


class TestDescriptorWeights:

    def test_uniform_all_atoms(self):
        weights = DescriptorWeights(n_atoms=4, weight_mode="uniform")
        atomic_numbers = np.array([1, 1, 7, 7])
        w = weights(atomic_numbers)
        assert w.shape == (4,)
        assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(w, torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64))

    def test_uniform_selected_indices(self):
        weights = DescriptorWeights(n_atoms=5, atom_indices=[0, 2, 4], weight_mode="uniform")
        atomic_numbers = np.array([1, 7, 1, 7, 1])
        w = weights(atomic_numbers)
        assert w[1].item() == 0.0
        assert w[3].item() == 0.0
        assert w[0].item() > 0.0
        assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=torch.float64))

    def test_element_based_weights(self):
        element_weights = {7: 2.0, 1: 1.0}
        weights = DescriptorWeights(
            n_atoms=4, weight_mode="element_based", element_weights=element_weights
        )
        atomic_numbers = np.array([1, 1, 7, 7])
        w = weights(atomic_numbers)
        assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=torch.float64))
        # N atoms should have twice the weight of H atoms
        assert w[2].item() == w[3].item()
        assert w[0].item() == w[1].item()
        assert abs(w[2].item() / w[0].item() - 2.0) < 1e-10

    def test_element_based_zero_weight(self):
        element_weights = {7: 1.0, 1: 0.0}
        weights = DescriptorWeights(
            n_atoms=4, weight_mode="element_based", element_weights=element_weights
        )
        atomic_numbers = np.array([1, 1, 7, 7])
        w = weights(atomic_numbers)
        assert w[0].item() == 0.0
        assert w[1].item() == 0.0
        assert w[2].item() > 0.0

    def test_custom_weights(self):
        custom = np.array([0.1, 0.2, 0.3, 0.4])
        weights = DescriptorWeights(n_atoms=4, weight_mode="custom", custom_weights=custom)
        atomic_numbers = np.array([1, 1, 7, 7])
        w = weights(atomic_numbers)
        assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=torch.float64))

    def test_from_element_weights_classmethod(self):
        weights = DescriptorWeights.from_element_weights(
            element_weights={7: 2.0, 1: 1.0}, n_atoms=3
        )
        assert weights.weight_mode == "element_based"
        w = weights(np.array([1, 7, 7]))
        assert torch.allclose(w.sum(), torch.tensor(1.0, dtype=torch.float64))

    def test_from_indices_classmethod(self):
        weights = DescriptorWeights.from_indices(atom_indices=[0, 2], n_atoms=4)
        assert weights.weight_mode == "uniform"
        w = weights(np.array([1, 7, 1, 7]))
        assert w[1].item() == 0.0
        assert w[3].item() == 0.0

    def test_raises_without_custom_weights(self):
        with pytest.raises(ValueError, match="custom_weights required"):
            DescriptorWeights(n_atoms=4, weight_mode="custom")

    def test_raises_without_element_weights(self):
        with pytest.raises(ValueError, match="element_weights required"):
            DescriptorWeights(n_atoms=4, weight_mode="element_based")


class TestCreateBiasCalculator:

    def _make_mock_model(self, output_value=0.5):
        class MockModel(torch.nn.Module):
            def __init__(self, val):
                super().__init__()
                self.val = val

            def forward(self, x):
                return torch.tensor([[self.val]], dtype=torch.float32)

            def eval(self):
                return self

        return MockModel(output_value)

    def _make_mock_extractor(self, n_features=16):
        class MockExtractor:
            device = torch.device("cpu")

            def get_descriptors(self, atoms):
                return np.random.randn(n_features)

            def get_descriptors_differentiable(self, atoms, positions_tensor):
                return torch.randn(n_features, dtype=torch.float64, requires_grad=True)

        return MockExtractor()

    def test_positive_skewness_gives_lower_wall(self):
        model = self._make_mock_model()
        extractor = self._make_mock_extractor()
        calc = create_bias_calculator(
            cv_model=model,
            descriptor_extractor=extractor,
            cv_mean=0.5,
            cv_variance=0.04,
            cv_skewness=1.5,
            kappa=300.0,
            offset=1.0,
        )
        assert calc.is_lower_wall is True
        expected_at = 0.5 + (1.0 + np.sqrt(0.04))
        assert abs(calc.wall_position - expected_at) < 1e-10

    def test_negative_skewness_gives_upper_wall(self):
        model = self._make_mock_model()
        extractor = self._make_mock_extractor()
        calc = create_bias_calculator(
            cv_model=model,
            descriptor_extractor=extractor,
            cv_mean=0.5,
            cv_variance=0.04,
            cv_skewness=-1.5,
            kappa=300.0,
            offset=1.0,
        )
        assert calc.is_lower_wall is False
        expected_at = 0.5 - (1.0 + np.sqrt(0.04))
        assert abs(calc.wall_position - expected_at) < 1e-10

    def test_zero_skewness_gives_lower_wall(self):
        model = self._make_mock_model()
        extractor = self._make_mock_extractor()
        calc = create_bias_calculator(
            cv_model=model,
            descriptor_extractor=extractor,
            cv_mean=0.0,
            cv_variance=1.0,
            cv_skewness=0.0,
            kappa=300.0,
            offset=1.0,
        )
        assert calc.is_lower_wall is True


class TestSkewencoderBiasCalculator:

    def _make_bias_calc(self, wall_position=0.8, kappa=300.0, is_lower_wall=True, cv_output=0.5):
        class MockModel(torch.nn.Module):
            def __init__(self, val):
                super().__init__()
                self.val = val

            def forward(self, x):
                return x.sum() * 0 + self.val

            def eval(self):
                return self

        class MockExtractor:
            device = torch.device("cpu")

            def get_descriptors_differentiable(self, atoms, positions_tensor):
                return positions_tensor.mean(dim=0)[:3].double()

        model = MockModel(cv_output)
        extractor = MockExtractor()

        return SkewencoderBiasCalculator(
            cv_model=model,
            descriptor_extractor=extractor,
            wall_position=wall_position,
            kappa=kappa,
            is_lower_wall=is_lower_wall,
        )

    def _make_atoms(self):
        from ase import Atoms
        return Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    def test_no_bias_when_cv_safe_lower_wall(self):
        # Lower wall at 0.3, CV=0.5 > 0.3 → no bias
        calc = self._make_bias_calc(wall_position=0.3, is_lower_wall=True, cv_output=0.5)
        atoms = self._make_atoms()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        assert energy == 0.0
        assert np.allclose(forces, 0.0)

    def test_no_bias_when_cv_safe_upper_wall(self):
        # Upper wall at 0.8, CV=0.5 < 0.8 → no bias
        calc = self._make_bias_calc(wall_position=0.8, is_lower_wall=False, cv_output=0.5)
        atoms = self._make_atoms()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert energy == 0.0

    def test_energy_when_lower_wall_violated(self):
        # Lower wall at 0.8, CV=0.5 < 0.8 → bias active
        calc = self._make_bias_calc(wall_position=0.8, is_lower_wall=True, cv_output=0.5)
        atoms = self._make_atoms()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        expected = 300.0 * (0.5 - 0.8) ** 2
        assert abs(energy - expected) < 1e-4

    def test_energy_when_upper_wall_violated(self):
        # Upper wall at 0.3, CV=0.5 > 0.3 → bias active
        calc = self._make_bias_calc(wall_position=0.3, is_lower_wall=False, cv_output=0.5)
        atoms = self._make_atoms()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        expected = 300.0 * (0.5 - 0.3) ** 2
        assert abs(energy - expected) < 1e-4


@pytest.mark.skipif(
    not _can_import_mace(), reason="mace-torch not installed"
)
class TestMACEDescriptorExtractor:

    def test_simple_extraction_shape(self):
        from ase import Atoms
        from mace.calculators import mace_mp

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        calc = mace_mp(model="small", default_dtype="float64", device="cpu")
        atoms.calc = calc

        extractor = MACEDescriptorExtractor(mace_calc=calc, invariants_only=True, device="cpu")
        desc = extractor.get_descriptors(atoms)
        assert desc.ndim == 1
        assert desc.shape[0] > 0

    def test_differentiable_extraction_has_grad(self):
        from ase import Atoms
        from mace.calculators import mace_mp

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        calc = mace_mp(model="small", default_dtype="float64", device="cpu")
        atoms.calc = calc

        extractor = MACEDescriptorExtractor(mace_calc=calc, invariants_only=True, device="cpu")
        positions_tensor = torch.tensor(
            atoms.get_positions(), dtype=torch.float64, requires_grad=True
        )
        desc = extractor.get_descriptors_differentiable(atoms, positions_tensor)
        loss = desc.sum()
        loss.backward()
        assert positions_tensor.grad is not None
        assert not torch.all(positions_tensor.grad == 0)

    def test_invariants_only_smaller_dim(self):
        from ase import Atoms
        from mace.calculators import mace_mp

        atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]])
        calc = mace_mp(model="small", default_dtype="float64", device="cpu")
        atoms.calc = calc

        extractor_inv = MACEDescriptorExtractor(mace_calc=calc, invariants_only=True, device="cpu")
        extractor_all = MACEDescriptorExtractor(mace_calc=calc, invariants_only=False, device="cpu")

        desc_inv = extractor_inv.get_descriptors(atoms)
        desc_all = extractor_all.get_descriptors(atoms)
        assert desc_inv.shape[0] <= desc_all.shape[0]
