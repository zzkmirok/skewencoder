"""ASE-MACE-Skewencoder Workflow Demo.

This script demonstrates the iterative enhanced sampling workflow using:
- MACE foundation model as the interatomic potential (via ASE)
- MACE descriptors (node features) as collective variable inputs
- Skewencoder autoencoder for learning a 1D collective variable
- Harmonic wall bias applied through a custom ASE Calculator
- NVT molecular dynamics via ASE Langevin thermostat

The workflow iterates:
1. Run MD → collect MACE descriptors
2. Train skewencoder CV on descriptors
3. Compute CV statistics (mean, variance, skewness)
4. Apply harmonic wall bias to push sampling toward unexplored regions
5. Repeat with biased MD
"""

import os

import numpy as np
import torch
from ase import units
from ase.calculators.mixing import SumCalculator
from ase.io import read
from ase.md.langevin import Langevin
from mace.calculators import mace_mp
from scipy import stats

from skewencoder.gen_ASE import (
    DescriptorWeights,
    MACEDescriptorExtractor,
    collect_descriptors_along_md,
    create_bias_calculator,
)
from skewencoder.io import create_dataset_from_descriptors
from skewencoder.model_skewencoder import (
    cv_eval,
    skewencoder_model_init,
    skewencoder_model_normalization,
    skewencoder_model_trainer,
)

# ============================================================
# Step 0: Configuration
# ============================================================
# MD parameters
N_ITERATIONS = 5          # Number of bias-train-MD cycles
N_MD_STEPS = 100          # MD steps per iteration (fast demo)
TEMPERATURE_K = 410       # Temperature in Kelvin
TIMESTEP_FS = 0.5         # Timestep in femtoseconds
FRICTION = 0.01           # Langevin friction coefficient (1/fs)
COLLECT_EVERY = 1       # Collect descriptors every N steps

# Skewencoder training parameters
LOSS_COEFF = 0.1          # Weight for skewness loss
BATCH_SIZE = 10           

# Bias parameters
KAPPA = 300.0             # Wall spring constant
OFFSET = 1.0             # Base offset from CV mean for wall placement

# Dispersion correction
# Some MACE foundation models/heads are already trained on dispersion-inclusive
# reference data, in which case adding D3 double-counts dispersion. Set this to
# True only if the chosen head targets a plain (dispersion-free) functional.
USE_D3 = True            # Add explicit D3(BJ) dispersion via torch-dftd

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INIT_FILE = os.path.join(SCRIPT_DIR, "init.traj")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# The results directory holds the unbiased trajectory (below) and the per-
# iteration biased trajectories/checkpoints. Create it up front so nothing is
# silently skipped.
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Step 1: Load starting structure
# ============================================================
# Read the single starting configuration from init.traj. The system is 2 NH3 molecules on a
# Ru(105) surface.
print("Step 1: Loading starting structure...")
atoms = read(INIT_FILE)
print(f"  System: {atoms.get_chemical_formula()}, {len(atoms)} atoms")
print(f"  Cell: {atoms.get_cell().lengths()}")


# ============================================================
# Step 2: Set up MACE calculator
# ============================================================
# Use the MACE-MH-1 multi-head foundation model with the OC20 head, which is
# trained on catalysis data (adsorbates on metal surfaces) and thus suited to
# NH3 on Ru(001). Available heads in mh-1: matpes_r2scan, mp_pbe_refit_add,
# spice_wB97M, oc20_usemppbe, omol, omat_pbe.
# For production use, replace with a fine-tuned model:
#   mace_calc = MACECalculator(model_paths="path/to/model.model", ...)
print("Step 2: Setting up MACE calculator...")
device = "cuda" if torch.cuda.is_available() else "cpu"
mace_calc = mace_mp(model="mh-1", default_dtype="float64", device=device, head="oc20_usemppbe")

# Optionally add D3(BJ) dispersion via torch-dftd. Keep `mace_calc` as the pure
# MACE calculator (the descriptor extractor needs its internal MACE model), and
# sum the D3 correction into a separate potential used to drive the dynamics.
# When USE_D3 is False, the dynamics potential is MACE alone.
if USE_D3:
    # Imported lazily so users who leave USE_D3 disabled don't need torch-dftd.
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    d3_calc = TorchDFTD3Calculator(
        device=device,
        damping="bj",
        dtype=torch.float64,
        xc="pbe",
        cutoff=20.0 * units.Bohr,
    )
    pot_calc = SumCalculator([mace_calc, d3_calc])
else:
    d3_calc = None
    pot_calc = mace_calc

atoms.calc = pot_calc
initial_energy = atoms.get_potential_energy()
print(f"  Device: {device}")
print(f"  D3 dispersion: {'enabled' if USE_D3 else 'disabled'}")
print(f"  Initial energy: {initial_energy:.4f} eV")


# ============================================================
# Step 3: Set up descriptor extractor with configurable weights
# ============================================================
# Configure how per-atom MACE descriptors are aggregated into a system-level
# descriptor vector. Here we use element-based weights to emphasize the
# chemically relevant N and H atoms over the Ru substrate.
print("Step 3: Setting up descriptor extractor...")

# Element-based weights: emphasize N and H (the reacting species)
# Atomic numbers: H=1, N=7, Ru=44
element_weights = {7: 2.0, 1: 1.0, 44: 0.0}  # N > H > Ru(ignored)
weights = DescriptorWeights.from_element_weights(
    element_weights=element_weights,
    n_atoms=len(atoms),
)

# Create the extractor in simple mode (invariants only) for data collection.
# The bias calculator will internally use the differentiable mode.
descriptor_extractor = MACEDescriptorExtractor(
    mace_calc=mace_calc,
    weights=weights,
    invariants_only=True,
    device=device,
)

# Test extraction to determine descriptor dimension.
# Uses get_descriptors_differentiable to verify the full pipeline works.
positions_tensor = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
with torch.no_grad():
    test_desc = descriptor_extractor.get_descriptors_differentiable(atoms, positions_tensor)
n_features = len(test_desc)
print(f"  Descriptor dimension: {n_features}")
print("  Weights mode: element_based (N=2.0, H=1.0, Ru=0.0)")


# ============================================================
# Step 4: Run unbiased NVT MD and collect descriptors
# ============================================================
# Perform a short Langevin dynamics simulation without any bias.
# At regular intervals, extract MACE descriptors from the current frame.
print(f"Step 4: Running unbiased MD ({N_MD_STEPS} steps)...")


def make_dyn(atoms):
    """Build the Langevin NVT integrator from the demo's MD parameters.

    Owning MD setup here keeps the choice of thermostat and its parameters in
    the demo; the library's ``collect_descriptors_along_md`` is thermostat-
    agnostic and just runs whatever dynamics object it is handed.
    """
    return Langevin(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION / units.fs,
    )


unbiased_traj = os.path.join(RESULTS_DIR, "unbiased.traj")
descriptors = collect_descriptors_along_md(
    make_dyn(atoms), descriptor_extractor, N_MD_STEPS,
    collect_every=COLLECT_EVERY,
    trajectory_file=unbiased_traj,
)
print(f"  Collected {descriptors.shape[0]} frames, descriptor shape: {descriptors.shape}")
print(f"  Unbiased trajectory saved to: {unbiased_traj}")


# ============================================================
# Step 5: Define encoder architecture
# ============================================================
# The dataset/datamodule are created inside the training loop from the
# accumulated descriptors, so here we only fix the network architecture:
# input_dim -> hidden layers -> 1D latent (CV)
print("Step 5: Defining encoder architecture...")
encoder_layers = [n_features, 128, 64, 32, 16, 8, 1]
print(f"  Encoder architecture: {encoder_layers}")


# ============================================================
# Step 6: Iterative training loop
# ============================================================
# Each iteration:
#   a) Train the skewencoder on accumulated descriptors
#   b) Evaluate the CV and compute statistics
#   c) Create a harmonic wall bias based on the skewness
#   d) Run biased MD to explore new regions
#   e) Accumulate new descriptors for the next iteration
print(f"\nStep 6: Starting iterative training ({N_ITERATIONS} iterations)...")
print("=" * 60)

# The MultiTaskCV trains on two datasets: the autoencoder task uses all
# accumulated descriptors so the CV stays valid over the whole explored
# landscape, while the skewness task (and the wall-placement statistics) use
# only the most recent MD run, so the bias direction reflects the newest
# sampling.
all_descriptors = descriptors.copy()
latest_descriptors = descriptors.copy()

for iteration in range(N_ITERATIONS):
    iter_folder = os.path.join(RESULTS_DIR, f"iter_{iteration}")
    os.makedirs(iter_folder, exist_ok=True)
    print(f"\n--- Iteration {iteration} ---")

    # 6a: Train skewencoder model
    # The model is an autoencoder with an auxiliary skewness loss that
    # encourages the latent CV to have non-Gaussian (skewed) distribution.
    print("  Training skewencoder...")
    AE_dataset, skew_dataset, datamodule = create_dataset_from_descriptors(
        all_descriptors,
        skew_descriptors=latest_descriptors,
        multiple=iteration + 1,
        batch_size=BATCH_SIZE,
        verbose=False,
    )
    model = skewencoder_model_init(AE_dataset, encoder_layers, LOSS_COEFF)
    metrics = skewencoder_model_trainer(model, datamodule, iter_folder=iter_folder)

    # 6b: Normalize model output to [0, 1] range over all accumulated data
    model = skewencoder_model_normalization(model, AE_dataset)

    # 6c: Evaluate CV on the latest iteration's data and compute statistics
    # The skewness determines which direction to push: positive skewness
    # means data is piled on the left → push rightward (lower wall).
    nn_output = cv_eval(model, skew_dataset).flatten()
    mu = float(np.mean(nn_output))
    var = float(np.var(nn_output))
    skewness = float(stats.skew(nn_output))
    print(f"  CV stats: mean={mu:.4f}, var={var:.4f}, skewness={skewness:.4f}")

    # 6d: Create bias calculator
    # The factory function determines wall type and position from statistics:
    #   offset += sqrt(variance)
    #   skewness < 0 → upper wall at (mean - offset)
    #   skewness >= 0 → lower wall at (mean + offset)
    bias_calc = create_bias_calculator(
        cv_model=model,
        descriptor_extractor=descriptor_extractor,
        cv_mean=mu,
        cv_variance=var,
        cv_skewness=skewness,
        kappa=KAPPA,
        offset=OFFSET,
    )
    wall_type = "LOWER" if bias_calc.is_lower_wall else "UPPER"
    print(f"  Bias: {wall_type}_WALL at AT={bias_calc.wall_position:.4f}, kappa={KAPPA}")

    # 6e: Combine the base potential (MACE, optionally + D3) with the bias using
    # ASE's SumCalculator. All calculators contribute independently to energy
    # and forces.
    base_calcs = [mace_calc, d3_calc] if USE_D3 else [mace_calc]
    combined_calc = SumCalculator(base_calcs + [bias_calc])
    atoms.calc = combined_calc

    # 6f: Run biased MD and collect new descriptors
    # The bias pushes the system toward unexplored CV regions.
    print(f"  Running biased MD ({N_MD_STEPS} steps)...")
    new_descriptors = collect_descriptors_along_md(
        make_dyn(atoms), descriptor_extractor, N_MD_STEPS,
        collect_every=COLLECT_EVERY,
        trajectory_file=os.path.join(iter_folder, "biased.traj"),
    )
    print(f"  Collected {new_descriptors.shape[0]} new frames")

    # 6g: Accumulate descriptors for next iteration
    # The AE task trains on all data from all iterations so the model captures
    # the full explored landscape; the skewness task only sees this run's data.
    all_descriptors = np.vstack([all_descriptors, new_descriptors])
    latest_descriptors = new_descriptors
    print(f"  Total accumulated frames: {all_descriptors.shape[0]}")

    # Restore the unbiased MACE+D3 potential for the next iteration
    atoms.calc = pot_calc


# ============================================================
# Step 7: Final results
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Workflow complete!")
print(f"  Total frames collected: {all_descriptors.shape[0]}")
print(f"  Results saved in: {RESULTS_DIR}")
print(f"  Final model checkpoint: {os.path.join(RESULTS_DIR, f'iter_{N_ITERATIONS-1}', 'checkpoint.ckpt')}")
