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
from scipy import stats

from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase import units
from ase.calculators.mixing import SumCalculator

from mace.calculators import mace_mp

from skewencoder.io import create_dataset_from_descriptors
from skewencoder.model_skewencoder import (
    skewencoder_model_init,
    skewencoder_model_trainer,
    skewencoder_model_normalization,
    cv_eval,
)
from skewencoder.gen_ASE import (
    DescriptorWeights,
    MACEDescriptorExtractor,
    create_bias_calculator,
)

# ============================================================
# Step 0: Configuration
# ============================================================
# MD parameters
N_ITERATIONS = 5          # Number of bias-train-MD cycles
N_MD_STEPS = 100          # MD steps per iteration (fast demo)
TEMPERATURE_K = 410       # Temperature in Kelvin
TIMESTEP_FS = 1.0         # Timestep in femtoseconds
FRICTION = 0.01           # Langevin friction coefficient (1/fs)
COLLECT_EVERY = 10        # Collect descriptors every N steps

# Skewencoder training parameters
LOSS_COEFF = 0.1          # Weight for skewness loss
BATCH_SIZE = 0            # 0 = full batch training

# Bias parameters
KAPPA = 300.0             # Wall spring constant
OFFSET = 1.0             # Base offset from CV mean for wall placement

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJ_FILE = os.path.join(SCRIPT_DIR, "2NH3_CN_FT_410_912653.traj")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


# ============================================================
# Step 1: Load starting structure
# ============================================================
# Read the last frame from the test trajectory as our starting configuration.
# This trajectory contains 2 NH3 molecules on a Ru(001) surface.
print("Step 1: Loading starting structure...")
atoms = read(TRAJ_FILE, index=-1)
print(f"  System: {atoms.get_chemical_formula()}, {len(atoms)} atoms")
print(f"  Cell: {atoms.get_cell().lengths()}")


# ============================================================
# Step 2: Set up MACE calculator
# ============================================================
# Use the MACE-MP foundation model (small variant for fast demo).
# For production use, replace with a fine-tuned model:
#   mace_calc = MACECalculator(model_paths="path/to/model.model", ...)
print("Step 2: Setting up MACE calculator...")
device = "cuda" if torch.cuda.is_available() else "cpu"
mace_calc = mace_mp(model="small", default_dtype="float64", device=device)
atoms.calc = mace_calc
initial_energy = atoms.get_potential_energy()
print(f"  Device: {device}")
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
print(f"  Weights mode: element_based (N=2.0, H=1.0, Ru=0.0)")


# ============================================================
# Step 4: Run unbiased NVT MD and collect descriptors
# ============================================================
# Perform a short Langevin dynamics simulation without any bias.
# At regular intervals, extract MACE descriptors from the current frame.
print(f"Step 4: Running unbiased MD ({N_MD_STEPS} steps)...")


def run_md_collect_descriptors(
    atoms, descriptor_extractor, n_steps, temperature_K, timestep_fs, friction,
    collect_every=10, trajectory_file=None
):
    """Run Langevin MD and collect MACE descriptors at regular intervals.

    Uses get_descriptors_differentiable (with torch.no_grad) to extract
    descriptors directly from the MACE model's internal forward pass. This
    works regardless of what calculator is currently attached to atoms (e.g.,
    during biased MD with SumCalculator).

    Parameters
    ----------
    atoms : ase.Atoms
        Starting configuration (calculator must be attached).
    descriptor_extractor : MACEDescriptorExtractor
        Extracts descriptors from each frame.
    n_steps : int
        Total number of MD steps.
    temperature_K : float
        Target temperature in Kelvin.
    timestep_fs : float
        Integration timestep in femtoseconds.
    friction : float
        Langevin friction coefficient in 1/fs.
    collect_every : int
        Collect descriptors every this many steps.
    trajectory_file : str, optional
        If provided, write trajectory to this file.

    Returns
    -------
    np.ndarray
        Collected descriptors, shape (n_frames, n_features).
    """
    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction / units.fs,
    )

    descriptors_list = []

    def collect():
        # Use differentiable mode (with no_grad for efficiency) to extract
        # descriptors from the internal MACE model. This avoids depending on
        # atoms.calc being the MACE calculator (works during biased MD too).
        positions_tensor = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float64,
            device=descriptor_extractor.device,
        )
        with torch.no_grad():
            desc = descriptor_extractor.get_descriptors_differentiable(atoms, positions_tensor)
        descriptors_list.append(desc.cpu().numpy())

    dyn.attach(collect, interval=collect_every)

    if trajectory_file:
        traj = Trajectory(trajectory_file, "w", atoms)
        dyn.attach(traj.write, interval=collect_every)

    dyn.run(n_steps)

    if trajectory_file:
        traj.close()

    return np.array(descriptors_list)


descriptors = run_md_collect_descriptors(
    atoms, descriptor_extractor, N_MD_STEPS, TEMPERATURE_K, TIMESTEP_FS, FRICTION,
    collect_every=COLLECT_EVERY,
    trajectory_file=os.path.join(RESULTS_DIR, "unbiased.traj") if os.path.isdir(RESULTS_DIR) else None,
)
print(f"  Collected {descriptors.shape[0]} frames, descriptor shape: {descriptors.shape}")


# ============================================================
# Step 5: Create dataset from collected descriptors
# ============================================================
# Convert the numpy descriptor array into the DictDataset/DictModule format
# expected by skewencoder training functions.
print("Step 5: Creating dataset from descriptors...")
dataset, datamodule = create_dataset_from_descriptors(
    descriptors, batch_size=BATCH_SIZE, verbose=True
)

# Define encoder architecture: input_dim -> hidden layers -> 1D latent (CV)
encoder_layers = [n_features, 50, 30, 20, 10, 5, 1]
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

os.makedirs(RESULTS_DIR, exist_ok=True)
all_descriptors = descriptors.copy()

for iteration in range(N_ITERATIONS):
    iter_folder = os.path.join(RESULTS_DIR, f"iter_{iteration}")
    os.makedirs(iter_folder, exist_ok=True)
    print(f"\n--- Iteration {iteration} ---")

    # 6a: Train skewencoder model
    # The model is an autoencoder with an auxiliary skewness loss that
    # encourages the latent CV to have non-Gaussian (skewed) distribution.
    print("  Training skewencoder...")
    dataset, datamodule = create_dataset_from_descriptors(
        all_descriptors, batch_size=BATCH_SIZE, verbose=False
    )
    model = skewencoder_model_init(dataset, encoder_layers, LOSS_COEFF)
    metrics = skewencoder_model_trainer(model, datamodule, iter_folder=iter_folder)

    # 6b: Normalize model output to [0, 1] range
    model = skewencoder_model_normalization(model, dataset)

    # 6c: Evaluate CV on training data and compute statistics
    # The skewness determines which direction to push: positive skewness
    # means data is piled on the left → push rightward (lower wall).
    nn_output = cv_eval(model, dataset).flatten()
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

    # 6e: Combine MACE potential + bias using ASE's SumCalculator
    # Both calculators contribute independently to energy and forces.
    combined_calc = SumCalculator([mace_calc, bias_calc])
    atoms.calc = combined_calc

    # 6f: Run biased MD and collect new descriptors
    # The bias pushes the system toward unexplored CV regions.
    print(f"  Running biased MD ({N_MD_STEPS} steps)...")
    new_descriptors = run_md_collect_descriptors(
        atoms, descriptor_extractor, N_MD_STEPS, TEMPERATURE_K, TIMESTEP_FS, FRICTION,
        collect_every=COLLECT_EVERY,
        trajectory_file=os.path.join(iter_folder, "biased.traj"),
    )
    print(f"  Collected {new_descriptors.shape[0]} new frames")

    # 6g: Accumulate descriptors for next iteration
    # All data from all iterations is used for training to ensure the model
    # captures the full explored landscape.
    all_descriptors = np.vstack([all_descriptors, new_descriptors])
    print(f"  Total accumulated frames: {all_descriptors.shape[0]}")

    # Restore MACE calculator for next iteration's unbiased descriptor collection
    atoms.calc = mace_calc


# ============================================================
# Step 7: Final results
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Workflow complete!")
print(f"  Total frames collected: {all_descriptors.shape[0]}")
print(f"  Results saved in: {RESULTS_DIR}")
print(f"  Final model checkpoint: {os.path.join(RESULTS_DIR, f'iter_{N_ITERATIONS-1}', 'checkpoint.ckpt')}")
