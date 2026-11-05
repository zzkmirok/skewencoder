"""Loxodynamics with DFTB+ (GFN2-xTB): CH3NH2 + COCl2 starting from their van der Waals complex.

Same workflow as Ru001-NH3/template-2NH3-cn/RuNH3Demo.py with DFTB+ as the MD engine:
one unbiased segment, then per iteration: state detection, skewencoder training, and a biased
segment with a one-sided wall on the skewencoder CV. Each segment continues from the last frame
(geometry and velocities) of the previous one.

The CVs are the heavy-atom descriptors generated from the starting geometry, and the PLUMED input is
written by skewencoder.gen_plumed.PlumedInput, including the contact wall.

    python PhosgeneDemo.py [-c config.toml] [-test]
"""
import argparse
import os
import shutil
import subprocess
import tomllib
from collections.abc import Sequence
from os.path import dirname

import numpy as np
import torch
from scipy import stats

SCRIPT_DIR = dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

parser = argparse.ArgumentParser(description="Read TOML config")
parser.add_argument(
    "-c", "--config", type=str, default="config.toml", help="Path to TOML config"
)
parser.add_argument("-test", action="store_true", help="activate test mode (run DFTB+ without srun)")
ARGS = parser.parse_args()


def load_config():
    with open(ARGS.config, "rb") as f:
        config_data = tomllib.load(f)

    return config_data


CONFIG = load_config()

DRIVER_EXEC = "dftb+"

SYSTEM_FILE = os.path.join(
    CONFIG["simulation"]["systems_folder"], CONFIG["simulation"]["system"] + ".xyz"
)
GEOMETRY_FILE = "geo_start.gen"  # geometry the next segment starts from
VELOCITIES_FILE = "velocities.dat"  # velocities (A/ps) the next biased segment starts from

REPLACE_DICT = {
    "<GEOMETRY_FILE>": GEOMETRY_FILE,
    "<VELOCITIES_FILE>": VELOCITIES_FILE,
    "<N_STEPS>": f"{CONFIG['simulation']['n_steps']}",
    "<TIMESTEP>": f"{CONFIG['simulation']['timestep']}",
    "<TEMPERATURE>": f"{CONFIG['simulation']['temperature']:.2f}",
    "<COUPLING_STRENGTH>": f"{CONFIG['simulation']['coupling_strength']}",
    "<PRINT_STRIDE>": f"{CONFIG['plumed']['stride']}",
    "<CHARGE>": f"{CONFIG['simulation']['charge']}",
    "<ELECTRONIC_TEMPERATURE>": f"{CONFIG['simulation']['electronic_temperature']:.2f}",
    "<SEED>": f"{CONFIG['simulation']['seed']}",
}

# DFTB+ outputs removed before a run; the trajectory and logs of each segment are kept in its folder
DFTB_OUTPUTS = ["geo_end.xyz", "geo_end.gen", "md.out", "detailed.out", "band.out", "charges.bin", "dftb_pin.hsd"]

import skewencoder.state_detection as STADECT
from skewencoder.gen_plumed import PlumedInput
from skewencoder.io import GeometryParser, load_data
from skewencoder.model_skewencoder import (cv_eval, skewencoder_model_init,
                                           skewencoder_model_normalization,
                                           skewencoder_model_trainer)

RESULTS_FOLDER = "./results"
UNBIASED_FOLDER = "./unbiased"
LIGHTNING_LOGS = "./lightning_logs"

GEO_PARSER = GeometryParser(coord_file=SYSTEM_FILE)
# Heavy-atom descriptors, e.g. C1Cl2; the same labels in every segment because they come from the starting geometry
DESCRIPTOR_LIST = [pair[0] for pair in PlumedInput(geo_parser=GEO_PARSER, heavy_atom_only=True).heavy_atom_pairs]


def plumed_input(simulation_folder: str, **options) -> PlumedInput:
    pycv_library = CONFIG["plumed"].get("pycv_library", "")
    if pycv_library:
        options["pycv_library"] = pycv_library
    return PlumedInput(
        geo_parser=GEO_PARSER,
        heavy_atom_only=True,
        simulation_folder=simulation_folder,
        print_stride=CONFIG["plumed"]["stride"],
        time_step=0.001,  # PLUMED time unit in ps: fs
        vdw_constr=CONFIG["plumed"].get("contact_wall", True),
        **options,
    )


def skewencoder_training(
    state_detection: STADECT.State_detection,
    iter: int,
    encoder_layers: Sequence[int],
    loss_coeff: float,
    batch_size: int,
    input_pattern: str = r"^([A-Za-z]+)\d+([A-Za-z]+)\d+$",
):
    ITER_FOLDER = RESULTS_FOLDER + f"/iter_{iter}"
    os.makedirs(ITER_FOLDER)
    break_flag = False

    if iter == 0:
        filenames_iter = [f"{UNBIASED_FOLDER}/COLVAR"]
        filenames_all = filenames_iter
    else:
        filenames_all = [f"{RESULTS_FOLDER}/iter_{i}/COLVAR" for i in range(iter)]
        filenames_all.append(f"{UNBIASED_FOLDER}/COLVAR")
        filenames_iter = [f"{RESULTS_FOLDER}/iter_{iter - 1}/COLVAR"]
    AE_dataset, skewness_dataset, datamodule, AE_df, skewness_df = load_data(
        filenames_iter,
        filenames_all,
        multiple=(iter + 1),
        bs=batch_size,
        pattern=input_pattern,
    )

    if iter == 0:
        is_stable_state, is_new_state = state_detection(filenames_iter[0])
        model = skewencoder_model_init(AE_dataset, encoder_layers, loss_coeff)
    else:
        PREV_ITER_FOLDER = f"{RESULTS_FOLDER}/iter_{iter - 1}"
        is_stable_state, is_new_state = state_detection(filenames_iter[0])
        # each time back to or reaching a stable state: cold start
        apply_warm_start = not is_stable_state

        if not apply_warm_start:
            print("****************************")
            print("Restart from Scratch")
            print("****************************")
            model = skewencoder_model_init(AE_dataset, encoder_layers, loss_coeff)
        else:
            print("****************************")
            print("Apply Warm Start")
            print("****************************")
            model = skewencoder_model_init(
                AE_dataset,
                encoder_layers,
                loss_coeff,
                iter=iter,
                PREV_ITER_FOLDER=PREV_ITER_FOLDER,
            )

    metrics = skewencoder_model_trainer(model, datamodule, iter_folder=ITER_FOLDER)

    model = skewencoder_model_normalization(model, AE_dataset)

    traced_model = model.to_torchscript(
        file_path=f"{ITER_FOLDER}/model_autoencoder_{iter}.pt", method="trace"
    )

    return state_detection, model, ITER_FOLDER, skewness_dataset, break_flag


def gen_plumed_unbiased(simulation_folder=UNBIASED_FOLDER):
    with open("plumed.dat", "w") as f:
        print(plumed_input(simulation_folder).build(), file=f)


def gen_plumed_biased(model_name: str, simulation_folder, pos, skew, kappa, offset):
    # skew < 0: UPPER_WALLS at pos - offset, otherwise LOWER_WALLS at pos + offset
    skew_wall_heavy = {
        "is_lower_wall": bool(skew >= 0),
        "kappa": float(kappa),
        "pos": float(pos),
        "offset": float(offset),
    }
    with open("plumed.dat", "w") as f:
        print(
            plumed_input(
                simulation_folder,
                if_biased=True,
                pytorch_model_heavy=model_name,
                skew_wall_heavy=skew_wall_heavy,
            ).build(),
            file=f,
        )


def loxodynamics_simulation(iter_folder, model_name, model, dataset, kappa, offset):
    nn_output = cv_eval(model, dataset).flatten()
    mu_sknn = np.mean(nn_output)
    var_sknn = np.var(nn_output)
    skew_sknn = stats.skew(nn_output)
    offset += np.sqrt(var_sknn)
    gen_plumed_biased(
        model_name=model_name,
        simulation_folder=iter_folder,
        pos=mu_sknn,
        skew=skew_sknn,
        kappa=kappa,
        offset=offset,
    )


def gen_input_dftb_template(template_name: str, replacements=REPLACE_DICT):
    with open(
        os.path.join(
            CONFIG["simulation"]["config_folder"],
            CONFIG["driver"],
            f"{template_name}.template",
        ),
        "r",
    ) as f:
        template = f.read()

    final_script = template
    for placeholder, value in replacements.items():
        final_script = final_script.replace(placeholder, str(value))

    return final_script


def write_gen(file_path: str, symbols: Sequence[str], positions: np.ndarray, box: float):
    """DFTB+ gen file of a cubic periodic cell with edge `box` (A)."""
    species = list(dict.fromkeys(symbols))
    with open(file_path, "w") as f:
        f.write(f"{len(symbols)} S\n{' '.join(species)}\n")
        for i, (s, p) in enumerate(zip(symbols, positions)):
            f.write(f"{i + 1} {species.index(s) + 1} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
        f.write(f"0.0 0.0 0.0\n{box} 0.0 0.0\n0.0 {box} 0.0\n0.0 0.0 {box}\n")


def read_xyz_last_frame(file_path: str):
    """Symbols and rows of numbers of the last frame of an xyz (trajectory) file."""
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    n_atoms = int(lines[0])
    n_frames = len(lines) // (n_atoms + 2)
    frame = lines[(n_frames - 1) * (n_atoms + 2) + 2 : n_frames * (n_atoms + 2)]
    rows = [line.split() for line in frame]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:]] for r in rows])


def gen_start_geometry():
    """Centre the system in the cubic box and write it as the geometry of the unbiased segment."""
    symbols, data = read_xyz_last_frame(SYSTEM_FILE)
    box = CONFIG["simulation"]["box"]
    positions = data[:, :3]
    positions = positions - positions.mean(axis=0) + box / 2
    write_gen(GEOMETRY_FILE, symbols, positions, box)


def save_restart(segment_folder: str):
    """Keep the segment's trajectory and logs, and set up the next segment from its last frame.

    DFTB+ MD frames in geo_end.xyz: symbol, x, y, z (A), charge, vx, vy, vz (A/ps).
    """
    symbols, data = read_xyz_last_frame("geo_end.xyz")
    write_gen(GEOMETRY_FILE, symbols, data[:, 0:3], CONFIG["simulation"]["box"])
    np.savetxt(VELOCITIES_FILE, data[:, 4:7], fmt="%.10f")
    shutil.move("geo_end.xyz", f"{segment_folder}/traj.xyz")
    for file in ["md.out", "dftb_in.hsd", "dftb.out", "plumed.dat"]:
        shutil.copy(file, segment_folder)


def run_dftb(job_exec: str, template_name: str):
    for file in DFTB_OUTPUTS:
        if os.path.exists(file):
            os.remove(file)
    with open("dftb_in.hsd", "w") as f:
        f.write(gen_input_dftb_template(template_name))
    with open("dftb.out", "w") as out:
        result = subprocess.run(job_exec, shell=True, cwd=SCRIPT_DIR, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"{job_exec} failed with exit code {result.returncode}, see dftb.out")


def main(kappa):
    n_max_iter = CONFIG["loxodynamics"]["max_iter"]
    loss_coeff = 0.1
    torch.manual_seed(22)
    batch_size = CONFIG["loxodynamics"]["batch_size"]
    offset = 1.0
    job_exec = DRIVER_EXEC
    if not ARGS.test:
        job_exec = " ".join(["srun", DRIVER_EXEC])

    for folder in [RESULTS_FOLDER, LIGHTNING_LOGS, UNBIASED_FOLDER]:
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(UNBIASED_FOLDER)

    print("******************************************************")
    print(f"Start {CONFIG['driver']} {CONFIG['simulation']['system']} unbiased simulation")
    print("******************************************************")

    gen_start_geometry()
    gen_plumed_unbiased()
    run_dftb(job_exec, "dftb_in_unbiased.hsd")
    save_restart(UNBIASED_FOLDER)

    n_descriptors = len(DESCRIPTOR_LIST)
    hidden_layers = CONFIG["loxodynamics"]["skewencoder"]["hidden_layers"]
    encoder_layers = [n_descriptors, *hidden_layers, 1]
    threshold = CONFIG["loxodynamics"]["skewencoder"]["threshold"]

    bond_type_lib = STADECT.Bond_type_lib()
    bond_type_lib.build_default()
    bond_type_dict = bond_type_lib.bond_type_dict

    state_detection = STADECT.State_detection(
        (threshold, 1 - threshold),
        bond_type_dict=bond_type_dict,
        n_heavy_atom_pairs=n_descriptors,
    )

    os.makedirs(RESULTS_FOLDER)

    for iter in range(n_max_iter):
        state_detection, model, ITER_FOLDER, skewness_dataset, break_flag = (
            skewencoder_training(
                state_detection=state_detection,
                iter=iter,
                encoder_layers=encoder_layers,
                loss_coeff=loss_coeff,
                batch_size=batch_size,
            )
        )

        print("******************************************************")
        print(f"At the iteration {iter} training step, ")
        print(f"The current state is {state_detection.current_state}")
        print("******************************************************")
        model_name = f"{ITER_FOLDER}/model_autoencoder_{iter}.pt"
        loxodynamics_simulation(
            ITER_FOLDER, model_name, model, skewness_dataset, kappa, offset
        )
        run_dftb(job_exec, "dftb_in.hsd")
        save_restart(ITER_FOLDER)
        print(
            f"{CONFIG['driver']} {CONFIG['simulation']['system']} simulation at iteration {iter} with plumed ends"
        )

    print(f"there are in total {state_detection.n_states} states")


if __name__ == "__main__":
    kappa = CONFIG["loxodynamics"]["kappa"]
    main(kappa)
    for file in ["bck.*"]:
        subprocess.run(f"rm -f {file}", shell=True, cwd=SCRIPT_DIR)
