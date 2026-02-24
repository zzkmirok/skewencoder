import os
import re
import subprocess
import sys
from collections.abc import Sequence
from os.path import dirname

import numpy as np
import torch
from mpi4py import MPI
from scipy import stats

SCRIPT_DIR = dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(dirname(SCRIPT_DIR))  # (os.path.dirname(f"{PARENT_DIR}/utils"))
os.chdir(SCRIPT_DIR)
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

import argparse
import tomllib

parser = argparse.ArgumentParser(description="Read TOML config")
parser.add_argument(
    "-c", "--config", type=str, default="config.toml", help="Path to TOML config"
)
parser.add_argument("-test", action="store_true", help="activate test mode")
cv_group = parser.add_mutually_exclusive_group()
cv_group.add_argument("-custom", action="store_true", help="use customized cv input")
cv_group.add_argument("-all", action="store_true", help="use both cv inputs")
cv_group.add_argument("-normal", action="store_true", help="use normal cv inputs")
parser.set_defaults(normal=True)

ARGS = parser.parse_args()
# Set normal as default if no cv option specified
if not (ARGS.custom or ARGS.all or ARGS.normal):
    ARGS.normal = True


def load_config():
    with open(ARGS.config, "rb") as f:
        config_data = tomllib.load(f)

    return config_data


CONFIG = load_config()

EXEC_DICT = {
    "mace": "/home/rwth1997/lammps-custom-mace/bin/lmp",
    "metatomic": "/home/rwth1997/lammps-custom/bin/lmp",
}

REPLACE_DICT = {
    "<SYSTEM_FILE>": os.path.join(
        CONFIG["simulation"]["systems_folder"], CONFIG["simulation"]["system"] + ".data"
    ),
    "<MLIP>": CONFIG["simulation"]["mlip"],
    "<MODEL_FILE>": os.path.join(
        CONFIG["simulation"]["model_folder"], CONFIG["simulation"]["model"]
    ),
    "<ATOM_LIST>": " ".join(CONFIG["simulation"]["atom_list"]),
    "<PLUMED_LOG>": CONFIG["plumed"].get("log_file", "plumed.log"),
    "<TEMPERATURE>": f"{CONFIG['simulation']['temperature']:.2f}",
    "<SEED>": f"{CONFIG['simulation']['seed']}",
    "<RESTART_FILE>": CONFIG["simulation"]["restart_file"],
    "<FROZEN_LAYERS>": CONFIG["simulation"]["frozen_layers"],
}

# TODO: if windows/ if zsh
win_bash_exe_prefix = ["bash", "-c"]
zsh_prefix = ["/bin/zsh", "-c"]

# current_os = "windows"
current_os = "zsh"

if current_os == "windows":
    bash_prefix = win_bash_exe_prefix
else:
    bash_prefix = zsh_prefix


# TODO: change lammps input file

DESCRIPTOR_LIST = CONFIG["plumed"]["descriptors"]
ENV_BIAS_LIST = CONFIG["plumed"].get("env_bias_cv", [])
CUSTOM_DESCRIPTOR_LIST = CONFIG["plumed"].get("custom_descriptors", [])
CV_TYPE = CONFIG["plumed"]["cv_type"]

set_DISCRIPTOR_LIST = set(DESCRIPTOR_LIST)
REFINED_CUSTOM_DESCRIPTOR_LIST = [item for item in CUSTOM_DESCRIPTOR_LIST if item not in set_DISCRIPTOR_LIST]

SKEWENCODER_INPUT_LIST = DESCRIPTOR_LIST
if ARGS.custom:
    SKEWENCODER_INPUT_LIST = CUSTOM_DESCRIPTOR_LIST
elif ARGS.all:
    SKEWENCODER_INPUT_LIST = [*DESCRIPTOR_LIST, *CUSTOM_DESCRIPTOR_LIST]
else:
    pass

def descriptor_script():
    no_pbc = "" if CONFIG["plumed"]["pbc"] else "NOPBC"
    lines = []
    if CV_TYPE == "DISTANCE":
        for descr in DESCRIPTOR_LIST:
            atom_nums = re.findall(r"\d+", descr)

            if len(atom_nums) == 2:
                line = (
                    f"{descr}: {CV_TYPE} ATOMS={atom_nums[0]},{atom_nums[1]} {no_pbc}"
                )
                lines.append(line)

    return "\n".join(lines)


def custom_cv_script():
    return CONFIG["plumed"]["custom_cv_definition"]


def additional_biases():
    return CONFIG["plumed"]["env_bias"]


import skewencoder.state_detection as STADECT
from skewencoder.io import load_data
from skewencoder.model_skewencoder import (cv_eval, skewencoder_model_init,
                                           skewencoder_model_normalization,
                                           skewencoder_model_trainer)

RESULTS_FOLDER = "./results"
UNBIASED_FOLDER = "./unbiased"
LIGHTNING_LOGS = "./lightning_logs"


def skewencoder_training(
    state_detection: STADECT.State_detection,
    iter: int,
    encoder_layers: Sequence[int],
    loss_coeff: float,
    batch_size: int,
    input_pattern: str = r"^([A-Za-z]+)\d+([A-Za-z]+)\d+$",
):
    ITER_FOLDER = RESULTS_FOLDER + f"/iter_{iter}"
    subprocess.run([*bash_prefix, f"mkdir {ITER_FOLDER}"], cwd=SCRIPT_DIR)
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
        PREV_ITER_FOLDER = (
            f"{RESULTS_FOLDER}/iter_{iter - 1}"  # TODO: Might use os.path.dirname
        )
        is_stable_state, is_new_state = state_detection(filenames_iter[0])
        # TODO: different from original algorithm
        # so far each time back to or reach stable states then cold start
        # not e.g. state 1 - state 1 cold start, which was warm start before
        apply_warm_start = not is_stable_state

        # TODO: so far no early stopping
        # if state_detection.current_state == 1 and state_detection.n_states > 2:
        if not apply_warm_start:
            print("****************************")
            print("Restart from Scratch")
            print("Restart from Scratch")
            print("Restart from Scratch")
            print("****************************")
            model = skewencoder_model_init(AE_dataset, encoder_layers, loss_coeff)
        else:
            print("****************************")
            print("Apply Warm Start")
            print("Apply Warm Start")
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


def gen_plumed_unbiased(file_path=SCRIPT_DIR, simulation_folder=UNBIASED_FOLDER):
    file_path = f"{file_path}/plumed.dat"
    file = open(file_path, "w")
    input = f"""# vim:ft=plumed
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs

{descriptor_script()}
{custom_cv_script()}
{additional_biases()}
# PRINT all variables
PRINT FMT=%g STRIDE={CONFIG["plumed"]["stride"]} FILE={simulation_folder}/COLVAR ARG={",".join(ENV_BIAS_LIST) if len(ENV_BIAS_LIST) > 0 else ""}{"," if len(ENV_BIAS_LIST) > 0 else ""}{",".join(DESCRIPTOR_LIST) if len(DESCRIPTOR_LIST) > 0 else ""}{"," if len(DESCRIPTOR_LIST) > 0 else ""}{",".join(REFINED_CUSTOM_DESCRIPTOR_LIST) if len(REFINED_CUSTOM_DESCRIPTOR_LIST) > 0 else ""}
"""
    print(input, file=file)
    file.close()


def gen_plumed_biased(
    model_name: str, file_path: str, simulation_folder, pos, skew, kappa, offset
):
    file_path = f"{file_path}/plumed.dat"
    file = open(file_path, "w")
    input = f"""# vim:ft=plumed
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs
{descriptor_script()}
{custom_cv_script()}
{additional_biases()}

cv: PYTORCH_MODEL FILE={model_name} ARG={",".join(SKEWENCODER_INPUT_LIST)}
    """
    print(input, file=file)
    file.close()
    walltype = ""
    if skew < 0:
        walltype = "UPPER_WALLS"
        offset = -offset
    else:
        walltype = "LOWER_WALLS"
    with open(file_path, "a") as f:
        print(
            f"""
# Energy wall for aes cv
wall: {walltype} ARG=cv.node-0 AT={pos + offset} KAPPA={kappa} ExP=2 EPS=1 OFFSET=0.0
PRINT FMT=%g STRIDE={CONFIG["plumed"]["stride"]} FILE={simulation_folder}/COLVAR ARG={",".join(ENV_BIAS_LIST) if len(ENV_BIAS_LIST) > 0 else ""}{"," if len(ENV_BIAS_LIST) > 0 else ""}{",".join(DESCRIPTOR_LIST) if len(DESCRIPTOR_LIST) > 0 else ""}{"," if len(DESCRIPTOR_LIST) > 0 else ""}{",".join(REFINED_CUSTOM_DESCRIPTOR_LIST) if len(REFINED_CUSTOM_DESCRIPTOR_LIST) > 0 else ""},cv.*""",
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
        file_path=".",
        simulation_folder=iter_folder,
        pos=mu_sknn,
        skew=skew_sknn,
        kappa=kappa,
        offset=offset,
    )


def gen_input_lmp_template(lmp_file: str, replacements=REPLACE_DICT):
    with open(
        os.path.join(
            CONFIG["simulation"]["config_folder"],
            CONFIG["driver"],
            f"{lmp_file}.template",
        ),
        "r",
    ) as f:
        template = f.read()

    final_script = template
    for placeholder, value in replacements.items():
        final_script = final_script.replace(placeholder, str(value))

    return final_script


def main(kappa):
    n_max_iter = CONFIG["loxodynamics"]["max_iter"]
    loss_coeff = 0.1
    torch.manual_seed(22)
    batch_size = CONFIG["loxodynamics"]["batch_size"]
    offset = 1.0
    driver_exec = ""
    n_steps = 5000  # TODO: must also be changed later
    files_cleaned_pattern = []
    plumed_log = CONFIG["plumed"]["log_file"]

    n_gpus = CONFIG["simulation"]["n_gpus"]
    if CONFIG["driver"] == "lammps":
        files_cleaned_pattern = "bck* *.dmp *log* *restart* *.lammpstrj"
        driver_exec = EXEC_DICT[f"{CONFIG['simulation']['mlip']}"]
    job_exec = driver_exec
    if not ARGS.test:
        job_exec = " ".join(["srun", driver_exec])
    if RANK == 0:
        subprocess.run([*bash_prefix, f"rm -rf {RESULTS_FOLDER}"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"rm -rf {LIGHTNING_LOGS}"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"rm -rf {UNBIASED_FOLDER}"], cwd=SCRIPT_DIR)
        subprocess.run(
            [*bash_prefix, f"rm -rf {files_cleaned_pattern}"]
        )  # , cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"mkdir -p {UNBIASED_FOLDER}"])

        subprocess.run(
            [
                *bash_prefix,
                "echo '******************************************************'",
            ],
            cwd=SCRIPT_DIR,
        )
        subprocess.run(
            [
                *bash_prefix,
                f"echo Start {CONFIG['driver']} {CONFIG['simulation']['system']} unbiased simulation",
            ],
            cwd=SCRIPT_DIR,
        )
        subprocess.run(
            [
                *bash_prefix,
                "echo '******************************************************'",
            ],
            cwd=SCRIPT_DIR,
        )

        gen_plumed_unbiased()

        subprocess.run(
            [
                *bash_prefix,
                f"{job_exec} -k on g {n_gpus} -in in_unbiased.lammps -sf kk",
            ],
            cwd=SCRIPT_DIR,
        )
        # subprocess.run([*bash_prefix,f"srun {driver_exec} -k on g {n_gpus} -in in_unbiased.lammps -sf kk"], cwd=SCRIPT_DIR)

    COMM.Barrier()
    if RANK == 0:
        subprocess.run(
            [*bash_prefix, f"cp ./plumed.dat {UNBIASED_FOLDER}"], cwd=SCRIPT_DIR
        )
        subprocess.run(
            [
                *bash_prefix,
                f"mv {CONFIG['simulation']['restart_file']}.{n_steps} {CONFIG['simulation']['restart_file']}",
            ],
            cwd=SCRIPT_DIR,
        )
        subprocess.run([*bash_prefix, f"rm -f {plumed_log}"], cwd=SCRIPT_DIR)
        # TODO: change bond_type_lib
        n_descriptors = len(SKEWENCODER_INPUT_LIST)
        hidden_layers = CONFIG["loxodynamics"]["skewencoder"]["hidden_layers"]
        encoder_layers = [n_descriptors, *hidden_layers, 1]
        threshold = CONFIG["loxodynamics"]["skewencoder"]["threshold"]

        bond_type_lib = STADECT.Bond_type_lib()
        bond_type_lib.build_default()
        bond_type_dict = bond_type_lib.bond_type_dict

        detector_regex = CONFIG["plumed"].get(
            "detector_regex", r"^([A-Za-z]+)\d+([A-Za-z]+)\d+$"
        )
        state_detection = STADECT.State_detection(
            (threshold, 1 - threshold),
            bond_type_dict=bond_type_dict,
            n_heavy_atom_pairs=len(DESCRIPTOR_LIST),
            pattern=detector_regex,
        )
        input_pattern = CONFIG["plumed"].get(
            "input_regex", r"^([A-Za-z]+)\d+([A-Za-z]+)\d+$"
        )

        subprocess.run([*bash_prefix, f"mkdir -p {RESULTS_FOLDER}"])

    COMM.Barrier()

    for iter in range(n_max_iter):
        if RANK == 0:
            state_detection, model, ITER_FOLDER, skewness_dataset, break_flag = (
                skewencoder_training(
                    state_detection=state_detection,
                    iter=iter,
                    encoder_layers=encoder_layers,
                    loss_coeff=loss_coeff,
                    batch_size=batch_size,
                    input_pattern=input_pattern,
                )
            )
        COMM.Barrier()

        if RANK == 0:
            subprocess.run(
                [
                    *bash_prefix,
                    "echo '******************************************************'",
                ],
                cwd=SCRIPT_DIR,
            )
            subprocess.run(
                [*bash_prefix, f"echo At the iteration {iter} training step, "],
                cwd=SCRIPT_DIR,
            )
            subprocess.run(
                [
                    *bash_prefix,
                    f"echo The current state is {state_detection.current_state}",
                ],
                cwd=SCRIPT_DIR,
            )
            subprocess.run(
                [
                    *bash_prefix,
                    "echo '******************************************************'",
                ],
                cwd=SCRIPT_DIR,
            )
            model_name = f"{ITER_FOLDER}/model_autoencoder_{iter}.pt"
            loxodynamics_simulation(
                ITER_FOLDER, model_name, model, skewness_dataset, kappa, offset
            )
            subprocess.run(
                [*bash_prefix, f"{job_exec} -k on g {n_gpus} -in in.lammps -sf kk"],
                cwd=SCRIPT_DIR,
            )
            subprocess.run(
                [*bash_prefix, f"cp ./plumed.dat {ITER_FOLDER}"], cwd=SCRIPT_DIR
            )

        COMM.Barrier()

        if RANK == 0:
            subprocess.run(
                [
                    *bash_prefix,
                    f"mv {CONFIG['simulation']['restart_file']}.{(iter + 2) * n_steps} {CONFIG['simulation']['restart_file']}",
                ],
                cwd=SCRIPT_DIR,
            )
            subprocess.run([*bash_prefix, f"rm -f {plumed_log}"], cwd=SCRIPT_DIR)
            subprocess.run(
                [
                    *bash_prefix,
                    f"echo {CONFIG['driver']} {CONFIG['simulation']['system']} simulation at iteration {iter} with plumed ends",
                ],
                cwd=SCRIPT_DIR,
            )
        COMM.Barrier()

    print(f"there are in total {state_detection.n_states} states")


if __name__ == "__main__":
    kappa = CONFIG["loxodynamics"]["kappa"]
    if RANK == 0:
        with open("in_unbiased.lammps", "w") as f:
            f.write(gen_input_lmp_template(lmp_file="in_unbiased.lammps"))
        with open("in.lammps", "w") as f:
            f.write(gen_input_lmp_template(lmp_file="in.lammps"))
    COMM.Barrier()
    main(kappa)
    if CONFIG["driver"] == "lammps":
        files_deleted = "tmp* bck*"
        if RANK == 0:
            subprocess.run([*bash_prefix, f"rm -f {files_deleted}"], cwd=SCRIPT_DIR)
