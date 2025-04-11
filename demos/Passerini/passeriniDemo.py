import numpy as np
import pandas as pd
import torch
import sys
import os
from scipy import stats
from collections.abc import Sequence
from os.path import dirname
import subprocess
from functools import partial

# PARENT_DIR = dirname(dirname(os.path.abspath(__file__)))
SCRIPT_DIR = dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(dirname(SCRIPT_DIR)) #(os.path.dirname(f"{PARENT_DIR}/utils"))
os.chdir(SCRIPT_DIR)

# TODO: if windows/ if zsh
win_bash_exe_prefix = ["bash","-c"]
zsh_prefix = ["/bin/zsh", "-c"]

# current_os = "windows"
current_os = "zsh"

if current_os == "windows":
    bash_prefix = win_bash_exe_prefix
else:
    bash_prefix = zsh_prefix


import skewencoder.state_detection as STADECT
from skewencoder.io import load_dataframe, load_data, GeometryParser
import skewencoder.switchfunction as sf
from skewencoder.model_skewencoder import skewencoder_model_init, skewencoder_model_trainer, skewencoder_model_normalization, cv_eval
from skewencoder.gen_plumed import PlumedInput

RESULTS_FOLDER = f"./results"
UNBIASED_FOLDER = f"./unbiased"
LIGHTNING_LOGS = f"./lightning_logs"
COORD_FILE = f"./coord.dat"

# TODO: may be later move to the skewencoder module
def passerini_training(state_detection: STADECT.State_detection, iter: int, encoder_layers : Sequence[int], loss_coeff: float, batch_size: int, pattern : str, subfix: str | None = None):
    ITER_FOLDER = RESULTS_FOLDER + f"/iter_{iter}"
    if not os.path.isdir(ITER_FOLDER):
        subprocess.run([*bash_prefix,f"mkdir {ITER_FOLDER}"], cwd=SCRIPT_DIR)
    break_flag = False

    if iter == 0:
        filenames_iter = [f"{UNBIASED_FOLDER}/COLVAR"]
        filenames_all = filenames_iter
    else:
        filenames_all = [f"{RESULTS_FOLDER}/iter_{i}/COLVAR" for i in range(iter) ]
        filenames_all.append(f"{UNBIASED_FOLDER}/COLVAR")
        filenames_iter = [f"{RESULTS_FOLDER}/iter_{iter-1}/COLVAR"]
    AE_dataset, skewness_dataset, datamodule, AE_df, skewness_df = load_data(filenames_iter,filenames_all,multiple=(iter + 1), bs=batch_size, pattern=pattern)

    if iter == 0:
        is_stable_state, is_new_state = state_detection(filenames_iter[0])
        model = skewencoder_model_init(AE_dataset,encoder_layers, loss_coeff, subfix=subfix)
    else:
        PREV_ITER_FOLDER = f"{RESULTS_FOLDER}/iter_{iter-1}" # TODO: Might use os.path.dirname
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
            model = skewencoder_model_init(AE_dataset,encoder_layers, loss_coeff, subfix=subfix)
        else:
            print("****************************")
            print("Apply Warm Start")
            print("Apply Warm Start")
            print("Apply Warm Start")
            print("****************************")
            model = skewencoder_model_init(AE_dataset,encoder_layers, loss_coeff,iter=iter,PREV_ITER_FOLDER=PREV_ITER_FOLDER, subfix=subfix)

    metrics = skewencoder_model_trainer(model, datamodule, iter_folder=ITER_FOLDER, subfix=subfix)

    model = skewencoder_model_normalization(model, AE_dataset)

    if subfix is not None:
        traced_model = model.to_torchscript(file_path=f'{ITER_FOLDER}/model_autoencoder_{iter}_{subfix}.pt', method='trace')
    else:
        traced_model = model.to_torchscript(file_path=f'{ITER_FOLDER}/model_autoencoder_{iter}.pt', method='trace')
        

    return state_detection, model, ITER_FOLDER,skewness_dataset, break_flag


def passerini_simulation(model, dataset, kappa, offset):
    nn_output = cv_eval(model, dataset).flatten()
    mu_sknn = np.mean(nn_output)
    var_sknn = np.var(nn_output)
    skew_sknn = stats.skew(nn_output)
    offset += np.sqrt(var_sknn)
    is_lower_wall = True
    if skew_sknn < 0:
        is_lower_wall = False

    return {"is_lower_wall": is_lower_wall, "kappa": kappa, "pos": mu_sknn, "offset": offset}


def gen_plumed_passerini(plumed_input : PlumedInput = None, file_path = SCRIPT_DIR):
    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')
    input= plumed_input.build()
    print(input, file=file)
    file.close()


def main(kappa):
    n_max_iter = 40
    loss_coeff = 0.1
    torch.manual_seed(22)
    batch_size = 100
    offset = 1.0
    n_layer_factor = 3.5
    only_heavy = False

    pattern_heavy = r"^([A-GI-Za-gi-z]+)\d+([A-GI-Za-gi-z]+)\d+$"
    pattern_h_adj = r"^(H)\d+([A-GI-Za-gi-z]+)\d+$"

    subprocess.run([*bash_prefix,f"rm -rf {RESULTS_FOLDER}"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix,f"rm -rf {LIGHTNING_LOGS}"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix,f"rm -rf {UNBIASED_FOLDER}"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"rm -f {kappa[0]}_iter* all*.pdb"], cwd=SCRIPT_DIR)

    subprocess.run([*bash_prefix, f"mkdir -p {UNBIASED_FOLDER}"])

    subprocess.run([*bash_prefix, f"echo '******************************************************'"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"echo Start unbiased simulation"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"echo '******************************************************'"], cwd=SCRIPT_DIR)
    geo_parser : GeometryParser = GeometryParser(coord_file=COORD_FILE)
    plumed_input_unbiased : PlumedInput = PlumedInput(geo_parser=geo_parser, only_heavy=only_heavy, if_biased=False, simulation_folder = UNBIASED_FOLDER)
    gen_plumed_passerini(plumed_input = plumed_input_unbiased, file_path = SCRIPT_DIR)
    n_descriptors_heavy = len(plumed_input_unbiased.heavy_atom_pairs)
    if not only_heavy:
        n_descriptors_h_adj = len(plumed_input_unbiased.h_heavy_pairs)
    
    subprocess.run([*bash_prefix,"srun -n 12 cp2k.popt job.inp > output.log"], cwd=SCRIPT_DIR)

    # For command line interface testing
    # subprocess.run([*bash_prefix,"cp2k.popt job.inp > output.log"], cwd=SCRIPT_DIR)

    subprocess.run([*bash_prefix, "mv Passerini-1.restart newiter.restart"], cwd = SCRIPT_DIR)
    subprocess.run([*bash_prefix, "rm -f Passerini*.restart"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"mv Passerini-pos-1.xyz {kappa[0]}_iteration_Chaba_unbiased-pos.xyz"], cwd = SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"cat {kappa[0]}_iteration_Passerini_unbiased-pos.xyz > all_{kappa[0]}.xyz"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix,"rm -f PLUMED.OUT Passerini*"], cwd=SCRIPT_DIR)

    heavy_bond_type_dict, n_descriptors_heavy_from_colvar,heavy_atom_pairs_list = STADECT.parse_unbiased_colvar(colvar_file = f"{UNBIASED_FOLDER}/COLVAR", transform_colvar_key=partial(STADECT.transform_colvar_key, pattern = pattern_heavy))
    assert n_descriptors_heavy == n_descriptors_heavy_from_colvar

    if not only_heavy:
        h_adj_bond_type_dict, n_descriptors_h_adj_from_colvar,h_adj_atom_pairs_list = STADECT.parse_unbiased_colvar(colvar_file = f"{UNBIASED_FOLDER}/COLVAR", transform_colvar_key=partial(STADECT.transform_colvar_key, pattern = pattern_h_adj))
        assert n_descriptors_h_adj == n_descriptors_h_adj_from_colvar

    encoder_layer_heavy = [n_descriptors_heavy, int(n_layer_factor*n_descriptors_heavy), int(n_layer_factor/2*n_descriptors_heavy), int(n_layer_factor/4*n_descriptors_heavy), int(np.sqrt(n_layer_factor/4*n_descriptors_heavy)), 1]
    if not only_heavy:    
        encoder_layer_h_adj = [n_descriptors_h_adj, int(n_layer_factor*n_descriptors_h_adj), int(n_layer_factor/2*n_descriptors_h_adj), int(n_layer_factor/4*n_descriptors_heavy), int(np.sqrt(n_layer_factor/4*n_descriptors_h_adj)), 1]
    state_detection_heavy = STADECT.State_detection((0.3, 0.7), bond_type_dict=heavy_bond_type_dict, n_heavy_atom_pairs=n_descriptors_heavy_from_colvar)
    if not only_heavy:
        state_detection_h_adj = STADECT.State_detection((0.3, 0.7), bond_type_dict=h_adj_bond_type_dict, n_heavy_atom_pairs=n_descriptors_h_adj_from_colvar)

    subprocess.run([*bash_prefix, f"mkdir -p {RESULTS_FOLDER}"])

    for iter in range(n_max_iter):
        state_detection_heavy, model_heavy, ITER_FOLDER, skewness_dataset_heavy, break_flag_heavy = passerini_training(state_detection_heavy, iter, encoder_layer_heavy, loss_coeff, batch_size, pattern=pattern_heavy, subfix="heavy")
        if not only_heavy:
            state_detection_h_adj, model_h_adj, ITER_FOLDER, skewness_dataset_h_adj, break_flag_h_adj = passerini_training(state_detection_h_adj, iter, encoder_layer_h_adj, loss_coeff, batch_size, pattern=pattern_h_adj, subfix="h_adj")
        subprocess.run([*bash_prefix, f"echo '******************************************************'"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"echo At the iteration {iter} training step, "], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"echo The current state for heavy atom pairs is {state_detection_heavy.current_state}"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"echo The current state for h_adj atom pairs is {state_detection_h_adj.current_state}"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"echo '******************************************************'"], cwd=SCRIPT_DIR)
        model_name_heavy = f"{ITER_FOLDER}/model_autoencoder_{iter}_heavy.pt"
        model_name_h_adj = None
        if not only_heavy:
            model_name_h_adj = f"{ITER_FOLDER}/model_autoencoder_{iter}_h_adj.pt"
        
        skew_wall_h_adj = None
        skew_wall_heavy = passerini_simulation(model_heavy, skewness_dataset_heavy, kappa[0], offset)
        if not only_heavy:
            skew_wall_h_adj = passerini_simulation(model_h_adj, skewness_dataset_h_adj, kappa[1], offset)

        plumed_input_biased = PlumedInput(geo_parser=geo_parser, if_biased=True, heavy_atom_only=only_heavy, pytorch_model_heavy= model_name_heavy, pytorch_model_h= model_name_h_adj, skew_wall_heavy = skew_wall_heavy, skew_wall_h_adj = skew_wall_h_adj, simulation_folder = ITER_FOLDER)
        
        gen_plumed_passerini(plumed_input = plumed_input_biased, file_path = SCRIPT_DIR)
        
        subprocess.run([*bash_prefix,"srun -n 12 cp2k.popt job_restart.inp > output.log"], cwd=SCRIPT_DIR)
        # For command line interface testing
        # subprocess.run([*bash_prefix,"cp2k.popt job_restart.inp > output.log"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, "mv Passerini-1.restart newiter.restart"], cwd = SCRIPT_DIR)
        subprocess.run([*bash_prefix, "rm -f Passerini*.restart"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"mv Passerini-pos-1.xyz {kappa[0]}_iteration_Passerini_{iter}-pos.xyz"], cwd = SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"tail -n +3 {kappa[0]}_iteration_Passerini_{iter}-pos.xyz >> all_{kappa[0]}.xyz"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix,"rm -f PLUMED.OUT Passerini*"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"echo CP2K simulation at iteration {iter} with plumed ends"], cwd=SCRIPT_DIR)
    if not only_heavy:
        print(f"there are in total {state_detection_heavy.n_states} states and {state_detection_h_adj.n_states} states")
    else:
        print(f"there are in total {state_detection_heavy.n_states} states")

if __name__ == "__main__":
    kappa = [np.int_(sys.argv[1])]
    kappa.append(100)
    main(kappa)
    subprocess.run([*bash_prefix, "rm -f bck* Passerini* *.OUT"], cwd=SCRIPT_DIR)
