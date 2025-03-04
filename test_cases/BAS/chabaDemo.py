import numpy as np
import pandas as pd
import torch
import sys
import os
from scipy import stats
from collections.abc import Sequence
from os.path import dirname
import subprocess

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
from skewencoder.io import load_dataframe, load_data
import skewencoder.switchfunction as sf
from skewencoder.model_skewencoder import skewencoder_model_init, skewencoder_model_trainer, skewencoder_model_normalization, cv_eval

RESULTS_FOLDER = f"./results"
UNBIASED_FOLDER = f"./unbiased"
LIGHTNING_LOGS = f"./lightning_logs"

def chaba_training(state_detection: STADECT.State_detection, iter: int, encoder_layers : Sequence[int], loss_coeff: float, batch_size: int):
    ITER_FOLDER = RESULTS_FOLDER + f"/iter_{iter}"
    subprocess.run([*bash_prefix,f"mkdir {ITER_FOLDER}"], cwd=SCRIPT_DIR)
    break_flag = False

    if iter == 0:
        filenames_iter = [f"{UNBIASED_FOLDER}/COLVAR"]
        filenames_all = filenames_iter
    else:
        filenames_all = [f"{RESULTS_FOLDER}/iter_{i}/COLVAR" for i in range(iter) ]
        filenames_all.append(f"{UNBIASED_FOLDER}/COLVAR")
        filenames_iter = [f"{RESULTS_FOLDER}/iter_{iter-1}/COLVAR"]
    AE_dataset, skewness_dataset, datamodule, AE_df, skewness_df = load_data(filenames_iter,filenames_all,multiple=(iter + 1), bs=batch_size)

    if iter == 0:
        is_stable_state, is_new_state = state_detection(filenames_iter[0])
        model = skewencoder_model_init(AE_dataset,encoder_layers, loss_coeff)
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
            model = skewencoder_model_init(AE_dataset,encoder_layers, loss_coeff)
        else:
            print("****************************")
            print("Apply Warm Start")
            print("Apply Warm Start")
            print("Apply Warm Start")
            print("****************************")
            model = skewencoder_model_init(AE_dataset,encoder_layers, loss_coeff,iter=iter,PREV_ITER_FOLDER=PREV_ITER_FOLDER)

    metrics = skewencoder_model_trainer(model, datamodule, iter_folder=ITER_FOLDER)

    model = skewencoder_model_normalization(model, AE_dataset)

    traced_model = model.to_torchscript(file_path=f'{ITER_FOLDER}/model_autoencoder_{iter}.pt', method='trace')

    return state_detection, model, ITER_FOLDER,skewness_dataset, break_flag


def gen_plumed_chaba_unbiased(file_path = SCRIPT_DIR, simulation_folder = UNBIASED_FOLDER):

    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')
    input=f'''# vim:ft=plumed
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs
# O(BAS): o1: 17, o2: 22, o3: 26, o4: 34,
# O(MeOH) o5: 38
# H(CH3): h4: 43, h5: 44, h7: 46
# H(OH): h2: 39
# H(CH2): h3: 42, h6: 45
# H(BAS): h1: 37
# C: c1: 40, c2: 41
# DISTANCES between O(BAS) and H(CH3)
o4h7: DISTANCE ATOMS=34,46
o4h4: DISTANCE ATOMS=34,43
o4h5: DISTANCE ATOMS=34,44

o2h7: DISTANCE ATOMS=22,46
o2h4: DISTANCE ATOMS=22,43
o2h5: DISTANCE ATOMS=22,44

o3h7: DISTANCE ATOMS=26,46
o3h4: DISTANCE ATOMS=26,43
o3h5: DISTANCE ATOMS=26,44

o1h7: DISTANCE ATOMS=17,46
o1h4: DISTANCE ATOMS=17,43
o1h5: DISTANCE ATOMS=17,44

# DISTANCES between O(BAS) and H(CH2)
o4h3: DISTANCE ATOMS=34,42
o4h6: DISTANCE ATOMS=34,45

o2h3: DISTANCE ATOMS=22,42
o2h6: DISTANCE ATOMS=22,45

o3h3: DISTANCE ATOMS=26,42
o3h6: DISTANCE ATOMS=26,45

o1h3: DISTANCE ATOMS=17,42
o1h6: DISTANCE ATOMS=17,45

# DISTANCES between O(BAS) and H(OH)
o4h2: DISTANCE ATOMS=34,39
o2h2: DISTANCE ATOMS=22,39
o3h2: DISTANCE ATOMS=26,39
o1h2: DISTANCE ATOMS=17,39

# DISTANCES between O(BAS) and H(BAS)
o4h1: DISTANCE ATOMS=34,37
o2h1: DISTANCE ATOMS=22,37
o3h1: DISTANCE ATOMS=26,37
o1h1: DISTANCE ATOMS=17,37

# DISTANCES between O(MeOH) and C
o5c1: DISTANCE ATOMS=38,40
o5c2: DISTANCE ATOMS=38,41

# DISTANCES between O(MeOH) and H(CH3)
o5h7: DISTANCE ATOMS=38,46
o5h4: DISTANCE ATOMS=38,43
o5h5: DISTANCE ATOMS=38,44

# DISTANCES between O(MeOH) and H(CH2)
o5h3: DISTANCE ATOMS=38,42
o5h6: DISTANCE ATOMS=38,45

# DISTANCES between O(MeOH) and H(O)
o5h1: DISTANCE ATOMS=38,37
o5h2: DISTANCE ATOMS=38,39


# DISTANCE between atom 7 and 38
d17: DISTANCE ATOMS=7,38

# Apply upper wall to the distance between 7 and 38
uwall: UPPER_WALLS ARG=d17 AT=3.5 KAPPA=200.0

# PRINT all variables

PRINT FMT=%g STRIDE=10 FILE={simulation_folder}/COLVAR ARG=o4h7,o4h4,o4h5,o2h7,o2h4,o2h5,o3h7,o3h4,o3h5,o1h7,o1h4,o1h5,o4h3,o4h6,o2h3,o2h6,o3h3,o3h6,o1h3,o1h6,o4h2,o2h2,o3h2,o1h2,o4h1,o2h1,o3h1,o1h1,o5c1,o5c2,o5h7,o5h4,o5h5,o5h3,o5h6,o5h1,o5h2'''
    print(input, file=file)
    file.close()



def gen_plumed_chaba_biased(model_name : str,
                         file_path : str,
                         simulation_folder,
                         pos,
                         skew,
                         kappa,
                         offset):

    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')
    input=f'''# vim:ft=plumed
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs
# O(BAS): o1: 17, o2: 22, o3: 26, o4: 34,
# O(MeOH) o5: 38
# H(CH3): h4: 43, h5: 44, h7: 46
# H(OH): h2: 39
# H(CH2): h3: 42, h6: 45
# H(BAS): h1: 37
# C: c1: 40, c2: 41
# DISTANCES between O(BAS) and H(CH3)
o4h7: DISTANCE ATOMS=34,46
o4h4: DISTANCE ATOMS=34,43
o4h5: DISTANCE ATOMS=34,44

o2h7: DISTANCE ATOMS=22,46
o2h4: DISTANCE ATOMS=22,43
o2h5: DISTANCE ATOMS=22,44

o3h7: DISTANCE ATOMS=26,46
o3h4: DISTANCE ATOMS=26,43
o3h5: DISTANCE ATOMS=26,44

o1h7: DISTANCE ATOMS=17,46
o1h4: DISTANCE ATOMS=17,43
o1h5: DISTANCE ATOMS=17,44

# DISTANCES between O(BAS) and H(CH2)
o4h3: DISTANCE ATOMS=34,42
o4h6: DISTANCE ATOMS=34,45

o2h3: DISTANCE ATOMS=22,42
o2h6: DISTANCE ATOMS=22,45

o3h3: DISTANCE ATOMS=26,42
o3h6: DISTANCE ATOMS=26,45

o1h3: DISTANCE ATOMS=17,42
o1h6: DISTANCE ATOMS=17,45

# DISTANCES between O(BAS) and H(OH)
o4h2: DISTANCE ATOMS=34,39
o2h2: DISTANCE ATOMS=22,39
o3h2: DISTANCE ATOMS=26,39
o1h2: DISTANCE ATOMS=17,39

# DISTANCES between O(BAS) and H(BAS)
o4h1: DISTANCE ATOMS=34,37
o2h1: DISTANCE ATOMS=22,37
o3h1: DISTANCE ATOMS=26,37
o1h1: DISTANCE ATOMS=17,37

# DISTANCES between O(MeOH) and C
o5c1: DISTANCE ATOMS=38,40
o5c2: DISTANCE ATOMS=38,41

# DISTANCES between O(MeOH) and H(CH3)
o5h7: DISTANCE ATOMS=38,46
o5h4: DISTANCE ATOMS=38,43
o5h5: DISTANCE ATOMS=38,44

# DISTANCES between O(MeOH) and H(CH2)
o5h3: DISTANCE ATOMS=38,42
o5h6: DISTANCE ATOMS=38,45

# DISTANCES between O(MeOH) and H(O)
o5h1: DISTANCE ATOMS=38,37
o5h2: DISTANCE ATOMS=38,39


# DISTANCE between atom 7 and 38
d17: DISTANCE ATOMS=7,38

# Apply upper wall to the distance between 7 and 38
uwall: UPPER_WALLS ARG=d17 AT=3.5 KAPPA=200.0
cv: PYTORCH_MODEL FILE={model_name} ARG=o4h7,o4h4,o4h5,o2h7,o2h4,o2h5,o3h7,o3h4,o3h5,o1h7,o1h4,o1h5,o4h3,o4h6,o2h3,o2h6,o3h3,o3h6,o1h3,o1h6,o4h2,o2h2,o3h2,o1h2,o4h1,o2h1,o3h1,o1h1,o5c1,o5c2,o5h7,o5h4,o5h5,o5h3,o5h6,o5h1,o5h2

# UPPER_WALLS ARG=c1c2 AT=+8.5 KAPPA=250.0 EXP=2 LABEL=constr_c1c2 # Wall for potential constraints
    '''
    print(input, file=file)
    file.close()
    walltype=""
    if skew < 0:
        walltype = "UPPER_WALLS"
        offset = -offset
    else:
        walltype = "LOWER_WALLS"
    with open(file_path,"a") as f:
        print(f"""
# Energy wall for aes cv
wall: {walltype} ARG=cv.node-0 AT={pos+offset} KAPPA={kappa} ExP=2 EPS=1 OFFSET=0.0
PRINT FMT=%g STRIDE=10 FILE={simulation_folder}/COLVAR ARG=o4h7,o4h4,o4h5,o2h7,o2h4,o2h5,o3h7,o3h4,o3h5,o1h7,o1h4,o1h5,o4h3,o4h6,o2h3,o2h6,o3h3,o3h6,o1h3,o1h6,o4h2,o2h2,o3h2,o1h2,o4h1,o2h1,o3h1,o1h1,o5c1,o5c2,o5h7,o5h4,o5h5,o5h3,o5h6,o5h1,o5h2,cv.*""",file=f)


def chaba_simulation(iter_folder, model_name, model, dataset, kappa, offset):
    nn_output = cv_eval(model, dataset).flatten()
    mu_sknn = np.mean(nn_output)
    var_sknn = np.var(nn_output)
    skew_sknn = stats.skew(nn_output)
    offset += np.sqrt(var_sknn)
    gen_plumed_chaba_biased(model_name=model_name,
                         file_path=".",
                         simulation_folder=iter_folder,
                         pos=mu_sknn,
                         skew=skew_sknn,
                         kappa=kappa,
                         offset=offset)



def main(kappa):
    n_max_iter = 40
    loss_coeff = 0.1
    torch.manual_seed(22)
    batch_size = 100
    offset = 1.0

    subprocess.run([*bash_prefix,f"rm -rf {RESULTS_FOLDER}"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix,f"rm -rf {LIGHTNING_LOGS}"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix,f"rm -rf {UNBIASED_FOLDER}"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"rm -f {kappa}_iter* all*.pdb"], cwd=SCRIPT_DIR)

    subprocess.run([*bash_prefix, f"mkdir -p {UNBIASED_FOLDER}"])

    subprocess.run([*bash_prefix, f"echo '******************************************************'"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"echo Start unbiased simulation"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"echo '******************************************************'"], cwd=SCRIPT_DIR)
    gen_plumed_chaba_unbiased()
    subprocess.run([*bash_prefix,"srun -n 12 cp2k.popt job.inp > output.log"], cwd=SCRIPT_DIR)

    # For command line interface testing
    # subprocess.run([*bash_prefix,"cp2k.popt job.inp > output.log"], cwd=SCRIPT_DIR)

    subprocess.run([*bash_prefix, "mv Chaba-1.restart newiter.restart"], cwd = SCRIPT_DIR)
    subprocess.run([*bash_prefix, "rm -f Chaba*.restart"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"mv Chaba-pos-1.pdb {kappa}_iteration_Chaba_unbiased-pos.pdb"], cwd = SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"cat {kappa}_iteration_Chaba_unbiased-pos.pdb > all_{kappa}.pdb"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix,"rm -f PLUMED.OUT Chaba*"], cwd=SCRIPT_DIR)

    bond_type_dict, n_descriptors,heavy_atom_pairs_list = STADECT.parse_unbiased_colvar(colvar_file = f"{UNBIASED_FOLDER}/COLVAR")
    encoder_layers = [n_descriptors, 90, 40, 20, 5, 1]
    print(bond_type_dict)
    state_detection = STADECT.State_detection((0.3, 0.7), bond_type_dict=bond_type_dict, n_heavy_atom_pairs=n_descriptors)

    subprocess.run([*bash_prefix, f"mkdir -p {RESULTS_FOLDER}"])

    for iter in range(n_max_iter):
        state_detection, model, ITER_FOLDER, skewness_dataset, break_flag = chaba_training(state_detection, iter, encoder_layers, loss_coeff, batch_size)
        subprocess.run([*bash_prefix, f"echo '******************************************************'"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"echo At the iteration {iter} training step, "], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"echo The current state is {state_detection.current_state}"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"echo '******************************************************'"], cwd=SCRIPT_DIR)
        model_name = f"{ITER_FOLDER}/model_autoencoder_{iter}.pt"
        chaba_simulation(ITER_FOLDER, model_name, model, skewness_dataset, kappa, offset)
        subprocess.run([*bash_prefix,"srun -n 12 cp2k.popt job_restart.inp > output.log"], cwd=SCRIPT_DIR)
        # For command line interface testing
        # subprocess.run([*bash_prefix,"cp2k.popt job_restart.inp > output.log"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, "mv Chaba-1.restart newiter.restart"], cwd = SCRIPT_DIR)
        subprocess.run([*bash_prefix, "rm -f Chaba*.restart"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"mv Chaba-pos-1.pdb {kappa}_iteration_Chaba_{iter}-pos.pdb"], cwd = SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"tail -n +3 {kappa}_iteration_Chaba_{iter}-pos.pdb >> all_{kappa}.pdb"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix,"rm -f PLUMED.OUT Chaba*"], cwd=SCRIPT_DIR)
        subprocess.run([*bash_prefix, f"echo CP2K simulation at iteration {iter} with plumed ends"], cwd=SCRIPT_DIR)

    print(f"there are in total {state_detection.n_states} states")

if __name__ == "__main__":
    kappa = np.int_(sys.argv[1])
    main(kappa)
    subprocess.run([*bash_prefix, "rm -f bck* Chaba* *.OUT"], cwd=SCRIPT_DIR)
