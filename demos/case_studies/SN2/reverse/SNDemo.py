# IMPORT PACKAGES
# import pdb
import torch
from torch.utils.data import DataLoader
import sys
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import subprocess
import ast

# IMPORT from MLCVS
from mlcolvar.data import DictModule, DictLoader
from mlcolvar.core.transform import Normalization
from mlcolvar.core.transform.utils import Statistics
from mlcolvar.utils.fes import compute_fes
from mlcolvar.utils.io import create_dataset_from_files, load_dataframe
from utils.plot import muller_brown_potential_three_states, plot_isolines_2D, plot_metrics, paletteFessa
from mlcolvar.utils.trainer import MetricsCallback


# IMPORT utils functions fo input generation
from utils.generate_input import gen_input_md_aes,gen_plumed_aes_2,gen_plumed_aes_3
from mlcolvar.cvs import MultiTaskCV
from mlcolvar.cvs import AutoEncoderCV
from mlcolvar.core.loss import TDALoss

from skewloss import *

from scipy import stats
from scipy.special import softmax


def load_data(filenames_iter,filenames_all, multiple = 0, bs=0):
    # load for AE, filenames_all can be a list so that all data are loaded
    # TODO: verify if both variables can be lists
    AE_dataset, AE_df = create_dataset_from_files(filenames_all, return_dataframe=True, filter_args={'regex':'d1|d5'}, create_labels=False, verbose=True)
    # AE_dataset, AE_df = create_dataset_from_files(filenames_all, return_dataframe=True, filter_args={'regex':'d2|d3|d_10'}, create_labels=False, verbose=True)
    skewness_dataset, skewness_df = create_dataset_from_files(filenames_iter, return_dataframe=True, filter_args={'regex':'d1|d5'}, create_labels=False, verbose=True)
    # create multitask datamodule with both datasets
    if bs == 0:
        batch_size_list = bs
    else:
        batch_size_list=[[bs * multiple, bs],[bs * multiple, bs]]
    datamodule = DictModule(dataset=[AE_dataset, skewness_dataset], batch_size=batch_size_list)
    return AE_dataset, skewness_dataset, datamodule, AE_df, skewness_df

def aes_model(AE_dataset, encoder_layers, loss_coeff, iter=0, PREV_ITER_FOLDER="", *args):
    nn_args = {'activation': 'shifted_softplus'}
    optimizer_settings= {'weight_decay': 1e-5}
    options= {'encoder': nn_args, 'decoder': nn_args, 'optimizer': optimizer_settings}
    main_cv = AutoEncoderCV(encoder_layers, options=options)
    # encoder_layers =  [n_descriptors, 32, 16, n_components]
    aux_loss_fn = SkewLoss()

    model = MultiTaskCV(main_cv, auxiliary_loss_fns=[aux_loss_fn], loss_coefficients=[loss_coeff])

    # ##TODO##: must normalization. If accumulate all data, I think normalization will affect the skewness. So far no normalization
    if iter > 0:
        main_cv = model.__class__.load_from_checkpoint(checkpoint_path=f"{PREV_ITER_FOLDER}/checkpoint.ckpt", main_cv=main_cv, auxiliary_loss_fns=[aux_loss_fn], loss_coefficients=[loss_coeff])
    stat = Statistics()
    stat(AE_dataset['data'])
    model.norm_in.set_from_stats(stat)
    return model

# Define training and fit
def aes_trainer(model, datamodule, iter_folder):
    # define callbacks
    metrics = MetricsCallback()
    early_stopping = EarlyStopping(monitor="valid_loss", min_delta=1e-5, patience=10)
    # define trainer
    trainer = pl.Trainer(accelerator='cpu',callbacks=[metrics, early_stopping], max_epochs=1000,
                         enable_checkpointing=False, enable_model_summary=False, log_every_n_steps=10)
    # fit
    trainer.fit( model,datamodule)
    trainer.save_checkpoint(f"{iter_folder}/checkpoint.ckpt",weights_only=True)
    return metrics


def cv_eval(model, dataset):
    # data here is a torch tensor
    # TODO: extend the adaption of more data structure of data
    # data so far is a n * d matrix of which d indicates the dimension of data
    # TODO: add exception detection for if judgement
    data = dataset["data"]
    output = np.zeros(data.size(dim=1))
    with torch.no_grad():
        train_mode = model.training
        model.eval()
        output = model(data).numpy()
        model.training = train_mode
    return output

# TODO: develop some function setting kappa and offset
def gen_kappa_offset():
    kappa = 100
    offset = 1.0
    return kappa, offset


# Run plumed simulation
def aes_run_plumed(iter_folder, model_name, model, dataset):
    # TODO: so far 1 d
    nn_output = cv_eval(model,dataset).flatten()

    mu_aes = np.average(nn_output)
    var_aes = np.var(nn_output)
    skew_aes = stats.skew(nn_output)
    # create folder
    SIMULATION_FOLDER = f'{iter_folder}/data'
    subprocess.run(f"mkdir {SIMULATION_FOLDER}", shell=True)
    kappa, offset = gen_kappa_offset()
    offset = offset + np.sqrt(var_aes)
    # generate inputs
    gen_plumed_aes_3(model_name=model_name,
            file_path=".",
            simulation_folder=SIMULATION_FOLDER,
            pos=mu_aes,
            skew=skew_aes,
            kappa=kappa,
            offset=offset)
    return SIMULATION_FOLDER

# set init pos w.r.t different functions
def set_initial_position(iter, fun, last_conf=[0.,0.]):
    # TODO: if none of these functions match, raise an exception
    initial_positions = ""
    # muller_brown
    if fun == 1:
        if iter == 0:
            initial_positions = '-0.7,1.4'
        else:
            initial_positions = f"{last_conf[0]},{last_conf[1]}"
    return initial_positions

def plot_training_points(AE_df, skewness_df,iter_folder, iter):
    data_ref= load_dataframe("./fes-rew.dat")
    # plot_isolines_2D(muller_brown_potential_three_states,mode='contour',levels=np.linspace(0,24,12),ax=ax)
    d1 = np.array(data_ref.iloc[:]["d1"].values).reshape(101,101)
    d5 = np.array(data_ref.iloc[:]["d5"].values).reshape(101,101)
    fes = np.array(data_ref.iloc[:]["file.free"].values).reshape(101,101)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.contour(d1, d5, fes, levels=np.linspace(0,150,15), linewidths=1, cmap="Greys_r")
    # AE_df.plot.scatter('p.x','p.y',s=1,cmap='fessa',ax=ax)
    ax.scatter(AE_df['d5'],AE_df['d1'], color='blue', s=1, label='AE training data')
    # skewness_df.plot.scatter('p.x','p.y',s=1,ax=ax)
    ax.scatter(skewness_df['d5'],skewness_df['d1'], color='red', s=1, label='skewness training data')
    # ax.set_title(f'Training set - {iter}')
    # ax.legend()
    ax.set_xlim(1.2,4.2)
    ax.set_ylim(1.6,4.5)
    ax.set_xticks([2,3,4])
    ax.set_yticks([2,3,4])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    plt.savefig(f'{iter_folder}/training_set-{iter}.png')
    # plt.show()

# Analysis of CV
def ae_cv_isolines(model, n_components, iter_folder, iter):
    fig,axs = plt.subplots( 1, n_components, figsize=(4*n_components,3) )
    if n_components == 1:
        axs = [axs]
    for i in range(n_components):
        ax = axs[i]
        # plot_isolines_2D(muller_brown_potential_three_states,levels=np.linspace(0,24,12),mode='contour',ax=ax)
        plot_isolines_2D(model, component=i, levels=25, ax=ax)
        plot_isolines_2D(model, component=i, mode='contour', levels=25, ax=ax)
    ax.set_title(f'CV isolines - {iter}')
    plt.savefig(f'{iter_folder}/cv_isolines.png')
    # plt.show()

# Visualize Sampling
def ae_visualize_sampling(simulation_folder, iter):
    data = load_dataframe(f'{simulation_folder}/COLVAR')
    first_conf = data.iloc[0][['phi', 'psi']].values
    last_conf = data.iloc[-1][['phi', 'psi']].values
    fig, ax = plt.subplots(figsize=(4,3))
    # data.plot.hexbin('p.x', 'p.y', C='opes.bias',cmap='fessa', ax=ax)
    ax.scatter(first_conf[0], first_conf[1], s=20, c='cyan')
    ax.scatter(last_conf[0], last_conf[1], s=20, c='magenta')
    ax.set_title(f'Sampling - {iter}')
    plt.savefig(f'{simulation_folder}/sampling.png')

def ae_visualize_sampling_all(simulation_folder, result_folder, iter, AE_df):
    data = load_dataframe(f'{simulation_folder}/COLVAR')
    # data_ref= load_dataframe("./fes_ref.dat")
    # d1 = np.array(data_ref.iloc[:]["d1"].values).reshape(100,100)
    # d5 = np.array(data_ref.iloc[:]["d5"].values).reshape(100,100)
    # fes = np.array(data_ref.iloc[:]["file.free"].values).reshape(100,100)
    fig, ax = plt.subplots(figsize=(4,3))
    # ax.contour(d1, d5, fes, levels=np.linspace(0,50,10), linewidths=1, cmap="Greys_r")
    # data.plot.hexbin('p.x', 'p.y', C='opes.bias',cmap='fessa', ax=ax)
    ax.scatter(AE_df['d1'],AE_df['d5'], color='blue', s=1)
    ax.scatter(data.iloc[:]['d1'].values, data.iloc[:]['d5'].values, color='red', s=1)
    # ax.scatter(first_conf[0], first_conf[1], s=10, c='cyan')
    # ax.scatter(last_conf[0], last_conf[1], s=10, c='magenta')
    ax.set_xlim(1.0,4.0)
    ax.set_ylim(1.0,4.0)
    # ax.set_title(f'Sampling - {iter}')
    plt.savefig(f'{result_folder}/sampling-{iter}.png')

def aes_normalization(model, dataset, n_components):
    X = dataset[:]['data']
    with torch.no_grad():
        model.postprocessing = None # reset
        s = model(torch.Tensor(X))
    norm =  Normalization(n_components, mode='min_max', stats = Statistics(s) )
    model.postprocessing = norm
    return model

def check_if_in_which_basin(PREV_FOLDER_COLVAR):
    switch = np.average(load_dataframe(f'{PREV_FOLDER_COLVAR}').iloc[300:][['f1.contact-1', 'f1.contact-2']].values, axis=0)
    if switch[0] < 0.25 and switch[1] > 0.75:
        print("In basin 1")
        print("In basin 1")
        print("In basin 1")
        return 1
    elif switch[1] < 0.25 and switch[0] > 0.75:
        print("In basin 2")
        print("In basin 2")
        print("In basin 2")
        return 2
    else:
        print("Not in any basin, keep warm starting")
        print("Not in any basin, keep warm starting")
        print("Not in any basin, keep warm starting")
        return 0


# Training and sampling for restarting at state 1
# TODO: so far no bias for avoiding the system going back to the state 0. In this case go back to state_0
# Combine restart function with the following function
def training_and_sampling(state, states, iter, n_descriptors, n_components, loss_coeff, batch_size=0, use_all_data = True):
    # TODO: adjust the folder name to the function name
    RESULTS_FOLDER = f"results"
    UNBIASED_FOLDER = f"unbiased"
    # procedure parameters
    encoder_layers = [n_descriptors, 60, 30 , n_components]
    # create folder for current iter
    ITER_FOLDER = RESULTS_FOLDER+f'/iter_{iter}'
    subprocess.run(f"mkdir {ITER_FOLDER}", shell=True)
    break_flag = False
    if iter == 0:
        # Here we can set the folder name as constant
        filenames_iter = [f"{UNBIASED_FOLDER}/COLVAR"]
        filenames_all = filenames_iter
    else:
        if use_all_data:
            filenames_all = [f"{RESULTS_FOLDER}/iter_{i}/data/COLVAR" for i in range(iter) ]
            filenames_all.append(f"{UNBIASED_FOLDER}/COLVAR")
        else:
            filenames_all = [f"{RESULTS_FOLDER}/iter_{iter-1}/data/COLVAR"]
        filenames_iter = [f"{RESULTS_FOLDER}/iter_{iter-1}/data/COLVAR"]
    # 1 - Load and visualize unlabeled data
    # TODO: 2 data frames both needs plot
    AE_dataset, skewness_dataset, datamodule, AE_df, skewness_df = load_data(filenames_iter,filenames_all, multiple=(iter + 1), bs=batch_size)
    plot_training_points(AE_df, skewness_df, RESULTS_FOLDER, iter)
    # 2 - Initialize model
    if iter == 0:
        model = aes_model(AE_dataset,encoder_layers, loss_coeff)
    else:
        PREV_ITER_FOLDER = f"{RESULTS_FOLDER}/iter_{iter-1}"
        current_state = check_if_in_which_basin(filenames_iter[0])
        reach_new_state = False
        apply_warm_start = True
        if current_state not in states:
            states.append(current_state)
            reach_new_state = True
            apply_warm_start = False
            print("****************************")
            print("Reach another basin")
            print("Reach another basin")
            print("Reach another basin")
            print("****************************")
        else:
            if current_state == 1 and len(states) > 2:
                print("****************************")
                print(f"Went back to original basin. Simulation Terminated at iter {iter}")
                print(f"Went back to original basin. Simulation Terminated at iter {iter}")
                print(f"Went back to original basin. Simulation Terminated at iter {iter}")
                print("****************************")
                break_flag = True
        if not reach_new_state:
            if state == 0 and current_state != state:
                apply_warm_start = False

        state = current_state
        if not apply_warm_start:
            print("****************************")
            print("Restart from Scratch")
            print("Restart from Scratch")
            print("Restart from Scratch")
            print("****************************")
            model = aes_model(AE_dataset,encoder_layers, loss_coeff)
        else:
            print("****************************")
            print("Apply Warm Start")
            print("Apply Warm Start")
            print("Apply Warm Start")
            print("****************************")
            model = aes_model(AE_dataset,encoder_layers, loss_coeff,iter=iter,PREV_ITER_FOLDER=PREV_ITER_FOLDER)
    # 3 - Initialize trainer and tracer
    metrics = aes_trainer(model, datamodule, iter_folder=ITER_FOLDER)

    # 4 - Apply normalization on the output
    # TODO: Determine if normalization is necessary
    model = aes_normalization(model, AE_dataset, n_components)
    # 5 - Export and visualize the model
    traced_model = model.to_torchscript(file_path=f'{ITER_FOLDER}/model_autoencoder_{iter}.pt', method='trace')
    model_name = f"{ITER_FOLDER}/model_autoencoder_{iter}.pt"
    # ae_cv_isolines(model, n_components, ITER_FOLDER, iter)


    # 6 - RUM PLUMED simulation
    SIMULATION_FOLDER = aes_run_plumed(ITER_FOLDER, model_name, model=model, dataset=skewness_dataset)
    # ae_visualize_sampling_all(SIMULATION_FOLDER, RESULTS_FOLDER, iter, AE_df)
    # ae_visualize_sampling(SIMULATION_FOLDER, iter)
    return state, break_flag, states
def main():
    iter = np.int_(sys.argv[1])
    loss_coeff= np.float32(sys.argv[2])
    state = int(sys.argv[3])
    states_str = sys.argv[4]
    states = [ int(state_i) for state_i in ast.literal_eval(states_str) ]
    print(f"states = {states}")
    n_descriptors = 2
    n_components = 1
    torch.manual_seed(22)
    batch_size = 100
    state, break_flag, states = training_and_sampling(state, states, iter, n_descriptors, n_components, loss_coeff, batch_size)
    return state, break_flag, states


if __name__ == "__main__":
    state, break_flag, states = main()
    print("*************************")
    print(f"final output= {state} {break_flag} {states}")
