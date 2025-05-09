# IMPORT PACKAGES
import torch
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import subprocess

# IMPORT from MLCVS
from mlcolvar.data import DictModule
from mlcolvar.core.transform import Normalization
from mlcolvar.core.transform.utils import Statistics
from mlcolvar.utils.fes import compute_fes
from mlcolvar.utils.io import create_dataset_from_files, load_dataframe
from utils.plot import elongated_potential, opes_model_potential, plot_isolines_2D, plot_metrics, paletteFessa
from mlcolvar.utils.trainer import MetricsCallback


# IMPORT utils functions fo input generation
from utils.generate_input import gen_input_md_aes,gen_plumed_aes
from mlcolvar.cvs import MultiTaskCV
from mlcolvar.cvs import AutoEncoderCV
from mlcolvar.core.loss import TDALoss

from skewloss import *

from scipy import stats
from scipy.special import softmax



def ELOGATED_FORMULAR():
    A=2.0
    x1=0.0
    x2=20.0
    B=0.25
    y1=20.0
    C=0.25
    y2=0.0
    D=2.5
    scalar=4
    return f"1/(1+exp(-(y-x)/{scalar}))*({A}*(x-{x1})^2+{B}*(y-{y1})^2)+(1-1/(1+exp(-(y-x)/{scalar})))*({C}*(x-{x2})^2+{D}*(y-{y2})^2)"

def OPES_TUTOR_MODEL():
    return "1.34549*x^4+1.90211*x^3*y+3.92705*x^2*y^2-6.44246*x^2-1.90211*x*y^3+5.58721*x*y+1.33481*x+1.34549*y^4-5.55754*y^2+0.904586*y+19"
def load_data(filenames_iter,filenames_all, bs = 0, multiple = 0):
    # load for AE, filenames_all can be a list so that all data are loaded
    AE_dataset, AE_df = create_dataset_from_files(filenames_all, return_dataframe=True, filter_args={'regex':'p.x|p.y'}, create_labels=False, verbose=False)
    # load labeled data
    skewness_dataset, skewness_df = create_dataset_from_files(filenames_iter,return_dataframe=True, filter_args={'regex':'p.x|p.y'}, create_labels=False, verbose=False)
    if bs == 0:
        batch_size_list = bs
    else:
        batch_size_list = [[bs * multiple, bs],[bs * multiple, bs]]
    # create multitask datamodule with both datasets
    print("batch_size_list is: ", batch_size_list)
    datamodule = DictModule(dataset=[AE_dataset, skewness_dataset], lengths = (0.8, 0.2), batch_size = batch_size_list)
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
    trainer.fit( model, datamodule )
    trainer.save_checkpoint(f"{iter_folder}/checkpoint.ckpt",weights_only=True)
    return metrics


def cv_eval(model, dataset):
    # data here is a torch tensor
    data = dataset["data"]
    output = np.zeros(data.size(dim=1))
    with torch.no_grad():
        train_mode = model.training
        model.eval()
        output = model(data).numpy()
        model.training = train_mode
    return output

### for mueller brown potential not 3 states
# def gen_kappa_offset():
#     kappa = 4.2
#     offset = 1.2
#     return kappa, offset

def gen_kappa_offset(iter, max_iter, kappa_all=4.0):
    smallest_kappa = kappa_all / 2
    kappa = 4.4 # smallest_kappa + (kappa_all - smallest_kappa) * iter / max_iter
    # kappa = kappa_all * (iter + 1) / max_iter
    offset = 1.0
    return kappa, offset

# TODO: develop some function setting kappa and offset
# def gen_kappa_offset():
#     kappa = 25.0
#     offset = 0.7
#     return kappa, offset


# Run plumed simulation
def aes_run_plumed(iter, iter_folder, initial_position, nsteps, model, dataset, max_iter):
    # TODO: so far 1 d
    nn_output = cv_eval(model,dataset).flatten()
    
    mu_aes = np.average(nn_output)
    # mu_aes = mean_cutoff(PREV_SIMU_FOLDER, model)
    var_aes = np.var(nn_output)
    skew_aes = stats.skew(nn_output)

    # create folder
    SIMULATION_FOLDER = f'{iter_folder}/data'
    subprocess.run(f"mkdir {SIMULATION_FOLDER}", shell=True)
    PLUMED_EXE = "plumed"

    kappa, offset = gen_kappa_offset(iter, max_iter)
    offset = offset + np.sqrt(var_aes)
    # generate inputs
    gen_plumed_aes(model_name=f'model_autoencoder_{iter}.pt',
            file_path=SIMULATION_FOLDER,
            potential_formula=OPES_TUTOR_MODEL(),
            pos=mu_aes,
            skew=skew_aes,
            kappa=kappa,
            offset=offset)
    gen_input_md_aes(initial_position=initial_position, file_path=SIMULATION_FOLDER, nsteps=nsteps)

    subprocess.run(f'{PLUMED_EXE} pesmd < input_md.dat', cwd=SIMULATION_FOLDER, shell=True, executable='/bin/bash')
    last_conf = load_dataframe(f'{SIMULATION_FOLDER}/COLVAR').iloc[-1][['p.x', 'p.y']].values
    return last_conf, SIMULATION_FOLDER, skew_aes

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
    fig,ax = plt.subplots(figsize=(4,3))
    # plot_isolines_2D(muller_brown_potential_three_states,mode='contour',levels=np.linspace(0,24,12),ax=ax)
    plot_isolines_2D(opes_model_potential,limits=((-3.0, 3.0),(-2.8,2.8)),mode='contour',levels=np.linspace(0,30,12),ax=ax)
    # AE_df.plot.scatter('p.x','p.y',s=1,cmap='fessa',ax=ax)
    ax.scatter(AE_df['p.x'],AE_df['p.y'], color='blue', s=1, label='AE training data')
    # skewness_df.plot.scatter('p.x','p.y',s=1,ax=ax)
    ax.scatter(skewness_df['p.x'],skewness_df['p.y'], color='red', s=1, label='skewness training data')
    # ax.set_title(f'Training set - {iter}')
    # ax.legend()
    plt.savefig(f'{iter_folder}/training_set_{iter}.png')
    # plt.show()

# Analysis of CV
def ae_cv_isolines(model, n_components, iter_folder, iter, dataset):
    cm = 1/2.54
    figure_width = 3.5*4*cm
    plt.rcParams['figure.autolayout'] = True  # Enable tight_layout
    plt.rcParams.update({'font.size': 32})    # Base padding on font size 14

    # Customize font size, line width, and other properties
    ######################################################
    line_width = 3
    axis_thick = 7.5
    dot_size = 16
    axis_label_fontsize = 32
    tick_label_fontsize = 32
    ########################################################################################################################
    plt.rcParams['mathtext.default']= "regular"
    ax_pos = [0.18, 0.18, 0.75, 0.75]
    fig,axs = plt.subplots( 1, n_components, figsize=(figure_width, figure_width), sharey=True, constrained_layout=True)
    if n_components == 1:
        axs = [axs]
    for i in range(n_components):
        ax = axs[i]
        ax.set_position(ax_pos) 
        # plot_isolines_2D(muller_brown_potential_three_states,levels=np.linspace(0,24,12),mode='contour',ax=ax)
        plot_isolines_2D(opes_model_potential,limits=((-3.0, 3.0),(-3.0, 3.0)), levels=np.linspace(0,24,8),mode='contour',ax=ax, linewidths=line_width-0.5)
        plot_isolines_2D(model, component=i,limits=((-3.0, 3.0),(-3.0, 3.0)), levels=25, colorbar=False, ax=ax, alpha=0.2,  linewidths=line_width-0.5)
        plot_isolines_2D(model, component=i, limits=((-3.0, 3.0),(-3.0, 3.0)),mode="scatter", levels=25, ax=ax, dataset=dataset, s=dot_size)
        plot_isolines_2D(model, component=i,limits=((-3.0, 3.0),(-3.0, 3.0)), mode='contour', levels=25, ax=ax, alpha=0.5,  linewidths=line_width-0.5)
        ticks = [-3.0, -1.0, 1.0, 3.0]
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{tick:.1f}' for tick in ticks])
        ax.set_yticklabels([f'{tick:.1f}' for tick in ticks])
        ax.set_xlabel(r"$x$", fontsize=axis_label_fontsize, labelpad=-20)
        ax.set_ylabel(r"$y$", fontsize=axis_label_fontsize, labelpad=-40)
        ax.tick_params(axis='x', labelsize=tick_label_fontsize)
        ax.tick_params(axis='y', labelsize=tick_label_fontsize)
        ax.tick_params('x', length=8, width=2, which='major')
        ax.tick_params('y', length=8, width=2, which='major')
        ax.spines['top'].set_linewidth(axis_thick)    # Top axis line
        ax.spines['bottom'].set_linewidth(axis_thick) # Bottom axis line
        ax.spines['left'].set_linewidth(axis_thick)   # Left axis line
        ax.spines['right'].set_linewidth(axis_thick)  # Right axis line
        
        # Make each subplot square by setting aspect ratio
        ax.set_aspect('equal')
    # ax.set_title(f'CV isolines - {iter}')
    plt.savefig(f'{iter_folder}/cv_isolines_{iter}.png')
    plt.rcParams.update({'font.size': 14})    # Base padding on font size 14
    # ax.set_title(f'CV isolines - {iter}')
    # plt.show()

# Visualize Sampling
def ae_visualize_sampling(simulation_folder, iter):
    data = load_dataframe(f'{simulation_folder}/COLVAR')
    first_conf = data.iloc[0][['p.x', 'p.y']].values
    last_conf = data.iloc[-1][['p.x', 'p.y']].values
    fig, ax = plt.subplots(figsize=(4,3))
    data.plot.hexbin('p.x', 'p.y', C='opes.bias',cmap='fessa', ax=ax)
    ax.scatter(first_conf[0], first_conf[1], s=20, c='cyan')
    ax.scatter(last_conf[0], last_conf[1], s=20, c='magenta')
    ax.set_title(f'Sampling - {iter}')
    plt.savefig(f'{simulation_folder}/sampling.png')

def plot_2D_wall(
    function,
    kappa,
    pos,
    sign,
    component=None,
    limits=((-1.8, 1.2), (-0.4, 2.3)), # corrected based on certain molecule
    num_points=(100, 100),
    mode="contourf",
    levels=12,
    cmap=None,
    colorbar=None,
    max_value=None,
    ax=None,
    **kwargs,
):
    """Plot isolines of a function/model in a 2D space."""

    # Define grid where to evaluate function
    if type(num_points) == int:
        num_points = (num_points, num_points)
    xx = np.linspace(limits[0][0], limits[0][1], num_points[0])
    yy = np.linspace(limits[1][0], limits[1][1], num_points[1])
    xv, yv = np.meshgrid(xx, yy)

    # if torch module
    if isinstance(function, torch.nn.Module):
        z = np.zeros_like(xv)
        for i in range(num_points[0]):
            for j in range(num_points[1]):
                xy = torch.Tensor([xv[i, j], yv[i, j]])
                with torch.no_grad():
                    train_mode = function.training
                    function.eval()
                    s = function(xy).numpy()
                    if sign==True:
                        wall = kappa * (s - pos)**2 if s <= pos else 0.0
                    else:
                        wall = kappa * (s - pos)**2 if s >= pos else 0.0
                    function.training = train_mode
                    if component is not None:
                        s = s[component]
                z[i, j] = wall
    # else apply function directly to grid points
    else:
        z = function(xv, yv)

    if max_value is not None:
        z[z > max_value] = max_value

    # Setup plot
    return_axs = False
    if ax is None:
        return_axs = True
        _, ax = plt.subplots(figsize=(6, 4.0), dpi=100)

    # Color scheme
    if cmap is None:
        if mode == "contourf":
            cmap = "fessa"
        elif mode == "contour":
            cmap = "Greys_r"

    # Colorbar
    if colorbar is None:
        if mode == "contourf":
            colorbar = True
        elif mode == "contour":
            colorbar = False

    # Plot
    if mode == "contourf":
        pp = ax.contourf(xv, yv, z, levels=levels, cmap=cmap, **kwargs)
        if colorbar:
            plt.colorbar(pp, ax=ax)
    else:
        pp = ax.contour(xv, yv, z, levels=levels, cmap=cmap, **kwargs)

    if return_axs:
        return ax
    else:
        return None

def ae_cv_wall(model, n_components, iter_folder, iter, dataset,max_iter):
    fig,axs = plt.subplots(1, n_components, figsize=(4*n_components,3) )
    nn_output = cv_eval(model,dataset).flatten()
    mu_aes = np.average(nn_output)
    skew_aes = stats.skew(nn_output)
    var_aes = np.var(nn_output)
    kappa, offset = gen_kappa_offset(iter, max_iter)
    offset = offset + np.sqrt(var_aes)
    if skew_aes < 0:
        sign = False
        offset = -offset
    else:
        sign = True
    if n_components == 1:
        axs = [axs]
    pos=mu_aes+offset
    for i in range(n_components):
        ax = axs[i]
        # plot_isolines_2D(muller_brown_potential_three_states,levels=np.linspace(0,24,12),mode='contour',ax=ax)
        plot_isolines_2D(opes_model_potential, limits=((-3.0, 3.0),(-2.8,2.8)), levels=np.linspace(0,30,12),mode='contour',ax=ax)
        plot_2D_wall(model, kappa, pos, sign,component=i,limits=((-3.0, 3.0),(-2.8,2.8)), levels=np.linspace(0,10,12), colorbar=True, ax=ax, alpha=0.8)
    # ax.set_title(f'CV isolines - {iter}')
    plt.savefig(f'{iter_folder}/wall_{iter}.png')
    # plt.show()

def aes_normalization(model, dataset, n_components):
    X = dataset[:]['data']
    with torch.no_grad():
        model.postprocessing = None # reset
        s = model(torch.Tensor(X))
    norm =  Normalization(n_components, mode='min_max', stats = Statistics(s) )
    model.postprocessing = norm
    return model

def mean_cutoff(PREV_SIMU_FOLDER, model, n_cutoff=1000):
    df_cutoff = load_dataframe(f"{PREV_SIMU_FOLDER}/COLVAR").iloc[n_cutoff:][['p.x', 'p.y']].values
    tensor_cutoff = torch.Tensor(df_cutoff)
    print(tensor_cutoff.shape)
    with torch.no_grad():
        train_mode = model.training
        model.eval()
        s = model(tensor_cutoff).numpy()
        model.training = train_mode
    print(s.shape)
    mean = np.mean(s)
    return mean

def check_if_in_which_basin(PREV_FOLDER_COLVAR, tol = 0.3):
    position = np.average(load_dataframe(f'{PREV_FOLDER_COLVAR}').iloc[1500:][['p.x', 'p.y']].values, axis=0)
    basin1 = np.array([-1.87990887, 0.78403725])
    basin2 = np.array([ 1.78623828, -0.8312396]) 
    distance1 = np.sqrt(np.sum((basin1 - position) ** 2))
    distance2 = np.sqrt(np.sum((basin2 - position) ** 2))
    if np.min((distance1, distance2)) > tol:
        print("Not in any basin, keep warm starting")
        print("Not in any basin, keep warm starting")
        print("Not in any basin, keep warm starting")
        return 0
    else:
        if distance1 < distance2:
            print("In basin 1")
            print("In basin 1")
            print("In basin 1")
            return 1
        else:
            print("In basin 2")
            print("In basin 2")
            print("In basin 2")
            return 2
# Training and sampling test
def training_and_sampling_test(fun_name, max_iter, n_descriptors, n_components, loss_coeff, batch_size=0, use_all_data = True):
    # TODO: adjust the folder name to the function name
    RESULTS_FOLDER = f"results"

    subprocess.run(f"rm -r lightning_logs/", shell=True)
    subprocess.run(f"rm -r {RESULTS_FOLDER}", shell=True)
    subprocess.run(f"mkdir {RESULTS_FOLDER}", shell=True)

    # procedure parameters
    encoder_layers = [n_descriptors, 60, 30, n_components]
    skew_all = np.zeros(max_iter)
    state = 1
    states = [0, 1]
    break_flag = False

    for iter in range(max_iter):
        # create folder for current iteration
        ITER_FOLDER = RESULTS_FOLDER+f'/iter_{iter}'
        subprocess.run(f"mkdir {ITER_FOLDER}", shell=True)

        if iter == 0:
            # Here we can set the folder name as constant
            filenames_iter = [f'input_data/{fun_name}/unbiased/COLVAR']
            filenames_all = filenames_iter
        else:
            if use_all_data:
                filenames_all = [f"{RESULTS_FOLDER}/iter_{i}/data/COLVAR" for i in range(iter) ]
                filenames_all.append(f"input_data/{fun_name}/unbiased/COLVAR")
            else:
                filenames_all = [f"{RESULTS_FOLDER}/iter_{iter-1}/data/COLVAR"]
            filenames_iter = [f"{RESULTS_FOLDER}/iter_{iter-1}/data/COLVAR"]

        # 1 - Load and visualize unlabeled data
        # TODO: 2 data frames both needs plot
        AE_dataset, skewness_dataset, datamodule, AE_df, skewness_df = load_data(filenames_iter,filenames_all, bs = batch_size, multiple = iter + 1)
        plot_training_points(AE_df, skewness_df, ITER_FOLDER, iter)
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
        model = aes_normalization(model, AE_dataset, n_components)
        # 5 - Export and visualize the model
        traced_model = model.to_torchscript(file_path=f'{ITER_FOLDER}/model_autoencoder_{iter}.pt', method='trace')
        ae_cv_isolines(model, n_components, ITER_FOLDER, iter, AE_dataset)



        # 6 - RUM PLUMED simulation
        if iter == 0:
            initial_position = '-1.8,0.5'
        else:
            initial_position = f"{last_conf[0]},{last_conf[1]}"
        last_conf, SIMULATION_FOLDER, skew_current_iter = aes_run_plumed(iter, ITER_FOLDER, initial_position=initial_position, nsteps=20000, model=model, dataset=skewness_dataset,max_iter=max_iter)
        skew_all[iter] = np.abs(skew_current_iter)
        ae_cv_wall(model, n_components, SIMULATION_FOLDER, iter, skewness_dataset,max_iter=max_iter)
        subprocess.run(f"sed '2,101d' COLVAR > COLVAR_CUTOFF", cwd=SIMULATION_FOLDER, shell=True, executable='/bin/bash')
        # ae_visualize_sampling(SIMULATION_FOLDER, iter)
        if break_flag:
            break
    return skew_all

def plot_skewness(fun_name, max_iter, skew_all):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(np.arange(max_iter), skew_all, s=5, c='red')
    ax.set_title(f'Skewness')
    plt.savefig(f'./skew-{fun_name}.png')


if __name__ == "__main__":
    n_descriptors = 2
    n_components = 1
    fun_name= "opes_model"
    max_iter = 20
    # loss_coeff = 0.1 ## for normal MB potential
    loss_coeff = 0.1
    batch_size = 100
    torch.manual_seed(22)
    skew_all = training_and_sampling_test(fun_name, max_iter, n_descriptors, n_components, loss_coeff, batch_size, use_all_data = True)
    plot_skewness(fun_name, max_iter, skew_all)

