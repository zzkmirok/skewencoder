# Tutorial

If you are looking for a simple tutorial, you can start with following the step-by-step tutorial in the notebook `Tutorial/chabaDemo.ipynb`

## Settig up the Tutorial

```shell
mkdir -p tutorial4loxodynamics
cd tutorial4loxodynamics
git clone https://github.com/zzkmirok/skewencoder.git 
cp -r skewencoder/Tutorial . 
cd Tutorial
```

Before starting the tutorial in a jupyter notebook, make sure that the following installation requirements are satisfied.

> **Installation requirements for Jupyter Notebook Tutorial**
> 1. PLUMED version >= 2.9.0 and pytorch module linked with `libtorch` is required.
> 2. CP2K installation is required and is already patched with PLUMED.
> 3. Python version >= 3.10
> 4. `Jupyter` and `ipykernel` should be available in the Python.

In the `Tutorial` folder there is an example `activate.sh` file demonstrating how to use the `module` system for the configuration of required env variables.

```shell
source ./activate.sh # Not necessary
```

The step above can be skipped if CP2K patched with PLUMED is already available in your `$PATH`. 

The configuration of the environment will be further verified in the Jupyter Notebook tutorial.

Then one can start the tutorial by:

```shell
jupyter notebook
```

and open the `chabaDemo.ipynb`.

# Customized Utilization
The following installation requirements must be satisfied when one want to installed `skewencoder` and use the package immediately to run demos.
## Installation requirements
1. PLUMED version >= 2.9.0 and pytorch module linked with `libtorch` is required.
2. CP2K installation is required and is already patched with PLUMED.
3. Python version >= 3.10
4. pytorch version >= 2.3
5. lightning >= 2.3.3
6. For a better implementation create a separate python virtual env and install the autoencoder.
7. slurm job system is required.
## How to install
```shell
git clone https://github.com/zzkmirok/skewencoder.git
cd skewencoder
pip install [-e] .
```
## How to run
Set up your environment to run PLUMED with CP2K (or any MD driver compatible with PLUMED) in a `zsh` shell. Follow the example in `demos/BAS/chabaDemo.py` to establish your simulation workflow.
 
### Adjust your system to the Skewencoder workflow.

One can first create a folder and put the python script inside the folder.

```shell
mkdir -p [system_name]_simulation
cd [system_name]_simulation
touch [system_name]Demo.py
```

Basically the following functions need to be adjusted for a complete simulation.

- `main(kappa)`
- `get_plumed_[system_name]_unbiased`

    e.g. `get_plumed_chaba_unbiased`

- `[system_name]_training`

    e.g `chaba_training`

- `get_plumed_[system_name]_biased`

    e.g. `get_plumed_chaba_biased`

- `[system_name]_simulation`

    e.g. `chaba_simulation`

The simulation results (COLVAR files generated by PLUMED) along with the trained models will be stored at the `RESULTS_FOLDER = ./results`, and the initial unbiased simulation results will be stored at `UNBIASED_FOLDER = ./unbiased`. For CP2K simulations, trajectories will be generated every iteration and stored in the script folder `[system_name]_simulation`.

**Adjust functionalities step by step.**

1. `main(kappa)`

    The input is $\kappa$ referring to the stiffness parameter of the harmonic wall in the latent space, which is the only parameter that requires user initialization.

   This function is applied to configure all other related parameters and hyperparameters (they can remain as default values), intergrate with shell interface to run (un-)biased simulation over corresponding md drivers and train Skewencoder model based on the generated data.
  
-   (Hyper-)parameter initialization
    ```python
    n_max_iter = 40 # the maximal number of iterations of biased simulation
    loss_coeff = 0.1 # the skew loss coefficient for Skewencoder
    torch.manual_seed(22)
    batch_size = 100 # batch size for training
    offset = 1.0 # offset for setting up biased harmonic wall in latent space
    ...
    encoder_layers = [n_descriptors, 90, 40, 20, 5, 1] # The structure of encoder layers.
    ```
-  Generate PLUMED file for unbiased simulation and run unbiased simulation based on existing CP2K input file. The input descriptor set of the skew encoder should already be defined in this part.
    ```python
    gen_plumed_chaba_unbiased() # For further details in the specific section
    subprocess.run([*bash_prefix,"srun -n 12 cp2k.popt job.inp > output.log"], cwd=SCRIPT_DIR)
    ```

- Organize unbiased trajectory results for further analysis
  ```python
    subprocess.run([*bash_prefix, "mv Chaba-1.restart newiter.restart"], cwd = SCRIPT_DIR)
    subprocess.run([*bash_prefix, "rm -f Chaba*.restart"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"mv Chaba-pos-1.pdb {kappa}_iteration_Chaba_unbiased-pos.pdb"], cwd = SCRIPT_DIR)
    subprocess.run([*bash_prefix, f"cat {kappa}_iteration_Chaba_unbiased-pos.pdb > all_{kappa}.pdb"], cwd=SCRIPT_DIR)
    subprocess.run([*bash_prefix,"rm -f PLUMED.OUT Chaba*"], cwd=SCRIPT_DIR)
  ```

- Parse unbiased COLVAR file for storing the input discriptors information (mainly bond types and corresponding bond threshold distances) of the system. The information will be used later for state dection to determine whether to warm start training procedure.

    ```python
    bond_type_dict, n_descriptors,heavy_atom_pairs_list = STADECT.parse_unbiased_colvar(colvar_file = f"{UNBIASED_FOLDER}/COLVAR")
    state_detection = STADECT.State_detection((0.3, 0.7), bond_type_dict=bond_type_dict, n_heavy_atom_pairs=n_descriptors)
    # 0.3 is the state detection parameter q.
    ```

- In every iteration of the biased simulation
    The related model is trained, model name stored and the run a biased simulation.
    ```python
    state_detection, model, ITER_FOLDER, skewness_dataset, break_flag = chaba_training(state_detection, iter, encoder_layers, loss_coeff, batch_size)
    # For further details in the specific section
    ...
    model_name = f"{ITER_FOLDER}/model_autoencoder_{iter}.pt"
    chaba_simulation(ITER_FOLDER, model_name, model, skewness_dataset, kappa, offset)
    # For further details in the specific section
    ```

2. `get_plumed_[system_name]_unbiased`
   
    **Parameters:**  
   
    - `file_path = SCRIPT_DIR`

        The path where to store the generated PLUMED input file.
    
        `SCRIPT_DIR` by default.
     
    - `simulation_folder = UNBIASED_FOLDER` 
        
        The path where to store the COLVAR file generated from the unbiased simulation
    
   
   Create the PLUMED input file for unbiased simulation.

   The input descriptor is defined in this function. The name template for a specific distance variable is:
   ```shell
   [atom_type] [number] [atom_type] [number]
   ```
   For example
   ```shell
   o1o2: DISTANCE ...
   c1c2: DISTANCE ...
   c1o1: DISTANCE ...
   o2h1: DISTANCE ...
   ``` 
    The digits after each atom type can de specified by users. 
    
    Two definition criteria should be satisfied:

    - Two different atoms of the same type should have different number. 
    - Each involved atom should have a unique name.

    All related distance variables in the set of input descriptors should be displayed in the generated COLVAR file.
   ```shell
    PRINT FMT=%g STRIDE=10 FILE={simulation_folder}/COLVAR ARG=o4h7,o4h4,o4h5,o2h7,o2h4,o2h5,o3h7,o3h4,o3h5,o1h7,o1h4,o1h5,o4h3,o4h6,o2h3,o2h6,o3h3,o3h6,o1h3,o1h6,o4h2,o2h2,o3h2,o1h2,o4h1,o2h1,o3h1,o1h1,o5c1,o5c2,o5h7,o5h4,o5h5,o5h3,o5h6,o5h1,o5h2'''
   ```
1. `[system_name]_training` 

    **Parameters**

    - `state_detection: STADECT.State_detection`

        `State_detection` object for storing the state information of the system.

    - `iter: int`

        the current number of iteration.

    - `encoder_layers : Sequence[int]`

        the structure of encoder layers of Skewencoder

    - `loss_coeff: float`

        The loss coefficient of skew loss

    - `batch_size: int`

        batch size for training.

    Training procedure with warm-start are wrapped up in the function.

2. `[system_name]_simulation`

    **Parameters:**
    - `iter_folder`

        Folder name that stores the simulation results (COLVAR file) and the trained model

    - `model_name`

        The path of the trained model that is used for generating latent space for biased simulation.

    - `model` 

        The trained pytorch model object.

    - `dataset`

        Local dataset for skew loss training within the scope of one iteration.

    - `kappa`

        The stiffness parameter for biased wall

    - `offset`

        The offset parameter for biased wall.

    This function will generate the biased wall parameters for the biased simulation. Then invoke `get_plumed_[system_name]_biased`

3. `get_plumed_[system_name]_biased`

    **Parameters:**

      - model_name : str,

        Path to the trained model that is used for generating latent space.

      - file_path : str,

        Path to the generated PLUMED input file

      - simulation_folder

        Path to save the simulation results (COLVAR file)
      
      - pos, skew, kappa, offset

        Parameters generated from  `[system_name]_simulation` for the biased harmonic wall.

    Generate PLUMED input file for biased simulations.