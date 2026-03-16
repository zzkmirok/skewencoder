# Step by step configuration for Running loxodynamics with lammps MLIP (Ru105-NH3 as example)
1. Prepare the script folder like this:

```shell
mkdir Ru001-N2-3H2-CN # This name for the parent folder will be used later.
cd  Ru001-N2-3H2-CN    
tree .
```
It will show
```shell
.
├── config_all.zsh
├── run_all.zsh
└── template-N2-NH3-cn
    ├── config.toml
    ├── RuNH3Demo.py
    └── runsbatch.sh

```

2. Outside this folder, create `simulation`, `models` and `systems`.

In `simulation` are input templates for lammps-mace (in folder `lammps`) and lammps-metatomic (in folder `lammps-metatomic`). All variables in the pair <> will be set by the `config.toml`. Remember: atom types in metatomic is integers but input as str, i.e. `["44", "7", "1"]`

In `models` are the MLIP models. For MACE potential, remember to convert them to Lammps supporting models from the raw pretrained models.

In `systems` are the geometry files for `read_data` in lammps.

3. Now take the `config.toml` in `template-2NH3-cn` as an example to explain what we should set in the `config.toml`

```
# for locating the templates, so far "lammps" and "lammps-metatomic". Keep the future scalability to other driver like cp2k.
driver = "lammps"

[simulation]
mlip = "mace" # For choosing which lammps exec we used to run the simulation. A `dict EXEC_DICT` in the `RuNH3Demo.py` stores the actual path to your lammps exec. 
model = "mace-mh-1-oc20_usemppbe.model-lammps.pt" # The MLP model name.
system = "SURFACE_2NH3_relax_mh_1" # The geometry data name, without subfix .data
systems_folder = "../../systems" # path to the folder storing geometry data
model_folder = "../../models" # path to the folder storing the models.
n_gpus = 1 # number of gpus for set the correponding param when running lammps kokos
atom_list = ["Ru", "N", "H"] # For MLP model settings in the `pair_coeff`
temperature = 500.0
restart_file = "tmp.restart" # tmp file for restarting in every iteration.
frozen_layers = "1:12 25:36 49:60" # Frozen layers for the system.
config_folder = "../../simulation" # Path for the folders storing lammps input templates.
seed = 42 # seed for lammps, no need to change.
# Remarks: in the python script `RuNH3Demo.py`, `REPLACE_DICT` stores all the keys and values that have to be changed in the template. If new keys are added, or old keys are deleted, one should also add or remove the keys correspondingly both here and in the dict `REPLACE_DICT`.

[loxodynamics]
kappa = 300 # For setting up kappa
max_iter = 1 
batch_size = 100 # Do not change
warm_start = false # Just leave for future interface extension, not deactivated
state_detection = false # Just leave future interface extension, not deactivated

[loxodynamics.skewencoder]
hidden_layers = [50, 30, 20, 10, 5] # Only Hidden layers
threshold = 0.3 # p value in the switching function

[plumed]
log_file = "plumed.log" # Do not have to change
stride = 10 # Don't change for stride printing.
# environment CVs and biases (for contrain the system)
env_bias = """
com_Ru: CENTER ATOMS=13-24,37-48,61-72
dist73: DISTANCE ATOMS=73,com_Ru COMPONENTS
dist77: DISTANCE ATOMS=77,com_Ru COMPONENTS

dist74: DISTANCE ATOMS=74,com_Ru COMPONENTS
dist75: DISTANCE ATOMS=75,com_Ru COMPONENTS
dist76: DISTANCE ATOMS=76,com_Ru COMPONENTS
dist78: DISTANCE ATOMS=78,com_Ru COMPONENTS
dist79: DISTANCE ATOMS=79,com_Ru COMPONENTS
dist80: DISTANCE ATOMS=80,com_Ru COMPONENTS

abs_d73_z: MATHEVAL ARG=dist73.z FUNC=abs(x) PERIODIC=NO
abs_d77_z: MATHEVAL ARG=dist77.z FUNC=abs(x) PERIODIC=NO

abs_d74_z: MATHEVAL ARG=dist74.z FUNC=abs(x) PERIODIC=NO
abs_d75_z: MATHEVAL ARG=dist75.z FUNC=abs(x) PERIODIC=NO
abs_d76_z: MATHEVAL ARG=dist76.z FUNC=abs(x) PERIODIC=NO
abs_d78_z: MATHEVAL ARG=dist78.z FUNC=abs(x) PERIODIC=NO
abs_d79_z: MATHEVAL ARG=dist79.z FUNC=abs(x) PERIODIC=NO
abs_d80_z: MATHEVAL ARG=dist80.z FUNC=abs(x) PERIODIC=NO

uwall73: UPPER_WALLS ARG=abs_d73_z AT=20.0 KAPPA=150.0 EXP=2
uwall77: UPPER_WALLS ARG=abs_d77_z AT=20.0 KAPPA=150.0 EXP=2

uwall74: UPPER_WALLS ARG=abs_d74_z AT=20.0 KAPPA=150.0 EXP=2
uwall75: UPPER_WALLS ARG=abs_d75_z AT=20.0 KAPPA=150.0 EXP=2
uwall76: UPPER_WALLS ARG=abs_d76_z AT=20.0 KAPPA=150.0 EXP=2
uwall78: UPPER_WALLS ARG=abs_d78_z AT=20.0 KAPPA=150.0 EXP=2
uwall79: UPPER_WALLS ARG=abs_d79_z AT=20.0 KAPPA=150.0 EXP=2
uwall80: UPPER_WALLS ARG=abs_d80_z AT=20.0 KAPPA=150.0 EXP=2
"""
# The realtime values of the above biases will be printed for checking if they play some roles in the simulation.
env_bias_cv = [
    "uwall73.bias",
    "uwall77.bias",
    "uwall74.bias",
    "uwall75.bias",
    "uwall76.bias",
    "uwall78.bias",
    "uwall79.bias",
    "uwall80.bias",
]

cv_type = "DISTANCE" # Do not change, in principle for state detection only DISTANCE is supported
pbc = true # For creating PBC or Non PBC DISTANCES
# Names of DISTANCES CVs, must follow the naming rules, <ATOM1_TYPE><ATOM1_ID><ATOM2_TYPE><ATOM2_ID>
descriptors = [
    "H74N73",
    "H75N73",
    "H76N73",
    "H78N73",
    "H79N73",
    "H80N73",
    "H74N77",
    "H75N77",
    "H76N77",
    "H78N77",
    "H79N77",
    "H80N77",
    "N73N77",
    "H74H75",
    "H74H76",
    "H75H76",
    "H74H78",
    "H74H79",
    "H74H80",
    "H75H78",
    "H75H79",
    "H75H80",
    "H76H78",
    "H76H79",
    "H76H80",
]
# Define customized CVs, those CVs are just for loxodynamcis biasing but not for state detections.
custom_cv_definition = """
WHOLEMOLECULES ENTITY0=13-24 ENTITY1=37-48 ENTITY2=61-72
Layer1: CENTER ATOMS=13-24
Layer2: CENTER ATOMS=37-48
Layer3: CENTER ATOMS=61-72
group_N: GROUP ATOMS=73,77
group_H: GROUP ATOMS=74-76,78-80
group_Ru_1: GROUP ATOMS=24,48,72
group_Ru_2: GROUP ATOMS=22,46,70
group_Ru_3: GROUP ATOMS=20,44,69
group_Ru_4: GROUP ATOMS=23,71,47
group_Ru_5: GROUP ATOMS=21,45,69
group_Ru_6: GROUP ATOMS=19,43,67

cn_N_N:    COORDINATION GROUPA=group_N R_0=1.45
cn_H_H:    COORDINATION GROUPA=group_H R_0=0.75
cn_N_H:    COORDINATION GROUPA=group_N GROUPB=group_H R_0=1.02
cn_N_Ru_1: COORDINATION GROUPA=group_N GROUPB=group_Ru_1 R_0=2.2
cn_N_Ru_2: COORDINATION GROUPA=group_N GROUPB=group_Ru_2 R_0=2.2
cn_N_Ru_3: COORDINATION GROUPA=group_N GROUPB=group_Ru_3 R_0=2.2
cn_N_Ru_4: COORDINATION GROUPA=group_N GROUPB=group_Ru_4 R_0=2.2
cn_N_Ru_5: COORDINATION GROUPA=group_N GROUPB=group_Ru_5 R_0=2.2
cn_N_Ru_6: COORDINATION GROUPA=group_N GROUPB=group_Ru_6 R_0=2.2
cn_H_Ru_1: COORDINATION GROUPA=group_H GROUPB=group_Ru_1 R_0=1.9
cn_H_Ru_2: COORDINATION GROUPA=group_H GROUPB=group_Ru_2 R_0=1.9
cn_H_Ru_3: COORDINATION GROUPA=group_H GROUPB=group_Ru_3 R_0=1.9
cn_H_Ru_4: COORDINATION GROUPA=group_H GROUPB=group_Ru_4 R_0=1.9
cn_H_Ru_5: COORDINATION GROUPA=group_H GROUPB=group_Ru_5 R_0=1.9
cn_H_Ru_6: COORDINATION GROUPA=group_H GROUPB=group_Ru_6 R_0=1.9
"""
# Record the above customized CV for PRINT, so that the data loader can load those CVs
# Remark: the names should have a common regex pattern so that the dataloader can load smoothly.
custom_descriptors = [
"cn_N_N",  
"cn_H_H",  
"cn_N_H",  
"cn_N_Ru_1",
"cn_N_Ru_2",
"cn_N_Ru_3",
"cn_N_Ru_4",
"cn_N_Ru_5",
"cn_N_Ru_6",
"cn_H_Ru_1",
"cn_H_Ru_2",
"cn_H_Ru_3",
"cn_H_Ru_4",
"cn_H_Ru_5",
"cn_H_Ru_6",
]

input_regex = '^cn_.*$' # For dataloader read the input cvs for skewencoder
detector_regex = '^([A-Za-z]+)\d+([A-Za-z]+)\d+$' # for state_detection
```

4. `RuNH3Demo.py` can parse the following args.
```
python RuNH3Demo.py [-c config.toml] [-test] <-custom | -normal | -all>
```

In `RuNH3Demo.py` it shows:
```python
parser.add_argument(
    "-c", "--config", type=str, default="config.toml", help="Path to TOML config"
)
parser.add_argument("-test", action="store_true", help="activate test mode")
cv_group = parser.add_mutually_exclusive_group()
cv_group.add_argument("-custom", action="store_true", help="use customized cv input")
cv_group.add_argument("-all", action="store_true", help="use both cv inputs")
cv_group.add_argument("-normal", action="store_true", help="use normal cv inputs")
# -all -normal -custom are mutually exclusive, and -normal by default
```

5. In `RuNH3Demo.py`, one may have to change the path in `EXEC_DICT` and the necessary key-value pairs in `REPLACE_DICT`
```python

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
```
6. In `runsbatch.sh`, check the `--account`. The `LMP_EXE` must not be necessary to set.
7. First test if it can ran properly in the template folder. Either by directly submit or using the following command in `login23-g-1` node. 
```shell
python RuNH3Demo.py -test -custom & 
```
8. How to use `config_all.zsh`
```shell
./config_all.zsh -s ./template-2NH3-cn-petmad1.5 -t 2NH3_CN_PETMAD1.5 -k "200 5 400"   
```
   - `-s` means source template folder is `./template-2NH3-cn-petmad1.5`.
   - `-t` means all simulation replica folders will be named begining with the prefix `2NH3_CN_PETMAD1.5`
   - `-k` means kappa is ranging from 200 to 400 and the step is 5, the same as `$(seq 200 5 400)`
   - `-m` means max number of iterations for loxodynamics, here by default 20.
  
  **ATTENTION**:

  This command will create for each kappa 5 replicas with different random seeds for lammps simulation, then with the command above you will have 200 replicas. If you do not want to have so many replicas, change the range of kappa or change the number of the random replicas in this line, instead of 5:
```shell
# line 141
SEEDS=$(shuf -i 100000-999999 -n 5)
```

9.  How to use `run_all.zsh`

```shell
./run_all.zsh -t 2NH3_CN_PETMAD1.5
```
    The prefix must correspond to the prefix above in `config_all.zsh` command.

    The command will submit all jobs created in the previous step.