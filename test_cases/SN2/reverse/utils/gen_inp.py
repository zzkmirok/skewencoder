import numpy as np
from scipy import stats
import sys
import plumed
from scipy.special import softmax

def postprocessing1(p_kappa, p_alpha, *args):
    error = '--- ERROR: %s \n'
    # p_kappa = 50.0
    # p_alpha = 1.5

    data = plumed.read_as_pandas(f"./data/COLVAR/COLVAR_{iteration}")
    data_rotated = np.column_stack((data["p.x"], data["p.y"]))

    # parameter for fitting
    n = np.shape(data_rotated)[0]
    d = np.shape(data_rotated)[1]

    mu = np.average(data_rotated, axis=0)
    array_skew = np.zeros(d)
    for i in range(d):
        array_skew[i] = stats.skew(data_rotated[:,i])

    kappa = array_skew * p_kappa

    pos = mu + array_skew / np.abs(array_skew) * softmax(array_skew) * p_alpha

    info = np.concatenate((mu, array_skew, pos, kappa))


    with open(File, "a") as txt_file:
        np.savetxt(txt_file, info, fmt='%.18f', newline=' ', delimiter=',')

    # Recording how many walls added
    wallstr = ""

    # TODO: First consider all walls are activated, later consider adding deactivated function.
    # So far we haven't considered the effects of covariances and correlations.
    for i, key in enumerate(list(data.keys())[1:d+1]):
        if kappa[i] < 0:
            with open("./plumed.dat","a") as f:
                print(f"""
# Energy wall for {key}
uwall{i}: UPPER_WALLS ARG={key} AT={pos[i]} KAPPA={-kappa[i]} ExP=2 EPS=1 OFFSET=0.0
""",file=f)
            wallstr = wallstr + f"uwall{i}.bias"
        else:
            with open("./plumed.dat","a") as f:
                print(f"""
# Energy wall for {key}
lwall{i}: LOWER_WALLS ARG={key} AT={pos[i]} KAPPA={kappa[i]} ExP=2 EPS=1 OFFSET=0.0
""",file=f)
            wallstr = wallstr + f"lwall{i}.bias"
        if i < (d-1):
            wallstr = wallstr + ","

    with open("./plumed.dat","a") as f:
        print(f"""
PRINT ARG=p.x,p.y,{wallstr} FILE=./data/COLVAR/COLVAR STRIDE={STRIDE}
""",file=f)
    return mu



def postprocessing2(p_kappa, p_alpha, *args):
    error = '--- ERROR: %s \n'
    # p_kappa = 50.0
    # p_alpha = 1.5

    data = plumed.read_as_pandas(f"./data/COLVAR/COLVAR_{iteration}")

    data_rotated, new_state = point_filtering(dim, data)

    # parameter for fitting
    n = np.shape(data_rotated)[0]
    d = np.shape(data_rotated)[1]

    mu = np.average(data_rotated, axis=0)
    array_skew = np.zeros(d)
    for i in range(d):
        array_skew[i] = stats.skew(data_rotated[:,i])

    kappa = array_skew * p_kappa

    pos = mu + array_skew / np.abs(array_skew) * softmax(array_skew) * p_alpha

    info = np.concatenate((mu, array_skew, pos, kappa, [new_state]))

    with open(File, "a") as txt_file:
        np.savetxt(txt_file, info, fmt='%.18f', newline=' ', delimiter=',')

    # TODO: First consider all walls are activated, later consider adding deactivated function.
    # So far we haven't considered the effects of covariances and correlations.
    wallstr = ""
    for i, key in enumerate(list(data.keys())[1:d+1]):
        if kappa[i] < 0:
            with open("./plumed.dat","a") as f:
                print(f"""
# Energy wall for {key}
uwall{i}: UPPER_WALLS ARG={key} AT={pos[i]} KAPPA={-kappa[i]} ExP=2 EPS=1 OFFSET=0.0
""",file=f)
            wallstr = wallstr + f"uwall{i}.bias"
        else:
            with open("./plumed.dat","a") as f:
                print(f"""
# Energy wall for {key}
lwall{i}: LOWER_WALLS ARG={key} AT={pos[i]} KAPPA={kappa[i]} ExP=2 EPS=1 OFFSET=0.0
""",file=f)
            wallstr = wallstr + f"lwall{i}.bias"
        if i < (d-1):
            wallstr = wallstr + ","

    with open("./plumed.dat","a") as f:
        print(f"""
PRINT ARG=p.x,p.y,{wallstr} FILE=./data/COLVAR/COLVAR STRIDE={STRIDE}
""",file=f)
    return mu


### For filtering out points that are not affected by the walls
def point_filtering(dim, data):
    index = 0
    flag = False

    # select the points until the first one with no wall affects
    for i,_ in enumerate(data['time']):
        for key in list(data.keys())[dim+1:dim+dim+1]:
            if data[key][i] == 0.0:
                flag = True
            else:
                flag = False
                break
        if flag:
            index = i
            break
    indices_to_delete = [i for i in range(index)]

    # justify if this is a new state
    flag = False
    new_state = False
    length = len(data['time'])
    for i in range(length - 1, length-101, -1):
        for key in list(data.keys())[dim+1:dim+dim+1]:
            if data[key][i] == 0.0:
                new_state = True
            else:
                new_state = False
                break
        if not new_state:
            break
    # # if this is a new state, select only those points in the vicinity of the new state
    if new_state:
        for i in range(length-101,-1,-1):
            for key in list(data.keys())[dim+1:dim+dim+1]:
                if data[key][i] != 0.0:
                    flag = True
                    break
            if flag:
                index = i
                break
        indices_to_delete = [i for i in range(index)]

    for key in data.keys():
        for i in sorted(indices_to_delete, reverse=True):
            del data[key][i]


    # create a list of keys referring to the positions (e.g. p.x, p.y)
    poskeys = list(data.keys())[1:dim+1]
    data_rotated = np.column_stack([data[keys] for keys in poskeys])

    return data_rotated, new_state

def gen_plumed(model_name : str,
                file_path : str,
                potential_formula : str,
                ):

    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')

    input=f'''# vim:ft=plumed
UNITS NATURAL

p: POSITION ATOM=1

# define modified Muller Brown potential
ene: CUSTOM ARG=p.x,p.y PERIODIC=NO ...
FUNC={potential_formula}
...

pot: BIASVALUE ARG=ene

# load deep cv pytorch model
cv: PYTORCH_MODEL FILE=../{model_name} ARG=p.x,p.y

# apply bias
opes: {opes_mode} ARG=cv.node-0 PACE=500 BARRIER=16 STATE_WSTRIDE=10000 STATE_WFILE=State.data

PRINT FMT=%g STRIDE=100 FILE=COLVAR ARG=p.x,p.y,cv.*,opes.*

ENDPLUMED

    '''
    print(input, file=file)
    file.close()


def gen_input_md(inital_position : str,
                file_path : str,
                nsteps : int ):
    file_path = f'{file_path}/input_md.dat'
    file = open(file_path, 'w')
    input=f'''nstep             {nsteps}
tstep             0.005
temperature       1.0
friction          10.0
random_seed       42
plumed_input      plumed.dat
dimension         2
replicas          1
basis_functions_1 BF_POWERS ORDER=1 MINIMUM=-4.0 MAXIMUM=+3.0
basis_functions_2 BF_POWERS ORDER=1 MINIMUM=-1.0 MAXIMUM=+2.5
input_coeffs       input_md-potential.dat
initial_position   {inital_position}
output_coeffs           /dev/null
output_potential        /dev/null
output_potential_grid   10
output_histogram        /dev/null'''
    print(input, file=file)
    file.close()

