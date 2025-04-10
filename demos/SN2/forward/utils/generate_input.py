def gen_plumed(model_name : str,
                file_path : str,
                potential_formula : str,
                opes_mode : str = 'OPES_METAD'):

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

def gen_plumed_aes(model_name : str,
                file_path : str,
                simulation_folder,
                pos,
                skew,
                kappa,
                offset):

    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')

# TODO: figure out why ves and why only one ATOM

    input=f'''# vim:ft=plumed
MOLINFO STRUCTURE=./input.ala2.pdb
phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2

# Default Energy wall for testing
enewall: UPPER_WALLS ARG=psi AT=1.5 KAPPA=50 ExP=2 EPS=1 OFFSET=0.0
enewall1: LOWER_WALLS ARG=phi AT=-1.5 KAPPA=50 ExP=2 EPS=1 OFFSET=0.0

cv: PYTORCH_MODEL FILE=../{model_name} ARG=phi,psi

# use this command to write phi and psi on a file named colvar.dat, every 100 steps
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
PRINT FMT=%g STRIDE=100 FILE=../{simulation_folder}/COLVAR ARG=phi,psi,cv.*
""",file=f)

def gen_plumed_aes_2(model_name : str,
                file_path : str,
                simulation_folder,
                pos,
                skew,
                kappa,
                offset):

    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')

# TODO: figure out why ves and why only one ATOM

    input=f'''# vim:ft=plumed
MOLINFO STRUCTURE=./input.ala2.pdb
phi: TORSION ATOMS=@phi-2
psi: TORSION ATOMS=@psi-2

d1: DISTANCE ATOMS=2,5 COMPONENTS NOPBC # Distance between CH3 and C
d2: DISTANCE ATOMS=2,7 COMPONENTS NOPBC # Distance between CH3 and N
d3: DISTANCE ATOMS=2,9 COMPONENTS NOPBC # Distance between CH3 and CA
d4: DISTANCE ATOMS=2,11 COMPONENTS NOPBC  # Distance between CH3 and CB
d5: DISTANCE ATOMS=5,7 COMPONENTS NOPBC  # Distance between C and N
d6: DISTANCE ATOMS=5,9 COMPONENTS NOPBC  # Distance between C and CA
d7: DISTANCE ATOMS=5,11 COMPONENTS NOPBC  # Distance between C and CB
d8: DISTANCE ATOMS=7,9 COMPONENTS NOPBC  # Distance between N and CA
d9: DISTANCE ATOMS=7,11 COMPONENTS NOPBC  # Distance between N and CB
d_10: DISTANCE ATOMS=9,11 COMPONENTS NOPBC  # Distance between CA and CB
cv: PYTORCH_MODEL FILE=../{model_name} ARG=d1.x,d1.y,d1.z,d2.x,d2.y,d2.z,d3.x,d3.y,d3.z,d4.x,d4.y,d4.z,d5.x,d5.y,d5.z,d6.x,d6.y,d6.z,d7.x,d7.y,d7.z,d8.x,d8.y,d8.z,d9.x,d9.y,d9.z,d_10.x,d_10.y,d_10.z
# use this command to write phi and psi on a file named colvar.dat, every 100 steps
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
PRINT FMT=%g STRIDE=100 FILE=../{simulation_folder}/COLVAR ARG=phi,psi,d1.x,d1.y,d1.z,d2.x,d2.y,d2.z,d3.x,d3.y,d3.z,d4.x,d4.y,d4.z,d5.x,d5.y,d5.z,d6.x,d6.y,d6.z,d7.x,d7.y,d7.z,d8.x,d8.y,d8.z,d9.x,d9.y,d9.z,d_10.x,d_10.y,d_10.z,cv.*
""",file=f)



def gen_plumed_aes_3(model_name : str,
                file_path : str,
                simulation_folder,
                pos,
                skew,
                kappa,
                offset):

    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')

# TODO: figure out why ves and why only one ATOM

    input=f'''# vim:ft=plumed
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs

ene: ENERGY
d1: DISTANCE ATOMS=1,2 NOPBC 
d5: DISTANCE ATOMS=1,6 NOPBC

c1:  COMBINE ARG=d1,d5 COEFFICIENTS=0.500,-0.500 PERIODIC=NO

h1: DISTANCE ATOMS=2,3 NOPBC
h2: DISTANCE ATOMS=2,4 NOPBC
h3: DISTANCE ATOMS=2,5 NOPBC

UPPER_WALLS ARG=d1 AT=+4.0 KAPPA=150.0 EXP=2 LABEL=uwall_1
UPPER_WALLS ARG=d5 AT=+4.0 KAPPA=150.0 EXP=2 LABEL=uwall_2
LOWER_WALLS ARG=h1 AT=+1.7 KAPPA=150.0 EXP=2 LABEL=lwall_1
LOWER_WALLS ARG=h2 AT=+1.7 KAPPA=150.0 EXP=2 LABEL=lwall_2
LOWER_WALLS ARG=h3 AT=+1.7 KAPPA=150.0 EXP=2 LABEL=lwall_3
cv: PYTORCH_MODEL FILE=./{model_name} ARG=d1,d5
# use this command to write phi and psi on a file named colvar.dat, every 100 steps
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

f1: CONTACTMAP ...
    ATOMS1=1,2 SWITCH1={{RATIONAL R_0=1.8}} 
    ATOMS2=1,6 SWITCH2={{RATIONAL R_0=2.5}}
...   
PRINT FMT=%g STRIDE=10 FILE=./{simulation_folder}/COLVAR ARG=d1,d5,c1,cv.*,f1.*""",file=f)

def gen_plumed_restart(file_path : str,
        potential_formula : str):

    file_path = f'{file_path}/plumed.dat'
    file = open(file_path, 'w')
    input=f'''# vim:ft=plumed
UNITS NATURAL

# p: POSITION ATOM=1
p: DISTANCE ATOMS=1,2 COMPONENTS

# define model potential
ene: CUSTOM ARG=p.x,p.y PERIODIC=NO ...
FUNC={potential_formula}
...

pot: BIASVALUE ARG=ene

PRINT FMT=%g STRIDE=10 FILE=COLVAR ARG=p.x,p.y
'''
    print(input, file=file)
    file.close()


def gen_plumed_tica(model_name : str,
                    file_path : str,
                    rfile_path : str,
                    potential_formula : str,
                    static_bias_cv : str = None,
                    static_model_path : str = None,
                    opes_mode : str = 'OPES_METAD',
                    opes_args : str = 'tica.node-0'):

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
tica: PYTORCH_MODEL FILE=../{model_name} ARG=p.x,p.y
'''
    print(input, file=file)
    if static_model_path is not None:

        input=f'cv: PYTORCH_MODEL FILE={static_model_path} ARG=p.x,p.y'
        print(input, file=file)

    if static_bias_cv is not None:
        input=f'''# apply static bias from previous sim
static: OPES_METAD ARG={static_bias_cv} ...
    RESTART=YES
    STATE_RFILE={rfile_path}
    BARRIER=16
    PACE=10000000
...
'''
        print(input, file=file)

    input=f'''# apply bias
opes: {opes_mode} ARG={opes_args} PACE=500 BARRIER=16
'''
    print(input, file=file)

    if static_bias_cv is not None:
        input=f'''PRINT FMT=%g STRIDE=100 FILE=COLVAR ARG=p.x,p.y,tica.*,opes.*,{static_bias_cv},static.*

ENDPLUMED'''
    else:
        input=f'''PRINT FMT=%g STRIDE=100 FILE=COLVAR ARG=p.x,p.y,tica.*,opes.*

ENDPLUMED'''

    print(input, file=file)
    file.close()


def gen_input_md(initial_position : str,
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
initial_position   {initial_position}
output_coeffs           /dev/null
output_potential        /dev/null
output_potential_grid   10
output_histogram        /dev/null'''
    print(input, file=file)
    file.close()


def gen_input_md_aes(initial_position : str,
                file_path : str,
                nsteps : int ):
    file_path = f'{file_path}/input_md.dat'
    file = open(file_path, 'w')
    # TODO: t step could be 0.005?
    # TODO: temperature 0.5 or 1.0?
    input=f'''nstep             {nsteps}
tstep         0.05
temperature   0.5
friction      10.0
periodic      false
dimension     2
ipos          {initial_position}
plumed        plumed.dat'''
    print(input, file=file)
    file.close()


def gen_input_md_restart(initial_position : str,
                file_path : str,
                nsteps : int ):
    file_path = f'{file_path}/input_md.dat'
    file = open(file_path, 'w')
    # TODO: t step could be 0.005?
    # TODO: temperature 0.5 or 1.0?
    input=f'''nstep             {nsteps}
tstep         0.05
temperature   0.5
friction      10.0
periodic      false
dimension     2
ipos          {initial_position}
plumed        plumed.dat'''
    print(input, file=file)
    file.close()


def gen_input_md_potential(file_path : str):
    file_path = f'{file_path}/input_md-potential.dat'
    file = open(file_path, 'w')
    input=f'''#! FIELDS idx_dim1 idx_dim2 pot.coeffs index description
#! SET type LinearBasisSet
#! SET ndimensions  2
#! SET ncoeffs_total  1
#! SET shape_dim1  2
#! SET shape_dim2  2
    0       0         1.0000000000000000e+00       0  1*1
#!-------------------'''
    print(input, file=file)
    file.close()
