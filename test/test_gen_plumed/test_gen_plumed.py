from skewencoder.gen_plumed import PlumedInput

from skewencoder.io import GeometryParser

from skewencoder.plumedkits import DISTANCE, WALL

import pathlib

data_path = (pathlib.Path(__file__).parent.resolve())

def test_PlumedInput_initialization():
    coord_file =(data_path/ "MeNC.dat") 
    mencparser = GeometryParser(coord_file=coord_file)
    plumed_input1 = PlumedInput(geo_parser=mencparser)
    assert plumed_input1.heavy_atom_pairs == [("C1C6", (1,6)), ("C1N5", (1,5)), ("C6N5",(6,5))] 
    assert plumed_input1.h_heavy_pairs == [('H2C1', (2, 1)), ('H3C1', (3, 1)), ('H4C1', (4, 1))]

def test_PlumedInput_gen_Distances_snippet():
    coord_file =(data_path/ "MeNC.dat") 
    mencparser = GeometryParser(coord_file=coord_file)
    plumed_input1 = PlumedInput(geo_parser=mencparser)
    heavy_atom_pairs = plumed_input1.heavy_atom_pairs
    assert plumed_input1.gen_plumed_DISTANCE_snippet(heavy_atom_pairs) == """C1C6: DISTANCE ATOMS=1,6
C1N5: DISTANCE ATOMS=1,5
C6N5: DISTANCE ATOMS=6,5"""

def test_PlumedInput_build():
    coord_file =(data_path/ "MeNC.dat") 
    mencparser = GeometryParser(coord_file=coord_file)
    Options={"distance_options": ["NOPBC"],"pytorch_model_heavy": "./dumbmodel1", "pytorch_model_h": "./dumbmodel2"}
    plumed_input1 = PlumedInput(geo_parser=mencparser, if_biased=False, **Options)
    plumed_input2 = PlumedInput(geo_parser=mencparser, if_biased=True, heavy_atom_only= False ,distance_options= ["NOPBC"], pytorch_model_heavy= "./dumbmodel1", pytorch_model_h= "./dumbmodel2", skew_wall_heavy = {"is_lower_wall": True, "kappa": 100.0, "pos": 0.0, "offset": 0.1}, skew_wall_h_adj = {"is_lower_wall": False, "kappa": 200.0, "pos": 0.1, "offset": 0.2})
    plumed_input2.add_additional_command(DISTANCE(label="c1", atoms=(2,3), Options=["NOPBC"]))
    plumed_input2.add_additional_command(WALL(label="wall1", ARG="c1", AT=0.0, KAPPA=100.0))
    assert plumed_input1.build() ==f'''# vim:ft=plumed
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs


C1C6: DISTANCE ATOMS=1,6 NOPBC
C1N5: DISTANCE ATOMS=1,5 NOPBC
C6N5: DISTANCE ATOMS=6,5 NOPBC


H2C1: DISTANCE ATOMS=2,1 NOPBC
H3C1: DISTANCE ATOMS=3,1 NOPBC
H4C1: DISTANCE ATOMS=4,1 NOPBC


constr_C1C6: UPPER_WALLS ARG=C1C6 AT=6.0 KAPPA=200 EXP=2.0 EPS=1.0 OFFSET=0.0
constr_C1N5: UPPER_WALLS ARG=C1N5 AT=6.0 KAPPA=200 EXP=2.0 EPS=1.0 OFFSET=0.0
constr_C6N5: UPPER_WALLS ARG=C6N5 AT=6.0 KAPPA=200 EXP=2.0 EPS=1.0 OFFSET=0.0


PRINT FMT=%g STRIDE=10 FILE=./COLVAR ARG=C1C6,C1N5,C6N5,H2C1,H3C1,H4C1'''
   
    assert plumed_input2.build() == f'''# vim:ft=plumed
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs


C1C6: DISTANCE ATOMS=1,6 NOPBC
C1N5: DISTANCE ATOMS=1,5 NOPBC
C6N5: DISTANCE ATOMS=6,5 NOPBC


H2C1: DISTANCE ATOMS=2,1 NOPBC
H3C1: DISTANCE ATOMS=3,1 NOPBC
H4C1: DISTANCE ATOMS=4,1 NOPBC


c1: DISTANCE ATOMS=2,3 NOPBC


constr_C1C6: UPPER_WALLS ARG=C1C6 AT=6.0 KAPPA=200 EXP=2.0 EPS=1.0 OFFSET=0.0
constr_C1N5: UPPER_WALLS ARG=C1N5 AT=6.0 KAPPA=200 EXP=2.0 EPS=1.0 OFFSET=0.0
constr_C6N5: UPPER_WALLS ARG=C6N5 AT=6.0 KAPPA=200 EXP=2.0 EPS=1.0 OFFSET=0.0


cv_heavy: PYTORCH_MODEL FILE=./dumbmodel1 ARG=C1C6,C1N5,C6N5
cv_h_adj: PYTORCH_MODEL FILE=./dumbmodel2 ARG=H2C1,H3C1,H4C1


wall1: LOWER_WALLS ARG=c1 AT=0.0 KAPPA=100.0 EXP=2.0 EPS=1.0 OFFSET=0.0
cv_wall_heavy: LOWER_WALLS ARG=cv_heavy.node-0 AT=0.1 KAPPA=100.0 EXP=2.0 EPS=1.0 OFFSET=0.0
cv_wall_h_adj: UPPER_WALLS ARG=cv_h_adj.node-0 AT=-0.1 KAPPA=200.0 EXP=2.0 EPS=1.0 OFFSET=0.0


PRINT FMT=%g STRIDE=10 FILE=./COLVAR ARG=C1C6,C1N5,C6N5,H2C1,H3C1,H4C1,c1,cv_heavy.*,cv_h_adj.*'''

