from skewencoder.io import GeometryParser

import pathlib

data_path = (pathlib.Path(__file__).parent.resolve())

def test_GeometryPaser():
    coord_file =(data_path/ "MeNC.dat") 
    mencparser = GeometryParser(coord_file=coord_file)
    assert mencparser.atom_list == {"C": [0, 5],
                                    "H": [1, 2, 3],
                                    "N": [4]}
    assert mencparser.adjacent_list == {0: [1, 2, 3, 4],
                                        1: [0],
                                        2: [0],
                                        3: [0],
                                        4: [0,5],
                                        5: [4]}
    
