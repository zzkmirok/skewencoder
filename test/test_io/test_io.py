from skewencoder.io import GeometryParser

import pathlib

data_path = (pathlib.Path(__file__).parent.resolve())

def test_GeometryPaser_dat_simple():
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

def test_GeometryPaser_dat_hard():
    coord_file = (data_path/ "coord.dat")
    coordparser = GeometryParser(coord_file=coord_file)
    assert coordparser.connected_components == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}, {11, 12, 13, 14}]
    assert coordparser.vdw_pairs == [(18.054884635877816, (4, 8)), (14.845077225972176, (4, 13)), (4.174504733657137, (10, 14))]
    assert coordparser.vdw_pairs_heavy_only == [(19.452534888132565, (2, 5)), (15.680222793817299, (2, 12)), (5.224895494827247, (10, 11))]

def test_GeometryPaser_xyz():
    coord_file =(data_path/ "test.xyz") 
    xyzparser = GeometryParser(coord_file=coord_file)
    assert xyzparser.atom_list == {'C': [0, 5, 10, 11], 
                                   'O': [1, 2, 12], 
                                   'H': [3, 4, 6, 7, 8, 13, 14], 
                                   'N': [9]}
    assert xyzparser.adjacent_list == {0: [1, 2, 3], 
                                       1: [0], 
                                       2: [0, 4], 
                                       3: [0], 
                                       4: [2], 
                                       5: [6, 7, 8, 9], 
                                       6: [5], 
                                       7: [5], 
                                       8: [5], 
                                       9: [5, 10], 
                                       10: [9, 11, 12], 
                                       11: [10, 12, 13, 14], 
                                       12: [10, 11], 
                                       13: [11], 
                                       14: [11]}
    
    assert xyzparser.connected_components == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}]
    assert xyzparser.vdw_pairs == [(13.238691312826008, (3, 14))]
    assert xyzparser.vdw_pairs_heavy_only == [(14.682579538313401, (0, 5))]


