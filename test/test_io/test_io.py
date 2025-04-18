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


def test_GeometryParser_xyz_dissociated():
    coord_file = (data_path / "test2.xyz")
    xyzparser = GeometryParser(coord_file=coord_file)
    assert xyzparser.atom_list == {'C': [0, 5, 10, 11], 
                               'O': [1, 2, 12], 
                               'H': [3, 4, 6, 7, 8, 13, 14], 
                               'N': [9]}
    assert xyzparser.adjacent_list == {0: [1], 
                                       1: [0], 
                                       2: [], 
                                       3: [], 
                                       4: [], 
                                       5: [9], 
                                       6: [], 
                                       7: [], 
                                       8: [], 
                                       9: [5], 
                                       10: [], 
                                       11: [12], 
                                       12: [11], 
                                       13: [], 
                                       14: []}

    assert xyzparser.connected_components == [{0, 1}, {2}, {3}, {4}, {9, 5}, {6}, {7}, {8}, {10}, {11, 12}, {13}, {14}]
    assert xyzparser.vdw_pairs == [(2.6992517149714557, (0, 2)), (12.017866990184016, (1, 5)), (22.374952502525975, (1, 10)), (14.485893016588888, (1, 11)), (12.609572203755663, (2, 5)), (22.246411219186342, (2, 10)), (14.28329119584782, (2, 11)), (11.911531969904996, (5, 10)), (4.17349782405302, (5, 11)), (8.617720255861196, (10, 12)), (2.8884964609210164, (3, 2)), (11.909939776267674, (4, 0)), (2.282871737370215, (6, 10)), (5.320284094605459, (7, 10)), (4.30813305799527, (8, 10)), (2.0460097248944904, (13, 11)), (1.8755753121696364, (14, 11))]
    assert xyzparser.vdw_pairs_heavy_only == [(2.6992517149714557, (0, 2)), (12.017866990184016, (1, 5)), (22.374952502525975, (1, 10)), (14.485893016588888, (1, 11)), (12.609572203755663, (2, 5)), (22.246411219186342, (2, 10)), (14.28329119584782, (2, 11)), (11.911531969904996, (5, 10)), (4.17349782405302, (5, 11)), (8.617720255861196, (10, 12)), (2.8884964609210164, (3, 2)), (11.909939776267674, (4, 0)), (2.282871737370215, (6, 10)), (5.320284094605459, (7, 10)), (4.30813305799527, (8, 10)), (2.0460097248944904, (13, 11)), (1.8755753121696364, (14, 11))]

    assert len(xyzparser.vdw_pairs) == len(xyzparser.vdw_pairs_heavy_only)

