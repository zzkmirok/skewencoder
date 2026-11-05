from skewencoder.state_detection import transform_colvar_key

def test_transform_colvar_key():
    colvar_key = "H4C1"
    bond_type = transform_colvar_key(colvar_key=colvar_key, pattern = r"^(H)\d+([A-GI-Za-gi-z]+)\d+$")
    assert bond_type == "H-C"

    colvar_key = "O12H13"
    bond_type = transform_colvar_key(colvar_key=colvar_key)
    assert bond_type == "H-O"

    colvar_key = "Cu1C14"
    bond_type = transform_colvar_key(colvar_key=colvar_key)
    assert bond_type == "C-Cu"
    
    colvar_key = "O12Cu1"
    bond_type = transform_colvar_key(colvar_key=colvar_key)
    assert bond_type == "O-Cu"
    
    colvar_key = "H14Cu1"
    bond_type = transform_colvar_key(colvar_key=colvar_key)
    assert bond_type == "H-Cu"
    
    colvar_key = "Cu2O2"
    bond_type = transform_colvar_key(colvar_key=colvar_key)
    assert bond_type == "O-Cu"
    
    # TODO: temporal compatibality test. Later "o3h11" should be extracted as capital "H-O"
    colvar_key = "o4h3"
    bond_type = transform_colvar_key(colvar_key=colvar_key)
    assert bond_type == "h-o"
    colvar_key = "o3h11"
    bond_type = transform_colvar_key(colvar_key=colvar_key)
    assert bond_type == "h-o"
    colvar_key = "o3h10"
    bond_type = transform_colvar_key(colvar_key=colvar_key)
    assert bond_type == "h-o"


def test_chlorine_bond_types():
    from skewencoder.state_detection import Bond_type_lib, transform_colvar_key

    lib = Bond_type_lib()
    lib.build_default()
    # phosgene + methylamine heavy-atom descriptors all resolve to a known bond type
    for key in ["C1Cl2", "Cl2Cl3", "Cl2O4", "Cl3N9", "H10Cl2", "C1C5", "C1O4", "C5N9", "O4N9"]:
        assert transform_colvar_key(key) in lib.bond_type_dict, key
    assert transform_colvar_key("Cl2C1") == "C-Cl"


def _colvar(path, columns, rows=40):
    names = list(columns)
    with open(path, "w") as f:
        f.write("#! FIELDS time " + " ".join(names) + "\n")
        for t in range(rows):
            f.write(f"{t} " + " ".join(str(columns[k]) for k in names) + "\n")
    return str(path)


def test_bond_switch_boundaries():
    from skewencoder.state_detection import Bond_type_lib, State_detection

    lib = Bond_type_lib()
    lib.build_default()
    sd = State_detection((0.3, 0.7), bond_type_dict=lib.bond_type_dict, n_heavy_atom_pairs=1)
    for bond_type, options in lib.bond_type_dict.items():
        L = options["bond length"]
        sw = sd.sw_dict[bond_type]
        assert sw(0.8 * L) == 1.0  # shorter (double) bond
        assert sw(1.12 * L) > 0.7  # stretched bond at 300 K
        assert 0.3 < sw(1.25 * L) < 0.7
        assert sw(1.45 * L) < 0.3  # 1-3 pair


def test_phosgene_methylamine_states(tmp_path):
    # last-quarter mean distances of the CH3NH2 + COCl2 DFTB+ run (test_300_42)
    from skewencoder.state_detection import Bond_type_lib, State_detection

    lib = Bond_type_lib()
    lib.build_default()
    sd = State_detection((0.3, 0.7), bond_type_dict=lib.bond_type_dict, n_heavy_atom_pairs=5)
    unbiased = {"C1Cl2": 1.83, "Cl2Cl3": 2.91, "C1O4": 1.19, "C1N9": 1.80, "Cl2O4": 2.63}
    iter_0 = {"C1Cl2": 1.97, "Cl2Cl3": 3.07, "C1O4": 1.20, "C1N9": 1.68, "Cl2O4": 2.74}
    iter_1 = {"C1Cl2": 4.31, "Cl2Cl3": 5.16, "C1O4": 1.20, "C1N9": 1.33, "Cl2O4": 3.26}
    assert sd(_colvar(tmp_path / "u", unbiased)) == (True, True)
    assert sd(_colvar(tmp_path / "0", iter_0)) == (True, False)
    assert sd(_colvar(tmp_path / "1", iter_1)) == (True, True)
    assert sd.n_states == 2
    assert sd.current_state == 2
