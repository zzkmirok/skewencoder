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
