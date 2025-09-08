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
