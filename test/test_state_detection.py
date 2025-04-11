from skewencoder.state_detection import transform_colvar_key

def test_transform_colvar_key():
    colvar_key = "H4C1"
    bond_type = transform_colvar_key(colvar_key=colvar_key, pattern = r"^(H)\d+([A-GI-Za-gi-z]+)\d+$")
    print(bond_type)

