from skewencoder.plumedkits import DISTANCE, PYTORCH_MODEL, WALL

def test_DISTANCES():
    distance1 = DISTANCE(label="d1", atoms=(1,2), Options=["NOPBC"])
    distance2 = DISTANCE(label="d2", atoms=(3,4), Options=["NOPBC", "COMPONENTS"])

    assert distance1.build() == "d1: DISTANCE ATOMS=1,2 NOPBC"
    assert distance2.build() == "d2: DISTANCE ATOMS=3,4 NOPBC COMPONENTS"

def test_WALLS():
    wall1 = WALL(label="lwall_1d", is_lower_wall=True, ARG="cv.node-0", AT=0.1, KAPPA=200.0)
    wall2 = WALL(label="lwall_2d", is_lower_wall=True, ARG=["d1","d2"], AT=[0.1, 0.2], KAPPA=[200.0,100.0])
    wall3 = WALL(label="uwall_2d_customize", is_lower_wall=False, ARG=["d3","d4"], AT=[0.1, 0.2], KAPPA=[50.0, 100.0], EXP=[2.0,1.0], EPS=[1.0,2.0], OFFSET=[0.0, 0.1])

    assert wall1.build() == "lwall_1d: LOWER_WALL ARG=cv.node-0 AT=0.1 KAPPA=200.0 EXP=2.0 EPS=1.0 OFFSET=0.0"
    assert wall2.build() == "lwall_2d: LOWER_WALL ARG=d1,d2 AT=0.1,0.2 KAPPA=200.0,100.0 EXP=2.0,2.0 EPS=1.0,1.0 OFFSET=0.0,0.0"
    assert wall3.build() == "uwall_2d_customize: UPPER_WALL ARG=d3,d4 AT=0.1,0.2 KAPPA=50.0,100.0 EXP=2.0,1.0 EPS=1.0,2.0 OFFSET=0.0,0.1"

def test_PYTORCH_MODELS():
    pytorch_model1 = PYTORCH_MODEL(label="cv", FILE="./module.pt", ARG="d1")
    pytorch_model2 = PYTORCH_MODEL(label="cv", FILE="./module.pt", ARG=["d1","d2"])
    assert pytorch_model1.build() == "cv: PYTORCH_MODEL FILE=./module.pt ARG=d1"
    assert pytorch_model2.build() == "cv: PYTORCH_MODEL FILE=./module.pt ARG=d1,d2"