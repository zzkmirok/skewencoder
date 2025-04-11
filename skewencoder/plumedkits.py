from abc import ABC, abstractmethod


__all__ = ["PLUMED_OBJ", "DISTANCE", "WALL", "PYTORCH_MODEL"]

class PLUMED_OBJ(ABC):
    def __init__(self, *, label: str = None, Options : str | list[str] = None):
        """
        Initialize a PLUMED object with a label and a set of options.
        
        Subclasses should extend this initializer with additional parameters.
        """
        if label is None:
            raise ValueError("the label should not be empty")
        self._label = label
        if isinstance(Options, str):
            Options = [Options]

        self.Options = Options
        if self.Options is not None:
            self.Options_str = " ".join(self.Options)
        else:
            self.Options_str = ""

    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, value):
        if not value:
            raise ValueError("the label should not be empty")
        self._label = value
    
    @abstractmethod
    def build(self):
        return ""

class DISTANCE(PLUMED_OBJ):
    def __init__(self, *, label: str, atoms: tuple[int], Options : str | list[str] = None):
        super().__init__(label=label, Options=Options)
        if len(atoms) != 2:
            raise ValueError(f"input atom list invalid, list of atoms = {atoms}")
        self.atoms = atoms

    def build(self):
        atoms_str = f"ATOMS={','.join(map(str, self.atoms))}"
        label_prefix = f"{self.label}:"
        return " ".join([label_prefix, self.__class__.__name__, atoms_str, self.Options_str]).rstrip()
    

class WALL(PLUMED_OBJ):
    def __init__(self, *, label = None, Options = None, is_lower_wall : bool = True, 
                ARG : str | list[str] = None, 
                AT : float | list[float] = None, 
                KAPPA : float | list[float] = None,
                EXP : float | list[float] = [2.0], 
                EPS : float | list[float] = [1.0],  
                OFFSET : float | list[float] = [0.0]):
        super().__init__(label=label, Options=Options)
        if ARG is None:
            raise ValueError(f"Input parameter ARG of {self.__class__.__name__} should not be none")

        if AT is None:
            raise ValueError(f"Input parameter AT of {self.__class__.__name__} should not be none")

        if KAPPA is None:
            raise ValueError(f"Input parameter KAPPA of {self.__class__.__name__} should not be none")

        if isinstance(ARG, str):
            ARG = [ARG]

        if isinstance(AT, float):
            AT = [AT]

        if isinstance(KAPPA, float):
            KAPPA = [KAPPA]

        if isinstance(EXP, float):
            EXP = [EXP]

        if isinstance(EPS, float):
            EPS = [EPS]

        if isinstance(OFFSET, float):
            OFFSET = [OFFSET]

        if len({len(lst) for lst in [ARG, AT, KAPPA]}) != 1:
            raise ValueError("Input dimensions of ARG, AT and KAPPA didn't match with each other")
        
        self.len_cvs = len(ARG)
        
        if EXP == [2.0] and EPS == [1.0] and OFFSET == [0.0]:
            self.exp = EXP * self.len_cvs
            self.eps = EPS * self.len_cvs
            self.offset = OFFSET * self.len_cvs
        else:
            if len({len(lst) for lst in [EXP, EPS, OFFSET]}) != 1:
                raise ValueError("Input dimensions EXP, EPS, OFFSET didn't match with each other")
            
            if len(EXP) != self.len_cvs:
                raise ValueError("EXP and ARG didn't match with each other")
            
            self.exp = EXP
            self.eps = EPS
            self.offset = OFFSET

        
        self.arg = ARG
        self.at = AT
        self.kappa = KAPPA
        self.is_lower_wall = is_lower_wall

    def build(self):
        label_prefix = f"{self.label}:"
        wall_type = "LOWER_WALL"
        if not self.is_lower_wall:
            wall_type = "UPPER_WALL"
        arg_str = f"ARG={','.join(self.arg)}"
        at_str = f"AT={','.join(map(str, self.at))}"
        kappa_str = f"KAPPA={','.join(map(str, self.kappa))}"
        exp_str = f"EXP={','.join(map(str, self.exp))}"
        eps_str = f"EPS={','.join(map(str, self.eps))}"
        offset_str = f"OFFSET={','.join(map(str, self.offset))}"
        return " ".join([label_prefix, wall_type, arg_str, at_str, kappa_str, exp_str, eps_str, offset_str, self.Options_str]).rstrip()
    

class PYTORCH_MODEL(PLUMED_OBJ):
    def __init__(self, *, label = None, Options = None, FILE: str = None, ARG : str | list[str] = None):
        super().__init__(label=label, Options=Options)

        if ARG is None:
            raise ValueError(f"Input parameter ARG of {self.__class__.__name__} should not be none")
        
        if FILE is None:
            raise ValueError(f"Input parameter FILE of {self.__class__.__name__} should not be none")

        if isinstance(ARG, str):
            ARG = [ARG]        
        self.arg = ARG
        self.file = FILE

    def build(self):
        label_prefix = f"{self.label}:"
        arg_str = f"ARG={','.join(self.arg)}"
        file_str = f"FILE={self.file}"
        return " ".join([label_prefix, self.__class__.__name__, file_str, arg_str]).rstrip()