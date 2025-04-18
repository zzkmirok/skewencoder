from .io import GeometryParser
import itertools
from collections.abc import Mapping
from .plumedkits import PLUMED_OBJ, DISTANCE, WALL, PYTORCH_MODEL
__all__ = ["PlumedInput"]

class PlumedInput:

    def __init__(self, geo_parser : GeometryParser = None, 
                 if_biased : bool = False, 
                 heavy_atom_only : bool = False,
                 **Options: Mapping[str, int | list[str] | str | Mapping[str, float | bool]]):
        
        self.unit = "A"
        self.time_step = 0.001
        self.heavy_atom_only = heavy_atom_only
        self.if_biased = if_biased
        self.Options = Options
        self.contr_all = False
        self.vdw_constr = True

        if geo_parser is None:
            raise ValueError("Empty input for geo_parser")
        self.adjacent_list : Mapping[int, list[int]] = {key + 1: [value + 1 for value in values] for key, values in geo_parser.adjacent_list.items()}
        self.atom_list : Mapping[str, list[int]] = {key: [value + 1 for value in values] for key, values in geo_parser.atom_list.items()}

        self.vdw_pairs : list[tuple[float, tuple[int, int]]] = None
        
        if geo_parser.vdw_pairs:
            self.vdw_pairs : list[tuple[float, tuple[int, int]]] = [(vdw_pair[0], (vdw_pair[1][0]+1, vdw_pair[1][1]+1)) for vdw_pair in geo_parser.vdw_pairs]
        # TODO: temporal solution for heavy_only is on
        # if not heavy_atom_only:
        #     if geo_parser.vdw_pairs:
        #         self.vdw_pairs : list[tuple[float, tuple[int, int]]] = [(vdw_pair[0], (vdw_pair[1][0]+1, vdw_pair[1][1]+1)) for vdw_pair in geo_parser.vdw_pairs]
        # else:
        #     if geo_parser.vdw_pairs_heavy_only:
        #         self.vdw_pairs : list[tuple[float, tuple[int, int]]] = [(vdw_pair[0], (vdw_pair[1][0]+1, vdw_pair[1][1]+1)) for vdw_pair in geo_parser.vdw_pairs_heavy_only]

        self.heavy_atom_pairs : list[tuple[str, tuple[int, int]]] = []
        self.h_heavy_pairs : list[tuple[str, tuple[int, int]]] = None

        for key, value in self.atom_list.items():
            if key != "H" and len(value) > 1:
                for pair in list(itertools.combinations(value,2)):
                    self.heavy_atom_pairs.append((f"{key}{pair[0]}{key}{pair[1]}", pair))
        
        heavy_atom_pairs_list_not_idential = list(itertools.combinations([key for key in self.atom_list.keys() if key !="H"], 2))

        for atom_type_pair in heavy_atom_pairs_list_not_idential:
            for pair in list(itertools.product(self.atom_list[atom_type_pair[0]], self.atom_list[atom_type_pair[1]])):
                self.heavy_atom_pairs.append((f"{atom_type_pair[0]}{pair[0]}{atom_type_pair[1]}{pair[1]}", pair))

        if not heavy_atom_only:
            self.h_heavy_pairs = []
            for H_atom in self.atom_list["H"]:
                for adj_atom in self.adjacent_list[H_atom]:
                    for atom_type in self.atom_list.keys():
                        if adj_atom in self.atom_list[atom_type]:
                            if atom_type == "H":
                                raise ValueError("2 H atoms are adjacent")
                            else:
                                self.h_heavy_pairs.append((f"H{H_atom}{atom_type}{adj_atom}", (H_atom, adj_atom)))
                            break
        
        self.distance_options : str | list[str] = None

        self.pytorch_model_files : Mapping[str, str] = {}

        # self.additional_commands : list[str] = None

        self.additional_Plumed_objects : list[PLUMED_OBJ] = None

        self.default_PYTORCH_MODEL_labels : list[str] = None
        self.default_NN_CV_WALL_labels : list[str] = None

        # TODO: Figure out a way to deal with input of bool: only_customized_pytorch model from outside
        self.only_customized_pytorch_model = False

        self.skew_wall_heavy : Mapping[str, float | bool] = None
        self.skew_wall_h_adj : Mapping[str, float | bool] = None
        self.print_stride : int = 10
        self.simulation_folder : str = "."
        # TODO: Is it normal to add assertion here?
        if self.Options is not None:
            for key in self.Options.keys():
                if key == "distance_options":
                    assert isinstance(self.Options[key], list)
                    self.distance_options = self.Options[key]
                elif key == "unit":
                    assert isinstance(self.Options[key], str)
                    self.unit = self.Options[key]
                elif key == "time_step":
                    assert isinstance(self.Options[key], float)
                    self.time_step = self.Options[key]
                elif key == "print_stride":
                    assert isinstance(self.Options[key], int)
                    self.print_stride = self.Options[key]
                elif key == "simulation_folder":
                    assert isinstance(self.Options[key], str)
                    self.simulation_folder = self.Options[key]
                elif key == "pytorch_model_heavy":
                    assert isinstance(self.Options[key], str)
                    self.pytorch_model_files["heavy"] = self.Options[key]
                elif key == "pytorch_model_h":
                    assert isinstance(self.Options[key], str)
                    self.pytorch_model_files["h_adj"] = self.Options[key]
                elif key == "skew_wall_heavy":
                    assert isinstance(self.Options[key], Mapping)
                    subkeys_to_include = ["is_lower_wall", "kappa", "pos", "offset"]
                    assert all(subkey in self.Options[key].keys() and self.Options[key][subkey] is not None for subkey in subkeys_to_include)
                    assert isinstance(self.Options[key]["is_lower_wall"], bool)
                    assert isinstance(self.Options[key]["kappa"], float)
                    assert isinstance(self.Options[key]["pos"], float)
                    assert isinstance(self.Options[key]["offset"], float)
                    self.skew_wall_heavy = self.Options[key]
                elif key == "skew_wall_h_adj":
                    assert isinstance(self.Options[key], Mapping)
                    subkeys_to_include = ["is_lower_wall", "kappa", "pos", "offset"]
                    assert all(subkey in self.Options[key].keys() and self.Options[key][subkey] is not None for subkey in subkeys_to_include)
                    assert isinstance(self.Options[key]["is_lower_wall"], bool)
                    assert isinstance(self.Options[key]["kappa"], float)
                    assert isinstance(self.Options[key]["pos"], float)
                    assert isinstance(self.Options[key]["offset"], float)
                    self.skew_wall_h_adj = self.Options[key]       
                elif key == "constr_all":
                    assert isinstance(self.Options[key], bool)
                    self.contr_all = self.Options[key]      
                elif key == "vdw_constr":
                    assert isinstance(self.Options[key], bool)
                    self.vdw_constr = self.Options[key]      
                else:
                    pass
        
        if self.if_biased:
            if not self.pytorch_model_files:
                raise ValueError("No pytorch file specified in input Options of a plumed input for a biased simulation")
        
            if not self.heavy_atom_only and ("h_adj" not in self.pytorch_model_files.keys()):
                raise ValueError("No h atom related pytorch file specified in input Options of a plumed input for a biased simulation")
            
            if not self.skew_wall_heavy:
                raise ValueError("No input parameter for heavy atom pair CV wall")

            if not self.heavy_atom_only and not self.skew_wall_h_adj:
                raise ValueError("No input parameter for h-adjacent atom pair CV wall")
    
    def build(self):
        plumed_input_file : list[str] = ["\n".join([self.gen_plumed_header(), self.gen_plumed_UNITS()])]

        plumed_input_file.append(self.gen_plumed_DISTANCE_snippet(self.heavy_atom_pairs, self.distance_options))

        if not self.heavy_atom_only:
            plumed_input_file.append(self.gen_plumed_DISTANCE_snippet(self.h_heavy_pairs, self.distance_options))

        if self.additional_Plumed_objects is not None:
            additional_snippet = self.gen_additional_commands_snippet()
            if additional_snippet is not None:
                plumed_input_file.append(self.gen_additional_commands_snippet())
        
        if self.vdw_constr and self.vdw_pairs:
            vdw_constr_snippet = self.gen_plumed_vdw_constr_snippet()
            if vdw_constr_snippet:
                plumed_input_file.append(vdw_constr_snippet)

        if self.contr_all:
            plumed_input_file.append(self.gen_plumed_pairwise_constr_snippet())
        
        if self.if_biased:
            additional_customize_pytorch_models : list[PYTORCH_MODEL] = []
            if self.additional_Plumed_objects is not None:
                additional_customize_pytorch_models = [model for model in self.additional_Plumed_objects if isinstance(model, PYTORCH_MODEL)]
            
            if not additional_customize_pytorch_models:
                additional_customize_pytorch_models = None
            plumed_input_file.append(self.gen_plumed_PYTORCH_MODEL_snippet(additional_customize_pytorch_models))

            additional_customize_walls : list[WALL] = []
            if self.additional_Plumed_objects is not None:
                additional_customize_walls = [wall for wall in self.additional_Plumed_objects if isinstance(wall, WALL)]
            
            if not additional_customize_walls:
                additional_customize_walls = None
            plumed_input_file.append(self.gen_plumed_WALL_snippet(additional_customize_walls))
        
        plumed_input_file.append(self.gen_plumed_PRINT())

        return "\n\n\n".join(plumed_input_file)
    

    # TODO: later move to plumedkit.py
    def gen_plumed_header(self):
        return "# vim:ft=plumed"
    

    # TODO: later move to plumedkit.py
    def gen_plumed_UNITS(self):
        return f"UNITS LENGTH={self.unit} TIME={self.time_step}  #Amstroeng, hartree, fs"
    
    def add_additional_command(self, plumed_object: PLUMED_OBJ = None):
        # if self.additional_commands is None :
        #     self.additional_commands : list[str] = []
        
        if self.additional_Plumed_objects is None:
            self.additional_Plumed_objects : list[PLUMED_OBJ] = []
        
        if plumed_object is not None:
            # self.additional_commands.append(plumed_object.build())
            self.additional_Plumed_objects.append(plumed_object)
        else:
            # self.additional_commands = None
            self.additional_Plumed_objects = None
            raise ValueError("No Plumed Object in add_additional_command()")

    def gen_additional_commands_snippet(self):
        if self.additional_Plumed_objects is None:
            raise ValueError("No additional commands added")
        else:
            additional_obj_not_wall_not_Pytorch = [model for model in self.additional_Plumed_objects if not (isinstance(model, WALL) or isinstance(model, PYTORCH_MODEL))]
            if len(additional_obj_not_wall_not_Pytorch) > 0:
                additional_commands = []
                for model in additional_obj_not_wall_not_Pytorch:
                    additional_commands.append(model.build())
                return "\n".join(additional_commands)
            else:
                return None       
    

    def gen_plumed_DISTANCE_snippet(self, atoms_pair_list : list[tuple[str, tuple]], Options : str | list[str] | None = None) -> str:
        DISTANCE_snippet : list[str] = []
        for pair in atoms_pair_list:
            DISTANCE_snippet.append(DISTANCE(label=pair[0], atoms=pair[1], Options=Options).build())

        if len(DISTANCE_snippet) > 0:
            return "\n".join(DISTANCE_snippet)
        else:
            raise ValueError("empty plumed DISTANCES snippet")    

    
    def gen_plumed_PYTORCH_MODEL_snippet(self, customized_models : list[PYTORCH_MODEL] | None = None):
        PYTORCH_MODEL_snippet : list[str] = []
        self.default_PYTORCH_MODEL_labels = []
        if self.only_customized_pytorch_model and customized_models is None:
            raise ValueError("No customized models defined when activate only_customized pytorch model.")
        
        if customized_models is not None:
            for model in customized_models:
                PYTORCH_MODEL_snippet.append(model.build())

        if not self.only_customized_pytorch_model:
            heavy_atom_pair_label_list = [pair[0] for pair in self.heavy_atom_pairs]
            PYTORCH_MODEL_snippet.append(PYTORCH_MODEL(label="cv_heavy", FILE=self.pytorch_model_files["heavy"], ARG=heavy_atom_pair_label_list).build())
            self.default_PYTORCH_MODEL_labels.append("cv_heavy")

            if not self.heavy_atom_only:
                h_adj_atom_pair_label_list = [pair[0] for pair in self.h_heavy_pairs]
                PYTORCH_MODEL_snippet.append(PYTORCH_MODEL(label="cv_h_adj", FILE=self.pytorch_model_files["h_adj"], ARG=h_adj_atom_pair_label_list).build())
                self.default_PYTORCH_MODEL_labels.append("cv_h_adj")

        return "\n".join(PYTORCH_MODEL_snippet)

    def gen_plumed_pairwise_constr_snippet(self):
        constr_snippet : list[str] = []
        # TODO: if a C-C-C-C-C chain exist, there will be a problem
        for pair in self.heavy_atom_pairs:
            temporal_contraint_wall = WALL(label="constr_"+pair[0], is_lower_wall=False, ARG=[pair[0]], AT=[8.0], KAPPA=[200])
            constr_snippet.append(temporal_contraint_wall.build())
        
        return "\n".join(constr_snippet)

            
    def gen_plumed_WALL_snippet(self, customized_walls : list[WALL] | None = None):
        WALL_snippet : list[str] = []
        self.default_NN_CV_WALL_labels = []
        
        if customized_walls is not None:
            for wall in customized_walls:
                WALL_snippet.append(wall.build())

        temp_AT = self.skew_wall_heavy["pos"] + (self.skew_wall_heavy["offset"] if self.skew_wall_heavy["is_lower_wall"] else (-self.skew_wall_heavy["offset"]))
        WALL_snippet.append(WALL(label="cv_wall_heavy", 
                                is_lower_wall=self.skew_wall_heavy["is_lower_wall"],
                                ARG=self.default_PYTORCH_MODEL_labels[0]+".node-0",
                                AT=temp_AT,
                                KAPPA=self.skew_wall_heavy["kappa"]).build()) #cv_heavy 
        self.default_NN_CV_WALL_labels.append("cv_wall_heavy")

        if not self.heavy_atom_only:
            temp_AT = self.skew_wall_h_adj["pos"] + (self.skew_wall_h_adj["offset"] if self.skew_wall_h_adj["is_lower_wall"] else (-self.skew_wall_h_adj["offset"]))
            WALL_snippet.append(WALL(label="cv_wall_h_adj", 
                                is_lower_wall=self.skew_wall_h_adj["is_lower_wall"],
                                ARG=self.default_PYTORCH_MODEL_labels[1]+".node-0",
                                AT=temp_AT,
                                KAPPA=self.skew_wall_h_adj["kappa"]).build()) #cv_heavy 
            self.default_NN_CV_WALL_labels.append("cv_wall_h_adj")

        return "\n".join(WALL_snippet)
    
    def gen_plumed_PRINT(self):
        print_args = [ pair[0] for pair in self.heavy_atom_pairs ]
        if not self.heavy_atom_only:
            for pair in self.h_heavy_pairs:
                print_args.append(pair[0])
        
        if self.additional_Plumed_objects is not None:
            for obj in self.additional_Plumed_objects:
                if isinstance(obj, DISTANCE):
                    print_args.append(obj.label)
        
        if self.if_biased:
            for label in self.default_PYTORCH_MODEL_labels:
                print_args.append((label+".*"))

            if self.additional_Plumed_objects is not None:
                for obj in self.additional_Plumed_objects:
                    if isinstance(obj, PYTORCH_MODEL):
                        print_args.append(obj.label+".*")
        
        return f"PRINT FMT=%g STRIDE={self.print_stride} FILE={self.simulation_folder}/COLVAR ARG={','.join(print_args)}"
    
    def gen_plumed_vdw_constr_snippet(self):
        vdw_constr_snippet : list[str] = []

        vdw_radius = {"C": 1.7, "O": 1.52, "H": 1.2, "N": 1.55}

        for index, vdw_pair in enumerate(self.vdw_pairs):
            vdw_distance = 0.0
            bond_type = set()
            for atom in vdw_pair[1]:
                for key,value in self.atom_list.items():
                    if atom in value:
                        bond_type.add(key)
            if len(bond_type) == 1:
                for key, value in vdw_radius.items():
                    if key in bond_type:
                        vdw_distance = 2*value
                        break
            elif len(bond_type) == 2:
                for key,value in vdw_radius.items():
                    if key in bond_type:
                        vdw_distance += value
            else:
                raise ValueError("Failed to read atom pairs with shortest distances among molecules")


            if vdw_pair[0] > vdw_distance:
                vdw_d_label = f"vdw_d_{index}"
                vdw_constr_snippet.append(DISTANCE(label=vdw_d_label, atoms=vdw_pair[1]).build())
                vdw_wall_label = f"vdw_wall_{index}"
                vdw_constr_snippet.append(WALL(label=vdw_wall_label,is_lower_wall=False,ARG=vdw_d_label, AT=vdw_distance, KAPPA=[250]).build())

        if len(vdw_constr_snippet) == 0:
            return None
        else:
            return  "\n".join(vdw_constr_snippet)
