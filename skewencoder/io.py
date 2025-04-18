"""Input/output functions."""
try:
    import pandas as pd
except ImportError as e:
    raise ImportError(
        "pandas is required to use the i/o utils (mlcolvar.utils.io)\n", e
    )
import logging

import numpy as np
import torch
import pathlib
import os
import urllib.request
from typing import Union

from collections.abc import Sequence, Mapping, Set

from mlcolvar.data import DictDataset, DictModule, DictLoader

from .switchfunction import SwitchFun

import networkx as nx

import itertools

__all__ = ["GeometryParser","load_dataframe", "plumed_to_pandas", "create_dataset_from_files", "load_data"]


def is_plumed_file(filename):
    """
    Check if given file is in PLUMED format.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    bool
        wheter is a plumed output file
    """
    headers = pd.read_csv(filename, sep=" ", skipinitialspace=True, nrows=0)
    is_plumed = True if " ".join(headers.columns[:2]) == "#! FIELDS" else False
    return is_plumed


def plumed_to_pandas(filename="./COLVAR"):
    """
    Load a PLUMED file and save it to a dataframe.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    df : DataFrame
        Collective variables dataframe
    """
    skip_rows = 1
    # Read header
    headers = pd.read_csv(filename, sep=" ", skipinitialspace=True, nrows=0)
    # Discard #! FIELDS
    headers = headers.columns[2:]
    # Load dataframe and use headers for columns names
    df = pd.read_csv(
        filename,
        sep=" ",
        skipinitialspace=True,
        header=None,
        skiprows=range(skip_rows),
        names=headers,
        comment="#",
    )

    return df


def load_dataframe(
    file_names, start=0, stop=None, stride=1, delete_download=True, with_numpy=False, **kwargs
):
    """Load dataframe(s) from file(s). It can be used also to open files from internet (if the string contains http).
    In case of PLUMED colvar files automatically handles the column names, otherwise it is just a wrapper for pd.load_csv function.

    Parameters
    ----------
    filenames : str or list[str]
        filenames to be loaded
    start: int, optional
        read from this row, default 0
    stop: int, optional
        read until this row, default None
    stride: int, optional
        read every this number, default 1
    delete_download: bool, optinal
        whether to delete the downloaded file after it has been loaded, default True.
    kwargs:
        keyword arguments passed to pd.load_csv function

    Returns
    -------
    pandas.DataFrame
        Dataframe

    Raises
    ------
    TypeError
        if data is not a valid type
    """

    # if it is a single string
    if type(file_names) == str:
        file_names = [file_names]
    elif type(file_names) != list:
        raise TypeError(
            f"only strings or list of strings are supported, not {type(file_names)}."
        )

    # list of file_names
    df_list = []
    for i, filename in enumerate(file_names):
        # check if filename is an url
        download = False
        if "http" in filename:
            download = True
            url = filename
            filename = "tmp_" + filename.split("/")[-1]
            urllib.request.urlretrieve(url, filename)

        # check if file is in PLUMED format
        if is_plumed_file(filename):
            df_tmp = plumed_to_pandas(filename)
            df_tmp["walker"] = [i for _ in range(len(df_tmp))]
            df_tmp = df_tmp.iloc[start:stop:stride, :]
            df_list.append(df_tmp)

        # else use read_csv with optional kwargs
        else:
            df_tmp = pd.read_csv(filename, **kwargs)
            df_tmp["walker"] = [i for _ in range(len(df_tmp))]
            df_tmp = df_tmp.iloc[start:stop:stride, :]
            df_list.append(df_tmp)

        # delete temporary data if necessary
        if download:
            if delete_download:
                os.remove(filename)
            else:
                print(f"downloaded file ({url}) saved as ({filename}).")

        # concatenate
        df = pd.concat(df_list)
        df.reset_index(drop=True, inplace=True)
    if with_numpy:
        return df, df.to_numpy()
    else:
        return df


def create_dataset_from_files(
    file_names: Union[list, str],
    folder: str = None,
    create_labels: bool = None,
    load_args: list = None,
    filter_args: dict = None,
    modifier_function=None,
    return_dataframe: bool = False,
    verbose: bool = True,
    **kwargs,
):
    """
    Initialize a dataset from (a list of) files. Suitable for supervised/unsupervised tasks.

    Parameters
    ----------
    file_names : list
        Names of files from which import the data
    folder : str, optional
        Common path for the files to be imported, by default None. If set, filenames become 'folder/file_name'.
    create_labels: bool, optional
        Assign a label to each file, default True if more than a file is given, otherwise False
    load_args: list[dict], optional
        List of dictionaries with the arguments passed to load_dataframe function for each file (keys: start,stop,stride and pandas.read_csv options), by default None
    filter_args: dict, optional
        Dictionary of arguments which are passed to df.filter() to select descriptors (keys: items, like, regex), by default None
        Note that 'time' and '*.bias' columns are always discarded.
    return_dataframe : bool, optional
        Return also the imported Pandas dataframe for convenience, by default False
    modifier_function : function, optional
        Function to be applied to the input data, by default None.
    verbose : bool, optional
        Print info on the datasets, by default True
    kwargs : optional
        args passed to mlcolvar.utils.io.load_dataframe

    Returns
    -------
    torch.Dataset
        Torch labeled dataset of the given data
    optional, pandas.Dataframe
        Pandas dataframe of the given data #TODO improve

    See also
    --------
    mlcolvar.utils.io.load_dataframe
        Function that is used to load the files

    """
    if isinstance(file_names, str):
        file_names = [file_names]

    num_files = len(file_names)

    # set file paths
    if folder is not None:
        file_names = [os.path.join(folder, fname) for fname in file_names]

    # check if per file args are given, otherwise set to {}
    if load_args is None:
        load_args = [{} for _ in file_names]
    else:
        if (not isinstance(load_args, list)) or (len(file_names) != len(load_args)):
            raise TypeError(
                "load_args should be a list of dictionaries of arguments of same length as file_names. If you want to use the same args for all file pass them directly as **kwargs."
            )

    # check if create_labels if given, otherwise set it to True if more than one file is given
    if create_labels is None:
        create_labels = False if len(file_names) == 1 else True

    # initialize pandas dataframe
    df = pd.DataFrame()

    # load data
    for i in range(num_files):
        df_tmp = load_dataframe(file_names[i], **load_args[i], **kwargs)

        # add label in the dataframe
        if create_labels:
            df_tmp["labels"] = i
        if verbose:
            print(f"Class {i} dataframe shape: ", np.shape(df_tmp))

        # update collective dataframe
        df = pd.concat([df, df_tmp], ignore_index=True)

    # filter inputs
    df_data = df.filter(**filter_args) if filter_args is not None else df.copy()
    df_data = df_data.filter(regex="^(?!.*labels)^(?!.*time)^(?!.*bias)^(?!.*walker)")

    if verbose:
        print(f"\n - Loaded dataframe {df.shape}:", list(df.columns))
        print(f" - Descriptors {df_data.shape}:", list(df_data.columns))

    # apply transformation
    if modifier_function is not None:
        df_data = df_data.apply(modifier_function)

    # create DictDataset
    dictionary = {"data": torch.Tensor(df_data.values)}
    if create_labels:
        dictionary["labels"] = torch.Tensor(df["labels"].values)
    dataset = DictDataset(dictionary, feature_names=df_data.columns.values)

    if return_dataframe:
        return dataset, df
    else:
        return dataset



def load_data(filenames_iter,filenames_all, multiple = 0, bs=0, pattern = r"^([A-Za-z]+)\d+([A-Za-z]+)\d+$", verbose=False):
    # load for AE, filenames_all can be a list so that all data are loaded
    AE_dataset, AE_df = create_dataset_from_files(filenames_all, return_dataframe=True, filter_args={'regex': pattern}, create_labels=False, verbose=verbose)
    skewness_dataset, skewness_df = create_dataset_from_files(filenames_iter, return_dataframe=True, filter_args={'regex': pattern}, create_labels=False, verbose=verbose)
    # create multitask datamodule with both datasets
    if bs == 0:
        batch_size_list = bs
    else:
        batch_size_list=[[bs * multiple, bs],[bs * multiple, bs]]
    datamodule = DictModule(dataset=[AE_dataset, skewness_dataset], batch_size=batch_size_list)
    return AE_dataset, skewness_dataset, datamodule, AE_df, skewness_df

class GeometryParser:
    def __init__(self, coord_file : str | pathlib.Path = None):
        if coord_file is None:
            raise ValueError("empty coord file")
        
        if isinstance(coord_file, str):
            coord_file = pathlib.Path(coord_file)
        self.coord_file = coord_file
            
        self.atom_list : Mapping[str, list[int]] = {}
        self.adjacent_list: Mapping[int, list[int]] = {}
        coord_list = self.parse_atoms_list(self.coord_file.suffix)
        if len(coord_list) > 0:
            self.coordinates_list = np.array(coord_list)
        else:
            raise ValueError("Failed to parse atom list")
        self.adjacent_list : Mapping[int, list[int]] = self.parse_adj_list()

        self.connected_components = self.group_components()

        if len(self.connected_components) == 0:
            raise ValueError("Failed to group components")
        
        self.vdw_pairs : list[tuple[float, tuple[int, int]]] = None
        self.vdw_pairs_heavy_only : list[tuple[float, tuple[int, int]]] = None

        if len(self.connected_components) > 1:
            self.vdw_pairs = self.gen_vdw_pairs()
            self.vdw_pairs_heavy_only = self.gen_vdw_pairs(only_heavy=True)



    def parse_atoms_list(self, file_extention : str):
        coordinates_list = []
        if file_extention == ".xyz":
            # read the last frame TODO: later may read several last frames
                with open(self.coord_file, 'r') as f:
                    lines = f.readlines()

                    # Extract number of atoms from first line
                    num_atoms = int(lines[0].strip())

                    # Calculate total lines per frame (atoms + 2)
                    lines_per_frame = num_atoms + 2

                    # Find start index for last frame
                    start_index_last_frame = len(lines) - lines_per_frame
                    
                    # Extract coordinates for last frame
                    last_frame_lines = lines[start_index_last_frame + 2:start_index_last_frame + 2 + num_atoms]
                
                for index, line in enumerate(last_frame_lines):
                    parts = line.split()
                    atom_type = parts[0]
                    x, y, z = map(float, parts[1:])
    
                # Update mapping dictionary
                    if atom_type not in self.atom_list:
                        self.atom_list[atom_type] = []
                    self.atom_list[atom_type].append(index)
    
                # Append coordinates to list
                    coordinates_list.append([x, y, z])
        elif file_extention == ".dat":
            with open(self.coord_file, 'r') as f:
                for index, line in enumerate(f):
                    parts = line.split()
                    atom_type = parts[0]
                    x, y, z = map(float, parts[1:])
    
                # Update mapping dictionary
                    if atom_type not in self.atom_list:
                        self.atom_list[atom_type] = []
                    self.atom_list[atom_type].append(index)
    
                # Append coordinates to list
                    coordinates_list.append([x, y, z])
        else:
            raise TypeError("coord_file not match .xyz or .dat")
        
        return coordinates_list
    
    def parse_adj_list(self, r_heavy_atoms: float = 1.7, r_H : float = 1.2):
        adj_list: Mapping[int, list[int]] = {}
        sw = SwitchFun(r0=r_heavy_atoms)
        for i in range(self.coordinates_list.shape[0] - 1):
            for j in range(i+1, self.coordinates_list.shape[0]):
                for key in self.atom_list.keys():
                    if (i in self.atom_list[key]) or (j in self.atom_list[key]):
                        if key == "H":
                            sw.set_r0(r_H)
                        break
                distance = np.linalg.norm(self.coordinates_list[i,:] - self.coordinates_list[j,:])
                if i not in adj_list:
                    adj_list[i] = []
                if j not in adj_list:
                    adj_list[j] = []
                if sw(distance) > 0.5:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
                sw.set_r0(r_heavy_atoms)
        return adj_list
    
    def group_components(self):
        G = nx.Graph(self.adjacent_list)

        # Find all connected components (isolated subgraphs)
        connected_components = list(nx.connected_components(G))
        
        return connected_components
    


    def shortest_distance_and_atoms(self, group_a_indices, group_b_indices):
        coords_a = self.coordinates_list[list(group_a_indices)]
        coords_b = self.coordinates_list[list(group_b_indices)]
        
        # Calculate pairwise distances using broadcasting
        diff = coords_a[:, np.newaxis] - coords_b[np.newaxis]
        distances = np.linalg.norm(diff, axis=2)
        
        # Find the minimum distance and its indices
        min_index_flat = np.argmin(distances)
        
        # Convert flat index back to row/column indices
        min_index_a_idx, min_index_b_idx = np.unravel_index(min_index_flat, distances.shape)
        
        min_distance = distances[min_index_a_idx][min_index_b_idx]
        
        # Extract corresponding node numbers from original groups
        atom_a_number = list(group_a_indices)[min_index_a_idx]
        atom_b_number = list(group_b_indices)[min_index_b_idx]
        
        return min_distance, (atom_a_number, atom_b_number)
    
    """
    Criterion
    1. The group with only H should be matched with the closest heavy atom (only one)
    2. Other groups with at least 1 heavy atom must match with all other groups with heavy atom (besides the group with only H)
    """
    
    def gen_vdw_pairs(self, only_heavy : bool = False):
        vdw_pairs : list[tuple[float, tuple[int, int]]] = []
        connected_components : Mapping[str, list[set[int]]] = {"with_heavy": [], "only_H": []}
        nodes_to_remove = set(self.atom_list["H"])
    # Filter each subgraph
        for subgraph in self.connected_components:
            # Remove nodes with label 'H'
            filtered_subgraph = {node for node in subgraph if node not in nodes_to_remove}
            if len(filtered_subgraph) == 0:
                connected_components["only_H"].append(subgraph)
            else:
                if only_heavy:
                    connected_components["with_heavy"].append(filtered_subgraph)
                else:
                    connected_components["with_heavy"].append(subgraph)
        
        if len(connected_components["with_heavy"]) > 0:
            combinations = itertools.combinations(connected_components["with_heavy"], 2)
            for subgraph_combo in combinations:
                min_distance, (atom_a, atom_b) = self.shortest_distance_and_atoms(subgraph_combo[0], subgraph_combo[1])
                vdw_pairs.append((min_distance, (atom_a, atom_b)))

            if len(connected_components["only_H"]) > 0:
                for H_group in connected_components["only_H"]:
                    vdw_pairs_for_compare : list[tuple[float, tuple[int, int]]] = []
                    for heavy_group in connected_components["with_heavy"]:
                        min_distance, (atom_a, atom_b) = self.shortest_distance_and_atoms(H_group, heavy_group)
                        vdw_pairs_for_compare.append((min_distance, (atom_a, atom_b)))
                    min_id = np.argmin(np.array([vdw_pair[0] for vdw_pair in vdw_pairs_for_compare]))
                    vdw_pairs.append(vdw_pairs_for_compare[min_id])
                        
        else:
            raise ValueError("System exploded!!!") 


        return vdw_pairs



if __name__ == "__main__":
    # test_datasetFromFile()
    x = torch.arange(1,11)
    d = {'data': x.unsqueeze(1), 'labels': x**2}
    dataloader = DictLoader(d, batch_size=1, shuffle=False)
    print(dataloader.dataset_len)