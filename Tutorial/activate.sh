#!/bin/zsh

echo "restoring modules for plumed env"
module load GCC/12.2.0
module load OpenMPI/4.1.4
module load PLUMED/2.9.0
module unload PLUMED/2.9.0

echo "adding libtorch to cpath"
_MY_LIBTORCH="/rwthfs/rz/cluster/home/yy508225/myC_lib/Libtorch/libtorch"
CPATH="${_MY_LIBTORCH}/include/torch/csrc/api/include/:${_MY_LIBTORCH}/include/:${_MY_LIBTORCH}/include/torch:$CPATH"
INCLUDE="${_MY_LIBTORCH}/include/torch/csrc/api/include/:${_MY_LIBTORCH}/include/:${_MY_LIBTORCH}/include/torch:$INCLUDE"
LIBRARY_PATH="${_MY_LIBTORCH}/lib:$LIBRARY_PATH"
LD_LIBRARY_PATH="${_MY_LIBTORCH}/lib:$LD_LIBRARY_PATH"

echo "loading modules for cp2k"
module load CMake/3.26.3
module load Libint/2.7.2-lmax-6-cp2k
module load libxc/6.1.0
module load libxsmm/1.17
module load HDF5/1.14.0
module load libvori/220621

echo "configuing plumed env"
MY_PLUMED_PATH="/rwthfs/rz/cluster/home/yy508225/myplumed/plumed2.9.0"
PATH="${MY_PLUMED_PATH}/bin:$PATH"
PLUMED_KERNEL="${MY_PLUMED_PATH}/lib/libplumedKernel.so"
PYTHONPATH="${MY_PLUMED_PATH}/lib/plumed/python:$PYTHONPATH"
PLUMED_ROOT="${MY_PLUMED_PATH}/lib/plumed/"
LD_LIBRARY_PATH="${MY_PLUMED_PATH}/lib/:$LD_LIBRARY_PATH"

PATH="/home/yy508225/mycp2k/cp2k-2023.1/exe/local:$PATH"
echo "complete CP2K env configuration"

PATH="/home/yy508225/.local/bin:${PATH}"
# The variables below must be unset in the deactivate phase
export CPATH
export INCLUDE
export LIBRARY_PATH

export PATH
export LD_LIBRARY_PATH

export PYTHONPATH
export PLUMED_KERNEL
export PLUMED_ROOT
