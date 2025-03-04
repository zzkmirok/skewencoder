## Installation
### Installation requirements
1. Plumed version >= 2.9.0 and pytorch module linked with libtorch.lib is required.
2. CP2K installation is required and is linked to plumed.
3. For a better implementation create a separate python virtual env and install the autoencoder.
4. slurm job system is required.
### How to install
```shell
git clone https://github.com/zzkmirok/skewencoder.git
cd skewencoder
pip install [-e] .
```
### How to run
By setting up all environments correctly and making plumed + CP2k runnable under a zsh based shell, all Demo.py can run.
