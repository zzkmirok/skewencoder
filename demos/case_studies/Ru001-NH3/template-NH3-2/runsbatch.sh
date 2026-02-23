#!/usr/bin/zsh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=test-NH3
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --account=rwth1997
#SBATCH --time=02:00:00
#SBATCH --partition=c23g
#SBATCH --output=NH3-mace.%J.txt

echo ${SLURM_JOB_ID}
if [ -n "${SLURM_JOB_ID}" ] ; then
SCRIPT_NAME=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
SCRIPT_NAME=$(realpath $0)
fi
SCRIPT_PATH=$(dirname "$SCRIPT_NAME")
echo The objective dir is $SCRIPT_PATH
cd $SCRIPT_PATH

source $WORK/RH9PYENV/PLUMED-GPU/bin/activate 

LMP_EXE="/home/rwth1997/lammps-custom-mace/bin/lmp"
python RuNH3Demo.py -custom