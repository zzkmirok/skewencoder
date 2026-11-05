#!/usr/bin/zsh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=test_300_42_sd
#SBATCH --cpus-per-task=16
#SBATCH --account=rwth1997
#SBATCH --time=02:00:00
#SBATCH --partition=c23ms
#SBATCH --output=test_300_42_sd.%J.txt

echo ${SLURM_JOB_ID}
if [ -n "${SLURM_JOB_ID}" ] ; then
SCRIPT_NAME=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
SCRIPT_NAME=$(realpath $0)
fi
SCRIPT_PATH=$(dirname "$SCRIPT_NAME")
echo The objective dir is $SCRIPT_PATH
cd $SCRIPT_PATH

# PLUMED (metatomic, with libtorch) first, then DFTB+ built against it
source ~/load_plumed-metatomic.zsh
module use /home/noco0020/custom-modules/noco
module load DFTB+/25.1-PLUMED-2.10.2-metatomic-0.1.13-CUDA-12.8.0
# Python 3.13 venv matching these modules: skewencoder (editable), torch, mlcolvar, pycv
source /rwthfs/rz/cluster/hpcwork/noco0020/ZZh/chemtrayzer-zzk/bin/activate

# DFTB+ runs with OpenMP threads (UseOmpThreads = Yes in the templates), one MPI rank
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

python PhosgeneDemo.py
