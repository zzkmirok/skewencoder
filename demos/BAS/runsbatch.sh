#!/bin/zsh

### Set the job to run exclusively.

#SBATCH --account=simcat
### Ask for exactly one node -> all allocated cpus must be on this one
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=12

### Ask for <1 or 2 or 4 or 6 or 8 or 12> cpus.
#SBATCH --cpus-per-task=1

### Divide the needed memory per task through the cpus-per-task,
### as slurm requests memory per cpu, not per task!
### Example:
### You need 2 GB memory per task, you have 8 cpus per task ordered
### order 2048/ <1 or 2 or 4 or 6 or 8 or 12> ->
### <2048M or 1024M or 512M or 342M or 256M or 171M> memory per task.
### M is the default and can therefore be omitted,
### but could also be K(ilo)|G(iga)|T(era).
#SBATCH --mem-per-cpu=2541M

### Name the job.
#SBATCH --job-name=k<KAPPA>_Chaba_biased

# Outputs of the job.
#SBATCH --output=k<KAPPA>_out_Chaba.%J.txt

# Wall clock limit.
#SBATCH --time=48:30:00
echo ${SLURM_JOB_ID}
if [ -n "${SLURM_JOB_ID}" ] ; then
SCRIPT_NAME=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
SCRIPT_NAME=$(realpath $0)
fi
SCRIPT_PATH=$(dirname "$SCRIPT_NAME")
echo The objective dir is $SCRIPT_PATH
cd $SCRIPT_PATH

# restore all module for local cp2k
source /rwthfs/rz/cluster/home/yy508225/PythonVENVCollections/CP2kPlumed/bin/activate
python chabaDemo.py <KAPPA>
