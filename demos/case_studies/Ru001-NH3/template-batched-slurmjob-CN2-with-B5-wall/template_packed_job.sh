#!/usr/bin/zsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --account=p0024037
#SBATCH --time=5:00:00 
#SBATCH --partition=c23g
#SBATCH --job-name=2NH3_CN2_MH1_D3
#SBATCH --output=2NH3_CN2_MH1_D3.%J.txt

export BASF_RUN_PREFIX="2NH3_CN2_MH1_D3"
export PYTHON_INPUT_SCRIPT="RuNH3Demo-batch.py"
export PYTHON_INPUT_CONFIG="-custom"

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
export OMPI_MCA_orte_tmpdir_base=/dev/shm
export OMPI_MCA_btl="^smcuda"

target_folders=<target_folders>
total_tasks=${#target_folders[@]}

echo "Found $total_tasks folders to process."
count=0
for folder in "${target_folders[@]}"; do
    ((count++))

    echo "[$count/$total_tasks] Starting task in $folder"
    
    (
        cd "$folder" || exit
        python "$PYTHON_INPUT_SCRIPT" "$PYTHON_INPUT_CONFIG" > "SimuLog.txt" 2>&1
    ) &

done

echo "All $total_tasks simulations are started."
wait
echo "All tasks in this batch are done!"
unset BASF_RUN_PREFIX
unset PYTHON_INPUT_SCRIPT
unset PYTHON_INPUT_CONFIG