#!/bin/zsh
OPTIND=1
CURRENT_PROGRAM=${0##*/}
ORIGIN_PWD=$PWD
MY_SEQUENCE=""

function validate_kappa_seq {
    local raw_input="$1"
    # set desemble $raw_input to $1 $2 $3...
    set -- $raw_input
    
    if [ $# -ne 3 ]; then
        echo "Error: 3 params for seq (start step end)。"
        echo "The number of your input: $# : '$raw_input'"
        return 1
    fi

    for arg in "$@"; do
        if ! [[ "$arg" =~ ^-?[0-9]+$ ]]; then
            echo "Error: param '$arg' not a number"
            return 1
        fi
    done
    return 0
}

function Usage_Exit {
    echo "$0 -s [which templete to copy] -t [target folder prefix] -k [kappa sequence]"
}

# [ "${(%):-%N}" ] && MODULE="${2}" || MODULE="${1}"

function Display_help {
    cat <<-EON
        usage: $CURRENT_PROGRAM (options)
        Options:
            -s | --source= [your targeted template]
            -t | --target= [target folder prefix]
            -m | --maxiter= [max number of iter, default 20]
            -k | --kappa= [the name of venv for installing CTY]
EON
}

SCRIPT_NAME=$(realpath $0)
SCRIPT_PATH=$(dirname "$SCRIPT_NAME")
echo $SCRIPT_PATH
cd $SCRIPT_PATH

TEMPLATE_PATH=""
KAPPA_SEQ=""
TARGET_PREFIX=""
MAXITER="20"
PYTHON_INPUT_CONFIG="normal"
PYTHON_INPUT_SCRIPT="RuNH3Demo-batch.py"
BATCH_SIZE="24"
while getopts "-:s:t:k:m:v:i:b:h" VAL ; do
    case $VAL in
        s )
            TEMPLATE_PATH="$OPTARG"  # Assign the next argument as the value of p
           ;;
        k )
            RAW_VAL="$OPTARG"
    
            temp_array=( $RAW_VAL )
            CLEAN_SEQ="${temp_array[@]:0:3}"
            validate_kappa_seq "$CLEAN_SEQ" || { 
                [[ "$0" != "$BASH_SOURCE" ]] && return 1 || exit 1
            }
            KAPPA_SEQ="$CLEAN_SEQ" 
            ;;
        t )
            TARGET_PREFIX="$OPTARG"  # Assign the next argument as the value of p
            ;;
        m )
            MAXITER="$OPTARG"
            ;;
        v )
            PYTHON_INPUT_CONFIG="$OPTARG"
            ;;
        i )
            PYTHON_INPUT_SCRIPT="$OPTARG"
            ;;
        b )
            BATCH_SIZE="$OPTARG"
            ;;
        h )
            Display_help
            [[ "$0" != "$BASH_SOURCE" ]] && return 0 || exit 0
            ;;
# ---------------------------------------------------------------------------------------------------
        -)
            case $OPTARG in
                target=*)
                    TARGET_PREFIX="${OPTARG#*=}"
                    ;;
                kappa=*)
                    RAW_VAL="${OPTARG#*=}"
                    temp_array=( $RAW_VAL )
                    CLEAN_SEQ="${temp_array[@]:0:3}"
                    validate_kappa_seq "$CLEAN_SEQ" || { 
                        [[ "$0" != "$BASH_SOURCE" ]] && return 1 || exit 1
                    }
                    KAPPA_SEQ="$CLEAN_SEQ" 
                    ;;
                source=*)
                    TEMPLATE_PATH="${OPTARG#*=}"  
                    ;;
                maxiter=*)
                    MAXITER="${OPTARG#*=}"  
                    ;;
                python-config=*)
                    PYTHON_INPUT_CONFIG="${OPTARG#*=}"  
                    ;;
                python-input=*)
                    PYTHON_INPUT_SCRIPT="${OPTARG#*=}"  
                    ;;
                batch-size=*)
                    BATCH_SIZE="${OPTARG#*=}"  
                    ;;
                help    )
                    Display_help
                    [[ "$0" != "$BASH_SOURCE" ]] && return 0 || exit 0
                    ;;
                *       )
                    echo "Unknown option: $OPTARG" >&2
                    [[ "$0" != "$BASH_SOURCE" ]] && return 1 || exit 1
                ;;
            esac
            ;;
# ---------------------------------------------------------------------------------------------------
        : )
            echo "No default setting Parameters must be provided"
            ;;
        * )
            echo "Unknown option: $OPTARG" >&2
            [[ "$0" != "$BASH_SOURCE" ]] && return 1 || exit 1
            ;;
    esac
done
# We add a `shift $((OPTIND -1))` to remove all the option-related arguments from further consideration
shift $((OPTIND -1))

# declare -i SHIFT=600
# 1. Check if path is absolute (starts with /)
if [[ "$TEMPLATE_PATH" != /* ]]; then
    echo "Relative path for template detected. Converting to absolute..."
    TEMPLATE_PATH=$(realpath "${ORIGIN_PWD}/${TEMPLATE_PATH}")
fi

# 2. Check if the directory exists
if [ ! -d "$TEMPLATE_PATH" ]; then

    echo "Error: Could not find path $TEMPLATE_PATH" >&2
    [[ "$0" != "$BASH_SOURCE" ]] && return 1 || exit 1
fi


# sed -i "/^export\ BASF_RUN_PREFIX/s/\(BASF_RUN_PREFIX=\).*/\1\"${TARGET_PREFIX}\"/" packed.sh
sed -i "/^export\ PYTHON_INPUT_SCRIPT/s/\(PYTHON_INPUT_SCRIPT=\).*/\1\"${PYTHON_INPUT_SCRIPT}\"/" template_packed_job.sh
sed -i "/^export\ PYTHON_INPUT_CONFIG/s/\(PYTHON_INPUT_CONFIG=\).*/\1\"-${PYTHON_INPUT_CONFIG}\"/" template_packed_job.sh
sed -i "/#SBATCH/s/\(--job-name=\).*/\1${TARGET_PREFIX}/" template_packed_job.sh
sed -i "/#SBATCH/s/\(--output=\).*/\1${TARGET_PREFIX}.\%J.txt/" template_packed_job.sh
_TOTAL_NODES="24"
_LAMMPS_SLURM_N_CPU_PER_TASK=$(( $_TOTAL_NODES / $BATCH_SIZE ))
sed -i "/#SBATCH/s/\(--ntasks-per-node=\).*/\1${BATCH_SIZE}/" template_packed_job.sh
sed -i "/#SBATCH/s/\(--cpus-per-task=\).*/\1${_LAMMPS_SLURM_N_CPU_PER_TASK}/" template_packed_job.sh
sed -i "/^\[simulation\]/,/^\[.*\]/ s/^\(n_cpus_per_lammps_task[[:space:]]*=[[:space:]]*\).*/\1${_LAMMPS_SLURM_N_CPU_PER_TASK}/" ${TEMPLATE_PATH}/config.toml

myarray=$(seq $KAPPA_SEQ)
echo start to copy $TEMPLATE_PATH
# for kappa in $(seq 250 50 600); do
for kappa in ${myarray[@]}; do
    SEEDS=$(shuf -i 100000-999999 -n 5)
    for SEED in $SEEDS
    do
        JOB_NAME=${TARGET_PREFIX}_${kappa}_${SEED}
        echo "creating folder ${SCRIPT_PATH}/${JOB_NAME}"
        cp -r "$TEMPLATE_PATH" "$JOB_NAME"
        cd "${SCRIPT_PATH}/${JOB_NAME}"
        sed -i "/^\[loxodynamics\]/,/^\[.*\]/ s/^\(kappa[[:space:]]*=[[:space:]]*\).*/\1${kappa}/" config.toml
        sed -i "/^\[loxodynamics\]/,/^\[.*\]/ s/^\(max_iter[[:space:]]*=[[:space:]]*\).*/\1${MAXITER}/" config.toml
        sed -i "/^\[simulation\]/,/^\[.*\]/ s/^\(seed[[:space:]]*=[[:space:]]*\).*/\1${SEED}/" config.toml
        cd $SCRIPT_PATH
    done
done

target_folders=($(find "$PWD" -maxdepth 1 -type d -name "${TARGET_PREFIX}_*"))
total_tasks=${#target_folders[@]}

echo "Found $total_tasks folders to process."

batch_size=$BATCH_SIZE
count=0

# build batches of folders and write a packed job for each full batch
current_batch=()
for folder in "${target_folders[@]}"; do
    ((count++))
    current_batch+=("$folder")

    if [[ $((count % batch_size)) -eq 0 ]]; then
        cp ./template_packed_job.sh ./packed_job_${TARGET_PREFIX}_${count}.sh
        batch_items=""
        for f in "${current_batch[@]}"; do
            batch_items+="\"$f\" "
        done
        batch_items="${batch_items% }"
        sed -i "s|^target_folders=.*|target_folders=(${batch_items})|" "./packed_job_${TARGET_PREFIX}_${count}.sh"
        sed -i "/#SBATCH/s/\(--job-name=\).*/\1${TARGET_PREFIX}_${count}/" "./packed_job_${TARGET_PREFIX}_${count}.sh"
        sed -i "/#SBATCH/s/\(--output=\).*/\1${TARGET_PREFIX}_${count}.\%J.txt/" "./packed_job_${TARGET_PREFIX}_${count}.sh"


        current_batch=()
    fi
done

# handle remaining folders that don't make a full batch
if [ ${#current_batch[@]} -gt 0 ]; then
    cp ./template_packed_job.sh ./packed_job_${TARGET_PREFIX}_${count}.sh
    batch_items=""
    for f in "${current_batch[@]}"; do
        batch_items+="\"$f\" "
    done
    batch_items="${batch_items% }"
    sed -i "s|^target_folders=.*|target_folders=(${batch_items})|" "./packed_job_${TARGET_PREFIX}_${count}.sh"
    sed -i "/#SBATCH/s/\(--job-name=\).*/\1${TARGET_PREFIX}_${count}/" "./packed_job_${TARGET_PREFIX}_${count}.sh"
    sed -i "/#SBATCH/s/\(--output=\).*/\1${TARGET_PREFIX}_${count}.\%J.txt/" "./packed_job_${TARGET_PREFIX}_${count}.sh"
fi

# done batching

echo "All $total_tasks simulations are finished."
unset BASF_RUN_PREFIX
unset PYTHON_INPUT_SCRIPT
unset PYTHON_INPUT_CONFIG