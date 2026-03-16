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
            -k | --kappa= [ranges for kappas]
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
while getopts "-:s:t:k:m:h" VAL ; do
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
        sed -i "/#SBATCH/s/\(--job-name=\).*/\1$JOB_NAME/" runsbatch.sh
        sed -i "/#SBATCH/s/\(--output=\).*/\1${JOB_NAME}.\%J.txt/" runsbatch.sh
        sed -i "/^\[loxodynamics\]/,/^\[.*\]/ s/^\(kappa[[:space:]]*=[[:space:]]*\).*/\1${kappa}/" config.toml
        sed -i "/^\[loxodynamics\]/,/^\[.*\]/ s/^\(max_iter[[:space:]]*=[[:space:]]*\).*/\1${MAXITER}/" config.toml
        sed -i "/^\[simulation\]/,/^\[.*\]/ s/^\(seed[[:space:]]*=[[:space:]]*\).*/\1${SEED}/" config.toml
        cd $SCRIPT_PATH
    done
done