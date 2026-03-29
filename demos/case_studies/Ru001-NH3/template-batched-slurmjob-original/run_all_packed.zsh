#!/bin/zsh
# Simple helper to submit many jobs created by `config_all.zsh`.
# Usage: ./run_all.zsh -t PREFIX    -> submits all PREFIX_* folders
#        ./run_all.zsh -t PREFIX -n -> dry-run, only print commands
OPTIND=1
CURRENT_PROGRAM=${0##*/}

# Resolve the script location and operate from there so relative paths work
SCRIPT_NAME=$(realpath $0)
SCRIPT_PATH=$(dirname "$SCRIPT_NAME")
cd "$SCRIPT_PATH"

# User-settable options
TARGET_PREFIX=""
DRY_RUN=0

# Make globs expand to empty when there is no match (zsh behavior)
setopt NULL_GLOB

function display_help {
    cat <<-EON
        usage: $CURRENT_PROGRAM -t PREFIX [-n]
        Options:
            -t | --target=   [target folder prefix]
            -n | --dry-run   [don't call sbatch, just show commands]
            -h | --help      [show this help]
EON
}

while getopts "-:t:nh" opt; do
    case $opt in
        t ) TARGET_PREFIX="$OPTARG" ;;
        n ) DRY_RUN=1 ;;
        h ) display_help ; exit 0 ;;
        - )
            case $OPTARG in
                target=*) TARGET_PREFIX="${OPTARG#*=}" ;;
                dry-run) DRY_RUN=1 ;;
                help) display_help ; exit 0 ;;
                *) echo "Unknown option: $OPTARG" >&2 ; exit 1 ;;
            esac
            ;;
        * ) echo "Unknown option" ; display_help ; exit 1 ;;
    esac
done
shift $((OPTIND -1))

if [ -z "$TARGET_PREFIX" ]; then
    echo "Error: target prefix is required"
    display_help
    exit 1
fi

found=0
# Submit all packed job scripts matching the target prefix.
# `NULL_GLOB` makes the pattern expand to nothing if there are no matches.
for job_script in "$SCRIPT_PATH/packed_job_${TARGET_PREFIX}_"*.sh; do
    if [ ! -f "$job_script" ]; then
        continue
    fi
    found=1
    echo "Submitting: $job_script"
    if [ $DRY_RUN -eq 0 ]; then
        sbatch "$job_script"
    fi
done

if [ $found -eq 0 ]; then
    echo "No scripts matching packed_job_${TARGET_PREFIX}_*.sh found in $SCRIPT_PATH"
    exit 1
fi

echo "Done."
