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
#SBATCH --mem-per-cpu=4096M

### Name the job.
#SBATCH --job-name=SN2_mlskew_Demo

# Outputs of the job.
#SBATCH --output=out_SN2.%J.txt

# Wall clock limit.
#SBATCH --time=3:30:00
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
source /rwthfs/rz/cluster/home/yy508225/zzkenvset.sh -m cp2k


# parameter settings
n_iterations=10
alpha=0.1

state=1
states="0,1"


if [ -d "./lightning_logs" ] ; then
    echo "Deleting lightning_logs"
    rm -rf ./lightning_logs
else
    echo "No lighting_logs directory"
fi

if ls ./*.xyz 1> /dev/null 2>&1 ; then
    echo "Deleting old .xyz files"
    rm -rf ./*.xyz
else
    echo "no .xyz file"
fi
if [ -d "./results" ] ; then
    echo "Deleting old results"
    rm -rf ./results
else
    echo "No results directory"
fi

if [ -d "./unbiased" ] ; then
    echo "Deleting old unbiased"
    rm -rf ./unbiased
else
    echo "No unbiased directory"
fi


rm -f PLUMED.OUT ch3f*

cat > "./plumed.dat" << EOF
UNITS LENGTH=A TIME=0.001  #Amstroeng, hartree, fs

ene: ENERGY
d1: DISTANCE ATOMS=1,2 NOPBC
d5: DISTANCE ATOMS=1,6 NOPBC

x1: DISTANCE ATOMS=2,6 NOPBC

c1:  COMBINE ARG=d1,d5 COEFFICIENTS=0.500,-0.500 PERIODIC=NO

h1: DISTANCE ATOMS=2,3 NOPBC
h2: DISTANCE ATOMS=2,4 NOPBC
h3: DISTANCE ATOMS=2,5 NOPBC

# mat: CONTACT_MATRIX ATOMS=1,2,6 SWITCH={GAUSSIAN R_0=0.1 D_0=1.8}
# mat: CONTACT_MATRIX ATOMS=1,2,6 SWITCH={RATIONAL R_0=0.1 D_0=1.8}

# UPPER_WALLS ARG=d1 AT=+4.0 KAPPA=150.0 EXP=2 LABEL=uwall_1
# UPPER_WALLS ARG=d5 AT=+4.0 KAPPA=150.0 EXP=2 LABEL=uwall_2
UPPER_WALLS ARG=d1 AT=+4.0 KAPPA=150.0 EXP=2 LABEL=uwall_1
UPPER_WALLS ARG=d5 AT=+4.0 KAPPA=150.0 EXP=2 LABEL=uwall_2
LOWER_WALLS ARG=h1 AT=+1.7 KAPPA=150.0 EXP=2 LABEL=lwall_1
LOWER_WALLS ARG=h2 AT=+1.7 KAPPA=150.0 EXP=2 LABEL=lwall_2
LOWER_WALLS ARG=h3 AT=+1.7 KAPPA=150.0 EXP=2 LABEL=lwall_3

# dfs: DFSCLUSTERING MATRIX=mat
# scls: CLUSTER_WITHSURFACE CLUSTERS=dfs RCUT_SURF=0.0
# lambda: CLUSTER_NATOMS CLUSTERS=scls CLUSTER=1

f1: CONTACTMAP ...
    ATOMS1=1,2 SWITCH1={RATIONAL R_0=2.5}
    ATOMS2=1,6 SWITCH2={RATIONAL R_0=1.8}
...

PRINT ...
 ARG=d1,d5,c1,f1.*
 STRIDE=10
 FILE=./unbiased/COLVAR
... PRINT
EOF

mkdir -p unbiased
mkdir -p results

echo "CP2K simulation UNBIASED begins"
# srun -n 12 cp2k.popt job.inp > output.log
# srun -n 12 cp2k.popt job1.inp > output.log
cp2k.popt job1.inp > output.log
# cp2k.popt job.inp > output.log
mv ch3f-1.restart newiter.restart
rm -f ch3f*.restart
mv ch3f-pos-1.xyz iteration_unbiased-pos.xyz
rm -f PLUMED.OUT ch3f*
echo "CP2K simulation UNBIASED ends"

for i in $(seq 0 1 $n_iterations);
do
echo "///////////////////////////////////////////"
echo "*******************************************"
echo iteration "$i"
echo iteration "$i"
echo iteration "$i"
echo "*******************************************"
echo "///////////////////////////////////////////"

echo Start training

# Data processing with python
python SNDemo.py $i $alpha $state $states > temp.txt
result_line=$(grep "final output=" temp.txt)
echo $result_line
variables=$(echo "$result_line" | awk -F= '{print $2}')
echo $variables
# Extract the first element for state
state=$(echo $variables | cut -d ' ' -f 1)
# Extract the list part and remove brackets
break_flag=$(echo $variables | cut -d ' ' -f 2)
states=$(echo $variables | cut -d ' ' -f 3- | tr -d '[],')
# Add commas back between the numbers
states=$(echo $states | sed 's/ /,/g')
echo state = $state
echo states = $states
echo break = $break_flag
# enter the newline in the file for next iteration
# positions are not utilized for the simulation, may cause some problems
echo Finish training and generating plumed file

echo Start cp2k simulation at iteration $i with plumed
# srun -n 12 cp2k.popt job_restart.inp > output.log
# srun -n 12 cp2k.popt job1_restart.inp > output.log
cp2k.popt job1_restart.inp > output.log
# cp2k.popt job_restart.inp > output.log
mv ch3f-1.restart newiter.restart
rm -f ch3f*.restart
mv ch3f-pos-1.xyz iteration_$i-pos.xyz
rm -f PLUMED.OUT ch3f*
echo CP2K simulation at iteration $i with plumed ends



# Check the Python output and set the shell boolean variable
if [ "$break_flag" = "True" ]; then
  # to_break=true
  to_break=false
else
  to_break=false
fi

# Check the boolean variable
if $to_break; then
  break
  echo "Simulation Terminated successfully"
fi

done

rm -f *.OUT
