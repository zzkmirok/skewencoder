#!/bin/bash
set -e
# parameter settings
cd "$(dirname "$0")"
echo $(pwd)
n_Files=1
nstep_plumed=20000
STRIDE=10
End_Line=$(( $nstep_plumed / $STRIDE ))
End_Line=$(( $End_Line + 2 ))
n_iterations=20

Temperature=1.0
select_func=2

x_init=-1.8 #2 -1.8 #4 -0.7
y_init=0.5 #2 0.5 #4 1.4

# parameter for Function 3
# A=2.0
# x1=0.0
# x2=5.0
# B=1.0
# y1=5.0
# C=1.5
# y2=0.0
# D=2.5
# scalar=3

A=2.0
x1=0.0
x2=20.0
B=0.25
y1=20.0
C=0.25
y2=0.0
D=2.5
scalar=4


# select function
if [[ $select_func -eq 1 ]]
then
FUNC="(y^2)*(y^2-4)+2*(x-y)^2"
elif [[ $select_func -eq 2 ]]
then
FUNC="1.34549*x^4+1.90211*x^3*y+3.92705*x^2*y^2-6.44246*x^2-1.90211*x*y^3+5.58721*x*y+1.33481*x+1.34549*y^4-5.55754*y^2+0.904586*y+19"
elif [[ $select_func -eq 3 ]]
then
FUNC="1/(1+exp(-(y-x)/$scalar))*($A*(x-$x1)^2+$B*(y-$y1)^2)+(1-1/(1+exp(-(y-x)/$scalar)))*($C*(x-$x2)^2+$D*(y-$y2)^2)"
elif [[ $select_func -eq 4 ]]
then
FUNC="0.15*(146.7-280*exp(-15*(x-1)^2+0*(x-1)*(y-0)-10*(y-0)^2)-170*exp(-1*(x-0.2)^2+0*(x-0)*(y-0.5)-10*(y-0.5)^2)-170*exp(-6.5*(x+0.5)^2+11*(x+0.5)*(y-1.5)-6.5*(y-1.5)^2)+15*exp(0.7*(x+1)^2+0.6*(x+1)*(y-1)+0.7*(y-1)^2))"
else
echo "wrong input"
fi

# if ls ./data/COLVAR/COLVAR* 1> /dev/null 2>&1 ; then
#     echo "Deleting old COLVAR files"
#     rm -rf ./data/COLVAR/COLVAR*
# else
#     echo "Directory COLVAR is empty"
# fi


# cat > "./data/skewness_vector.txt" << EOF
# EOF

cat > "./input.dat" << EOF
nstep         $nstep_plumed
tstep         0.05
temperature   $Temperature
friction      10.0
periodic      false
dimension     2
ipos          $x_init,$y_init
plumed        ./plumed.dat
EOF

cat > "./plumed.dat" << EOF
p: DISTANCE ATOMS=1,2 COMPONENTS
# Energy
ene: CUSTOM PERIODIC=NO ARG=p.x,p.y FUNC=$FUNC
potential: BIASVALUE ARG=ene

PRINT ARG=p.x,p.y FILE=./COLVAR STRIDE=$STRIDE
EOF

echo "Model Potential Test starts"
# sed -i '$ s/$/ /' data.txt # replace the newline operator with a space (btw / /)
plumed pesmd < ./input.dat


