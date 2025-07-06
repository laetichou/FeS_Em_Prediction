#!/bin/bash

# energy minimization
gmx grompp -f minim.mdp -c your_name.gro -p your_name_gromacs.top -o em.tpr
gmx mdrun -v -deffnm em