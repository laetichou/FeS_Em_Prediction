#!/bin/bash

# energy minimization
gmx grompp -f minim.mdp -c your_name.gro -p your_name_gro.top -o em.tpr
gmx mdrun -v -deffnm em

# equilibration nvt
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p your_name_gro.top -o nvt.tpr
gmx mdrun -deffnm nvt

# equilibration npt
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p your_name_gro.top -o npt.tpr
gmx mdrun -deffnm npt

# production phase
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p your_name_gro.top -o md.tpr
gmx mdrun -deffnm md
