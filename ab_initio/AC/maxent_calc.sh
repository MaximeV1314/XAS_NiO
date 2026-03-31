#!/bin/bash
#SBATCH --job-name=MaxEnt
#SBATCH --output=stdout_maxent
#SBATCH --partition=std
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00-24:00:00

## Move data (if necessary) and launch application
## (use 'srun' for launching mpi applications)

echo "Start of the MaxEnt calculation."
python3 maxent_calc.py
echo "Program --> Done !"

exit
