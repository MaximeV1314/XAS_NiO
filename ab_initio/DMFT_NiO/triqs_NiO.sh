#!/bin/bash
#SBATCH --job-name=DMFTCalculation
#SBATCH --output=stdout_DMFT
#SBATCH --partition=thin
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00-04:00:00

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


## Move data (if necessary) and launch application
## (use 'srun' for launching mpi applications)

echo "Start of the DMFT calculation."
mpirun -np 64 python3 triqs_NiO.py
echo "Program --> Done !"

exit
