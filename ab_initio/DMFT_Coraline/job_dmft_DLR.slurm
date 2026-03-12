#! /bin/bash
#SBATCH --job-name=dmft_DLR
#SBATCH --account=bpn@cpu
#SBATCH --ntasks=320
#SBATCH --ntasks-per-node=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

# number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores


module purge
# module loading
module load anaconda-py3/2023.09
module load intel-oneapi-all/2023.1
module load cmake/3.25.2
module load gcc/12.2.0
module load boost/1.78.0
module load fftw/3.3.10
module load gmp/6.2.1
module load hdf5/1.12.0

# add NFFT to the PATH
export HDF5_USE_FILE_LOCKING=FALSE
source $HOME/.bashrc


# submission directory
cd ${SLURM_SUBMIT_DIR}

conda activate triqs-3.3

# code exec
srun python dmft_DLR.py
