#!/bin/bash
#SBATCH --job-name="SpICh0_Final"
#SBATCH --output="SpICh0_Final_%j.out"
#SBATCH --error="SpICh0_Final_%j.error"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --export=ALL
#SBATCH -t 6:00:00


source init_var.sh
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

SOURCE_DIR=$(pwd)
echo "Start SpICh0 Final"
cd ${SOURCE_DIR}/build/example
rm -rf SpIch0_UL_Final_20.csv
for sparse_mat in matrix/*.mtx;
do
    echo "Processing ${sparse_mat}"
    ./SpIC0_UL_Final ${sparse_mat}
done

