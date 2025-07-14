#!/bin/bash
#SBATCH --job-name="SpTrSv_Final"
#SBATCH --output="SpTrSv_Final_%j.out"
#SBATCH --error="SpTrSv_Final_%j.error"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --export=ALL
#SBATCH -t 6:00:00

source init_var.sh
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64

SOURCE_DIR=$(pwd)
echo "Start SpTrSv Final"
cd ${SOURCE_DIR}/build/example
rm -rf SpTrSv_Final_20.csv
for sparse_mat in matrix/*.mtx;
do
    echo "Processing ${sparse_mat}"
    ./SpTrSv_Final ${sparse_mat}
done

