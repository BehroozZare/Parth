#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name="SpTrSv_Final"
#SBATCH --output="SpTrSv_Final_%j.out"
#SBATCH --error="SpTrSv_Final_%j.error"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --export=ALL
#SBATCH -t 6:00:00
#SBATCH --constraint=cascade


source init_env_var.sh
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

SOURCE_DIR=$(pwd)
echo "Start SpTrSv Final"
cd ${SOURCE_DIR}/build/example
rm -rf SpTrSv_Final_20.csv
for sparse_mat in matrix/*.mtx;
do
    echo "Processing ${sparse_mat}"
    ./SpTrSv_Final ${sparse_mat}
done

