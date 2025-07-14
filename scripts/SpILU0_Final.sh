#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name="SpILU0_Final"
#SBATCH --output="SpILU0_Final_%j.out"
#SBATCH --error="SpILU0_Final_%j.error"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --export=ALL
#SBATCH -t 6:00:00
#SBATCH --constraint=cascade


source init_env_var.sh

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
SOURCE_DIR=$(pwd)
echo "Start SpICh0 Final"
cd ${SOURCE_DIR}/build/example
rm -rf SpILU0_UL_Final_20.csv
for sparse_mat in matrix/*.mtx;
do
    echo "Processing ${sparse_mat}"
    ./SpILU0_Final ${sparse_mat}
done

