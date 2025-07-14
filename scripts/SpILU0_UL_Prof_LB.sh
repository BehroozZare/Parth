#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name="SpILU0_Prof_LB"
#SBATCH --output="SpILU0_Prof_LB_%j.out"
#SBATCH --error="SpILU0_Prof_LB_%j.error"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --export=ALL
#SBATCH -t 12:00:00
#SBATCH --constraint=cascade


source init_env_var.sh
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
SOURCE_DIR=$(pwd)
echo "Start SpILU0 Wait Time Analysis"
cd ${SOURCE_DIR}/build/example
rm -rf SpILU0_PAPI_LB.csv
for sparse_mat in matrix/*.mtx;
do
    echo "Processing ${sparse_mat}"
    ./SpILU0_Final_PAPI_LB ${sparse_mat}
done

