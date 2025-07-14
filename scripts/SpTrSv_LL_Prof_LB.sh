#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name="SpTrSv_Prof_LB"
#SBATCH --output="SpTrSv_Prof_LB_%j.out"
#SBATCH --error="SpTrSv_Prof_LB_%j.error"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --export=ALL
#SBATCH -t 12:00:00
#SBATCH --constraint=cascade


source init_env_var.sh
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
SOURCE_DIR=$(pwd)
echo "Start SpTrSv Load Balance"
cd ${SOURCE_DIR}/build/example
rm -rf SpTrSv_LL_PAPI_LB.csv
for sparse_mat in matrix/*.mtx;
do
    echo "Processing ${sparse_mat}"
    ./SpTrSv_LL_PAPI_LB ${sparse_mat}
done

