#!/bin/bash

export num_threads=20
export MKL_NUM_THREADS=$num_threads
export OMP_NUM_THREADS=$num_threads
export VECLIB_MAXIMUM_THREADS=$num_threads
export BASE_ADDRESS=/Users/behroozzare/Desktop/Graphics
export PROG_PATH=$BASE_ADDRESS/ParthSolverDev/cmake-build-release/demo/PARTH_SOLVER_PatchRemeshDemo

# Array of Order values to test
orders=("METIS" "AMD")

# Array of Parth values to test
parth_values=(0 1)

# Array of simulation names
# simulations=('nefertiti' 'wingnut' 'parsnip' 'cube' 'goathead' 'brucewick' 'sword' 'violin' 'koala' 'tower' 'sphere' 'boot' 'penguin' 'springer' 'cow' 'strawberry' 'stuffedtoy' 'cheese' 'armadillo' 'falconstatue' 'human' 'cat' 'hammer' 'mountain' 'plane' 'skull' 'pizza' 'boat' 'scorpion' 'demosthenes' 'spot' 'hand' 'fish' 'bunny' 'torus' 'lionstatue' 'mushroom')
simulations=('koala' 'tower' 'boot')
# Loop through each simulation
for sim in "${simulations[@]}"
do
  # Loop through each Parth value
  for parth in "${parth_values[@]}"
  do
    # Loop through each Order value
    for order in "${orders[@]}"
    do
      $PROG_PATH --SimName=$BASE_ADDRESS/PATCH_ACCELERATE/csv/${sim}_numThreads_20_SolverType_MKL_Parth_${parth}_Order_${order} --SolverType=ACCELERATE --numThreads=$num_threads --Parth=$parth --Order=$order --input=$BASE_ADDRESS/meshes/objects/${sim}/${sim}.obj
    done
  done
done
