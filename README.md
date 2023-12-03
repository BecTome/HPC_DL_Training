# HPC_DL_Training
Project developed and ran in cluster from Barcelona Supercomputing Center (BSC)

## How to run the code
   
- Exercise 1: 
  -  Change the number of epochs and the values of learning rate to test in [ex1_GradientDescent/gradient_descent.py](ex1_GradientDescent/gradient_descent.py). 
  -  Run `sbatch launch_ex1_gd.sh` in cluster.
  -  Output will be generated [output/ex1_GradientDescent](output/ex1_GradientDescent).
  -  The same for adam, momentum and multioptimizer, all of them called from `launch_ex1_gd.sh`

- Exercise 2:
  - Run `sbatch launch_ex2.sh` in cluster.
  - Output will be generated in [output/ex2_MnistSingleLayer](output/ex2_MnistSingleLayer).
  