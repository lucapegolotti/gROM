
time(mpirun -np 2 -ppn 2 -genv I_MPI_PIN_DOMAIN=socket -genv OMP_NUM_THREADS=40 python training.py )

