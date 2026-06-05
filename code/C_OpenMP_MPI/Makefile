CC      = gcc
MPICC   = mpicc
CFLAGS  = -O2 -Wall
OMP_FLAGS = -fopenmp
MPI_FLAGS =

all: scoring_openmp scoring_mpi

scoring_openmp: scoring_openmp.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o scoring_openmp scoring_openmp.c -lm

scoring_mpi: scoring_mpi.c
	$(MPICC) $(CFLAGS) -o scoring_mpi scoring_mpi.c -lm

clean:
	rm -f scoring_openmp scoring_mpi
