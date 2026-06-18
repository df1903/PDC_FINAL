CC      = gcc
MPICC   = mpicc
CFLAGS  = -O2 -Wall -Wextra
OMP_FLAGS = -fopenmp
MPI_FLAGS =

# The MPI compiler wrappers delegate to an underlying C compiler. Conda's MPICH
# wrapper defaults to x86_64-conda-linux-gnu-cc (often not installed); system
# OpenMPI uses cc. Force both to $(CC) so the build works in either environment.
export MPICH_CC = $(CC)
export OMPI_CC  = $(CC)

.PHONY: all openmp mpi clean

all: scoring_openmp scoring_mpi

openmp: scoring_openmp

mpi: scoring_mpi

scoring_openmp: scoring_openmp.c npy_io.c npy_io.h
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o scoring_openmp scoring_openmp.c npy_io.c -lm

scoring_mpi: scoring_mpi.c npy_io.c npy_io.h
	$(MPICC) $(CFLAGS) -o scoring_mpi scoring_mpi.c npy_io.c -lm

clean:
	rm -f scoring_openmp scoring_mpi
