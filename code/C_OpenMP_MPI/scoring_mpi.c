/*
 * Nivel 2 — C + MPI: Random Search con paralelismo de memoria distribuida.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N_SAMPLES 10
#define K_CANDIDATES 100000

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* TODO: root genera candidatos W, distribuye con MPI_Scatter,
             cada proceso evalua, MPI_Reduce recoge el maximo AUC */

    if (rank == 0) {
        printf("scoring_mpi: placeholder — implementar logica distribuida\n");
    }

    MPI_Finalize();
    return 0;
}
