/*
 * Nivel 2 — C + OpenMP: Random Search con paralelismo de memoria compartida.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N_SAMPLES 10
#define K_CANDIDATES 100000

/* Calcula AUC aproximado por conteo de pares concordantes */
double compute_auc(double *scores, int *labels, int n) {
    int pos = 0, neg = 0, concordant = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] == 1) pos++;
        else neg++;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (labels[i] == 1 && labels[j] == 0 && scores[i] > scores[j])
                concordant++;
        }
    }
    return (pos > 0 && neg > 0) ? (double)concordant / (pos * neg) : 0.5;
}

int main(int argc, char *argv[]) {
    /* TODO: cargar A y y desde archivos .npy (usar cnpy o formato CSV) */
    printf("scoring_openmp: placeholder — implementar carga de datos\n");
    return 0;
}
