/*
 * Nivel 3 — CUDA: Cada hilo evalua un candidato W_k.
 * Memoria compartida cachea filas de A por bloque.
 */
#include <cuda_runtime.h>
#include <stdio.h>

#define N_SAMPLES 10
#define BLOCK_SIZE 256

__global__ void scoring_kernel(
    const float *A,       /* N_SAMPLES x n_items */
    const int   *y,       /* N_SAMPLES labels */
    const float *W_pool,  /* K x 3 candidatos */
    float       *auc_out, /* K resultados */
    int n_items, int K
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    /* TODO: calcular P = W_pool[k] aplicado a perfiles T,S,F,
             luego Score = A * P, luego AUC por conteo de pares */
    auc_out[k] = 0.5f; /* placeholder */
}

int main() {
    printf("scoring_kernel: placeholder — implementar carga y lanzamiento\n");
    return 0;
}
