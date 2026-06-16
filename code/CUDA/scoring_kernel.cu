/*
 * Nivel 3 — CUDA: kernel de Random Search sobre el simplex W=(W1,W2,W3) que
 * maximiza AUC(y, Score(W)). Un hilo GPU evalua un candidato W_k (DEC-05/DEC-15).
 *
 * Equivalente por construccion a python/sequential.py, scoring_openmp.c y
 * scoring_mpi.c: el RNG SplitMix64, P=W*[T,S,F], Score=A*P y el AUC por conteo de
 * pares con empates a 0.5 son copias literales de scoring_openmp.c (Fase 2). Cada
 * hilo reconstruye W_k = sample_dirichlet(seed + k) desde su indice global k
 * (DEC-12) => mismo valor de AUC, determinista y sin transferir W_pool.
 *
 * Fuente derivada (DEC-09): este .cu se materializa con %%writefile dentro de
 * CUDA/scoring_cuda.ipynb y se compila con pycuda.compiler.SourceModule(-O2).
 * Contiene SOLO funciones __device__ + __global__ (sin main), por lo que para un
 * chequeo de sintaxis se compila con `nvcc -O2 -c scoring_kernel.cu` (solo objeto).
 *
 * Restricciones: BLOCK_SIZE=256 (DEC-05), seed=42 (DEC-03), < 400 LOC.
 */
#include <math.h>

#define N_SAMPLES 10   /* 5 sanas (y=0) + 5 enfermas (y=1) */
#define BLOCK_SIZE 256 /* DEC-05: multiplo de 32 (warp) */
#define MAX_ITEMS 256  /* cota de n_items para arrays locales/__shared__ (sin VLAs) */

/* ----------------------------------------------------------------------------
 * RNG: SplitMix64 -> Dirichlet(1,1,1) (DEC-12) — copia __device__ de scoring_openmp.c.
 * Aritmetica de 64 bits sin signo: envuelve mod 2^64 igual que el `uint64_t` de C,
 * por lo que el stream de W_k es identico al de las Fases 2/3.
 * -------------------------------------------------------------------------- */
__device__ unsigned long long splitmix64_next(unsigned long long *state) {
    unsigned long long z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Uniforme en (0, 1]: evita log(0) en la transformacion exponencial. */
__device__ double splitmix64_uniform01(unsigned long long *state) {
    unsigned long long v = splitmix64_next(state);
    return (double)(v >> 11) * (1.0 / 9007199254740992.0) + (1.0 / 18014398509481984.0);
}

/* Dirichlet(1,1,1) = 3 exponenciales(1) = -log(U) normalizadas.
 * W_k depende solo de seed_k (DEC-12): sin estado compartido entre hilos. */
__device__ void sample_dirichlet(double W[3], unsigned long long seed_k) {
    unsigned long long state = seed_k;
    double e[3], sum = 0.0;
    for (int i = 0; i < 3; i++) {
        e[i] = -log(splitmix64_uniform01(&state));
        sum += e[i];
    }
    for (int i = 0; i < 3; i++) W[i] = e[i] / sum;
}

/* ----------------------------------------------------------------------------
 * AUC por conteo de pares con empates acreditados a 0.5 (ec. 3, igual a
 * sklearn.roc_auc_score y a compute_auc de scoring_openmp.c, RIESGO-03).
 * Acumula en `double`. `labels` y `score` tienen longitud n (= N_SAMPLES).
 * -------------------------------------------------------------------------- */
__device__ double compute_auc_dev(const double *score, const int *labels, int n) {
    int pos = 0, neg = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] == 1) pos++;
        else neg++;
    }
    if (pos == 0 || neg == 0) return 0.5;

    double concordant = 0.0;
    for (int i = 0; i < n; i++) {
        if (labels[i] != 1) continue;
        for (int j = 0; j < n; j++) {
            if (labels[j] != 0) continue;
            if (score[i] > score[j]) concordant += 1.0;
            else if (score[i] == score[j]) concordant += 0.5;
        }
    }
    return concordant / ((double)pos * (double)neg);
}

/* ----------------------------------------------------------------------------
 * Kernel principal: un hilo = un candidato. `A` (N_SAMPLES x n_items, C-order) y
 * `profiles` (n_items x 3 = T,S,F) se cachean en memoria compartida por bloque
 * (carga cooperativa + __syncthreads()); los 256 hilos del bloque los reutilizan.
 * P (vector dim n_items) vive en un array local dimensionado por MAX_ITEMS (sin VLAs).
 * -------------------------------------------------------------------------- */
__global__ void scoring_kernel(const float *A,        /* N_SAMPLES x n_items */
                               const float *profiles,  /* n_items x 3 (T,S,F)  */
                               const int *labels,      /* N_SAMPLES            */
                               float *auc_out,         /* K resultados         */
                               int n_items, int K, unsigned long long seed) {
    __shared__ float sA[N_SAMPLES * MAX_ITEMS]; /* ~10 KB para n_items=50 */
    __shared__ float sProf[MAX_ITEMS * 3];      /* ~0.6 KB                */
    __shared__ int sY[N_SAMPLES];

    /* Carga cooperativa: TODOS los hilos del bloque participan antes del return. */
    for (int idx = threadIdx.x; idx < N_SAMPLES * n_items; idx += blockDim.x)
        sA[idx] = A[idx];
    for (int idx = threadIdx.x; idx < n_items * 3; idx += blockDim.x)
        sProf[idx] = profiles[idx];
    if (threadIdx.x < N_SAMPLES) sY[threadIdx.x] = labels[threadIdx.x];
    __syncthreads();

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return; /* despues de la barrera: no hay deadlock */

    /* W_k ~ Dirichlet(1,1,1) reconstruido desde el indice global k (DEC-12). */
    double W[3];
    sample_dirichlet(W, seed + (unsigned long long)k);

    /* P_i = W0*T_i + W1*S_i + W2*F_i (ec. 1). */
    double P[MAX_ITEMS];
    for (int i = 0; i < n_items; i++) {
        double T = sProf[i * 3 + 0], S = sProf[i * 3 + 1], F = sProf[i * 3 + 2];
        P[i] = W[0] * T + W[1] * S + W[2] * F;
    }

    /* Score_j = sum_i A[j,i] * P_i (ec. 2). */
    double score[N_SAMPLES];
    for (int j = 0; j < N_SAMPLES; j++) {
        double s = 0.0;
        for (int i = 0; i < n_items; i++) s += (double)sA[j * n_items + i] * P[i];
        score[j] = s;
    }

    auc_out[k] = (float)compute_auc_dev(score, sY, N_SAMPLES);
}
