/*
 * Nivel 2a — C + OpenMP: Random Search sobre el simplex W=(W1,W2,W3) que
 * maximiza AUC(y, Score(W)), equivalente a python/sequential.py|multicore.py.
 * Carga data/n_{n_items}/{matrix_A,profiles,labels}.npy via npy_io (DEC-06/10, solo lectura).
 * W_k ~ Dirichlet(1,1,1) via SplitMix64(seed+k) (DEC-12); best_auc determinista para cualquier P.
 * Ejecutar desde `code/`: ./C_OpenMP_MPI/scoring_openmp --n-items 50 --k-candidates 100000 --seed 42 --threads 4
 *   ./C_OpenMP_MPI/scoring_openmp --self-test
 */
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npy_io.h"

#define N_SAMPLES 10
#define DEFAULT_N_ITEMS 50
#define DEFAULT_K_CANDIDATES 100000
#define DEFAULT_SEED 42
#define BENCHMARK_CSV "results/benchmark.csv"
#define BENCHMARK_HEADER \
    "implementation,n_items,k_candidates,workers,best_auc,time_seconds," \
    "candidates_per_second,speedup,efficiency,speedup_vs_python\n"

typedef struct {
    float *A;         /* (N_SAMPLES x n_items), C-order */
    float *profiles;  /* (n_items x 3), C-order: T,S,F */
    int32_t *labels;  /* (N_SAMPLES) */
    int n_items;
} Dataset;

/* RNG: SplitMix64 -> Dirichlet(1,1,1) (DEC-12) */
static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Uniforme en (0, 1]: evita log(0) en la transformacion exponencial. */
static double splitmix64_uniform01(uint64_t *state) {
    uint64_t v = splitmix64_next(state);
    return (double)(v >> 11) * (1.0 / 9007199254740992.0) + (1.0 / 18014398509481984.0);
}

/* Dirichlet(1,1,1) = 3 exponenciales(1) = -log(U) normalizadas.
 * `W_k` depende solo de `seed_k` (DEC-12): sin estado compartido. */
static void sample_dirichlet(double W[3], uint64_t seed_k) {
    uint64_t state = seed_k;
    double e[3], sum = 0.0;
    for (int i = 0; i < 3; i++) {
        e[i] = -log(splitmix64_uniform01(&state));
        sum += e[i];
    }
    for (int i = 0; i < 3; i++) W[i] = e[i] / sum;
}

/* Score (ecs. 1-2) y AUC con empates (ec. 3, DEC-12/ISSUE-008) */
/* P_i = W1*T_i + W2*S_i + W3*F_i (ec. 1). `profiles` es (n_items x 3). */
static void compute_P(const float *profiles, int n_items, const double W[3], double *P) {
    for (int i = 0; i < n_items; i++) {
        double T = profiles[i * 3 + 0], S = profiles[i * 3 + 1], F = profiles[i * 3 + 2];
        P[i] = W[0] * T + W[1] * S + W[2] * F;
    }
}

/* Score = A @ P (ec. 2). `A` es (N_SAMPLES x n_items), `score` tiene N_SAMPLES. */
static void compute_score(const float *A, int n_items, const double *P, double score[N_SAMPLES]) {
    for (int j = 0; j < N_SAMPLES; j++) {
        double s = 0.0;
        for (int i = 0; i < n_items; i++) s += (double)A[j * n_items + i] * P[i];
        score[j] = s;
    }
}

/* AUC por conteo de pares con empates acreditados a 0.5 (ec. 3, igual a
 * sklearn.roc_auc_score): (concordantes + 0.5*empates) / (pos*neg). */
static double compute_auc(const double *score, const int32_t *labels, int n) {
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

/* Consistencia (ec. 4): theta = mediana(score); TP/5 + TN/5. */
static double scoring_consistency(const double *score, const int32_t *labels, int n) {
    double sorted[N_SAMPLES];
    memcpy(sorted, score, (size_t)n * sizeof(double));
    for (int i = 1; i < n; i++) {
        double key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key) {
            sorted[j + 1] = sorted[j];
            j--;
        }
        sorted[j + 1] = key;
    }
    double theta = (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];

    int tp = 0, tn = 0;
    for (int i = 0; i < n; i++) {
        if (score[i] > theta && labels[i] == 1) tp++;
        if (score[i] <= theta && labels[i] == 0) tn++;
    }
    return (double)tp / 5.0 + (double)tn / 5.0;
}

/* Self-test de compute_auc (RIESGO-03): valores conocidos de sklearn */
static int self_test(void) {
    struct {
        double score[4];
        int32_t labels[4];
        double expected;
        const char *desc;
    } cases[] = {
        {{1, 2, 2, 3}, {0, 0, 1, 1}, 0.875, "con empate"},
        {{1, 2, 3, 4}, {0, 0, 1, 1}, 1.0, "separable"},
        {{1, 1, 1, 1}, {0, 0, 1, 1}, 0.5, "todo empate"},
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        double auc = compute_auc(cases[i].score, cases[i].labels, 4);
        double diff = fabs(auc - cases[i].expected);
        int pass = diff < 1e-9;
        printf("[self-test] caso %zu (%s): auc=%.6f esperado=%.6f -> %s\n", i + 1, cases[i].desc,
               auc, cases[i].expected, pass ? "OK" : "FAIL");
        if (!pass) ok = 0;
    }
    return ok;
}

/* Carga y validacion de datos (espejo de validate_dataset, DEC-10) */
static Dataset load_dataset(int n_items) {
    Dataset ds;
    ds.n_items = n_items;
    char path[256];
    int rows, cols;

    snprintf(path, sizeof path, "data/n_%d/matrix_A.npy", n_items);
    ds.A = npy_load_f32(path, &rows, &cols);
    if (rows != N_SAMPLES || cols != n_items) {
        fprintf(stderr, "matrix_A invalida: shape=(%d,%d), esperado (%d,%d)\n", rows, cols, N_SAMPLES, n_items);
        exit(1);
    }

    snprintf(path, sizeof path, "data/n_%d/profiles.npy", n_items);
    ds.profiles = npy_load_f32(path, &rows, &cols);
    if (rows != n_items || cols != 3) {
        fprintf(stderr, "profiles invalido: shape=(%d,%d), esperado (%d,3)\n", rows, cols, n_items);
        exit(1);
    }

    snprintf(path, sizeof path, "data/n_%d/labels.npy", n_items);
    ds.labels = npy_load_i32(path, &rows, &cols);
    if (rows != N_SAMPLES || cols != 1) {
        fprintf(stderr, "labels invalido: shape=(%d,), esperado (%d,)\n", rows, N_SAMPLES);
        exit(1);
    }
    for (int i = 0; i < N_SAMPLES; i++) {
        if (ds.labels[i] != (i < 5 ? 0 : 1)) {
            fprintf(stderr, "labels invalido: se esperaba [0]*5 + [1]*5\n");
            exit(1);
        }
    }

    for (int i = 0; i < n_items; i++) {
        float T = ds.profiles[i * 3 + 0], S = ds.profiles[i * 3 + 1], F = ds.profiles[i * 3 + 2];
        if (T < 0.0f || T > 1.0f || S < 0.0f || S > 1.0f) {
            fprintf(stderr, "profiles invalido: T/S fuera de [0,1]\n");
            exit(1);
        }
        if (F != 0.0f && F != 1.0f && F != 2.0f) {
            fprintf(stderr, "profiles invalido: F fuera de {0,1,2}\n");
            exit(1);
        }
    }

    return ds;
}

static void free_dataset(Dataset *ds) {
    free(ds->A);
    free(ds->profiles);
    free(ds->labels);
}

/* Random Search paralelizado con OpenMP (best local por hilo + critical) */
static void random_search_openmp(const Dataset *ds, int K, uint64_t seed, int threads,
                                  double best_W[3], double *best_auc) {
    *best_auc = -1.0;
    best_W[0] = best_W[1] = best_W[2] = 0.0;

    omp_set_num_threads(threads);

#pragma omp parallel
    {
        double local_best = -1.0;
        double local_W[3] = {0.0, 0.0, 0.0};
        double *P = malloc((size_t)ds->n_items * sizeof(double));

        /* W_k depende solo de seed+k: muestreo sin seccion critica (DEC-12). */
#pragma omp for schedule(dynamic, 64) nowait
        for (int k = 0; k < K; k++) {
            double W[3], score[N_SAMPLES];
            sample_dirichlet(W, seed + (uint64_t)k);
            compute_P(ds->profiles, ds->n_items, W, P);
            compute_score(ds->A, ds->n_items, P, score);
            double auc = compute_auc(score, ds->labels, N_SAMPLES);
            if (auc > local_best) {
                local_best = auc;
                memcpy(local_W, W, sizeof W);
            }
        }
        free(P);

        /* Fusion del optimo global: best_auc determinista, best_W puede
         * variar ante empates de AUC (no afecta el criterio de equivalencia). */
#pragma omp critical
        {
            if (local_best > *best_auc) {
                *best_auc = local_best;
                memcpy(best_W, local_W, sizeof local_W);
            }
        }
    }
}

/* benchmark.csv: lectura de tiempos base y registro append-only (esquema 10 cols, DEC-13).
 * `read_last_time` retorna la ultima fila de `implementation` que coincida en
 * n_items/k_candidates (y en `workers` si `required_workers >= 0`); 1 si se encontro
 * (DEC-13: tambien lee el baseline propio P=1 de "C OpenMP", no solo "Python secuencial"). */
static int read_last_time(const char *csv_path, const char *implementation, int n_items,
                           int k_candidates, int required_workers, double *out_t) {
    FILE *f = fopen(csv_path, "r");
    if (!f) return 0;

    char line[1024];
    int found = 0;
    double last_t = 0.0;

    if (!fgets(line, sizeof line, f)) { fclose(f); return 0; }
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = '\0';
        char *fields[10];
        int nf = 0;
        char *tok = strtok(line, ",");
        while (tok != NULL && nf < 10) {
            fields[nf++] = tok;
            tok = strtok(NULL, ",");
        }
        if (nf < 6) continue;
        if (strcmp(fields[0], implementation) != 0) continue;
        if (atoi(fields[1]) != n_items || atoi(fields[2]) != k_candidates) continue;
        if (required_workers >= 0 && atoi(fields[3]) != required_workers) continue;
        last_t = atof(fields[5]);
        found = 1;
    }
    fclose(f);
    if (found) *out_t = last_t;
    return found;
}

/* Crea cabecera solo si el archivo falta/esta vacio; nunca sobrescribe filas
 * previas. speedup/efficiency y speedup_vs_python vacios sin baseline (DEC-13). */
static void append_benchmark(const char *csv_path, const char *implementation, int n_items,
                              int k_candidates, int workers, double best_auc, double time_seconds,
                              double cps, int has_speedup, double speedup, double efficiency,
                              int has_vs_python, double speedup_vs_python) {
    FILE *check = fopen(csv_path, "r");
    int need_header = 1;
    if (check) {
        char c;
        if (fread(&c, 1, 1, check) == 1) need_header = 0;
        fclose(check);
    }

    FILE *f = fopen(csv_path, "a");
    if (!f) {
        fprintf(stderr, "no se pudo abrir %s para escritura\n", csv_path);
        exit(1);
    }
    if (need_header) fputs(BENCHMARK_HEADER, f);

    char speedup_str[32] = "", efficiency_str[32] = "", vs_python_str[32] = "";
    if (has_speedup) {
        snprintf(speedup_str, sizeof speedup_str, "%.10g", speedup);
        snprintf(efficiency_str, sizeof efficiency_str, "%.10g", efficiency);
    }
    if (has_vs_python) snprintf(vs_python_str, sizeof vs_python_str, "%.10g", speedup_vs_python);

    fprintf(f, "%s,%d,%d,%d,%.10g,%.10g,%.10g,%s,%s,%s\n", implementation, n_items, k_candidates,
            workers, best_auc, time_seconds, cps, speedup_str, efficiency_str, vs_python_str);
    fclose(f);
}

/* CLI */
static void print_usage(const char *prog) {
    fprintf(stderr,
            "Uso: %s [--n-items N] [--k-candidates K] [--seed S] [--threads P] [--self-test]\n",
            prog);
}

int main(int argc, char *argv[]) {
    int n_items = DEFAULT_N_ITEMS;
    int k_candidates = DEFAULT_K_CANDIDATES;
    unsigned long seed = DEFAULT_SEED;
    int threads = omp_get_max_threads();
    int do_self_test = 0;

    static struct option long_options[] = {
        {"n-items", required_argument, 0, 'n'},
        {"k-candidates", required_argument, 0, 'k'},
        {"seed", required_argument, 0, 's'},
        {"threads", required_argument, 0, 't'},
        {"self-test", no_argument, 0, 'x'},
        {0, 0, 0, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:k:s:t:x", long_options, NULL)) != -1) {
        switch (opt) {
            case 'n': n_items = atoi(optarg); break;
            case 'k': k_candidates = atoi(optarg); break;
            case 's': seed = strtoul(optarg, NULL, 10); break;
            case 't': threads = atoi(optarg); break;
            case 'x': do_self_test = 1; break;
            default: print_usage(argv[0]); return 1;
        }
    }

    if (do_self_test) return self_test() ? 0 : 1;

    Dataset ds = load_dataset(n_items);

    double best_W[3], best_auc;
    double t0 = omp_get_wtime();
    random_search_openmp(&ds, k_candidates, seed, threads, best_W, &best_auc);
    double time_seconds = omp_get_wtime() - t0;

    double *P = malloc((size_t)ds.n_items * sizeof(double));
    double score[N_SAMPLES];
    compute_P(ds.profiles, ds.n_items, best_W, P);
    compute_score(ds.A, ds.n_items, P, score);
    double consistency = scoring_consistency(score, ds.labels, N_SAMPLES);
    free(P);

    double cps = (double)k_candidates / time_seconds;

    /* speedup/efficiency vs el propio baseline P=1 de "C OpenMP" (DEC-13); con
     * threads==1 el baseline es esta misma corrida. speedup_vs_python es la
     * comparacion explicita contra "Python secuencial", en columna separada. */
    double t_openmp1 = time_seconds, speedup = 0.0, efficiency = 0.0;
    int has_speedup = (threads == 1) ? 1
        : read_last_time(BENCHMARK_CSV, "C OpenMP", n_items, k_candidates, 1, &t_openmp1);
    if (has_speedup) {
        speedup = t_openmp1 / time_seconds;
        efficiency = speedup / threads;
    } else {
        fprintf(stderr, "[WARN] Falta fila 'C OpenMP' workers=1 en %s (n_items=%d k=%d); speedup/efficiency quedan vacios.\n",
                BENCHMARK_CSV, n_items, k_candidates);
    }

    double t_seq_py = 0.0, speedup_vs_python = 0.0;
    int has_vs_python = read_last_time(BENCHMARK_CSV, "Python secuencial", n_items, k_candidates, -1, &t_seq_py);
    if (has_vs_python) {
        speedup_vs_python = t_seq_py / time_seconds;
    } else {
        fprintf(stderr, "[WARN] Falta fila 'Python secuencial' en %s (n_items=%d k=%d); speedup_vs_python queda vacio.\n",
                BENCHMARK_CSV, n_items, k_candidates);
    }

    append_benchmark(BENCHMARK_CSV, "C OpenMP", n_items, k_candidates, threads, best_auc,
                      time_seconds, cps, has_speedup, speedup, efficiency, has_vs_python, speedup_vs_python);

    printf("Best W: [%.8f, %.8f, %.8f]\n", best_W[0], best_W[1], best_W[2]);
    printf("Best AUC: %.4f\n", best_auc);
    printf("Consistencia: %.4f\n", consistency);
    printf("Tiempo: %.4f s\n", time_seconds);
    if (has_speedup) printf("Speedup: %.4f | Efficiency: %.4f\n", speedup, efficiency);
    if (has_vs_python) printf("Speedup vs Python secuencial: %.4f\n", speedup_vs_python);

    free_dataset(&ds);
    return 0;
}
