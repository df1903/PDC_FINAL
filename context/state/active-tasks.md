# Active Tasks — Fase 2 (C + OpenMP)

**Estado: PLANIFICADA — pendiente de implementar** (plan confirmado 2026-06-15, ver
`traceability_data/2026_06_15_14-24.md`). Fase activa según `context/state/current-phase.md`.

> Fase 1 (Python Baseline) **COMPLETADA** (2026-06-14). Su plan histórico está en
> `traceability_data/2026_06_14_16-42.md` y los resultados en
> `context/state/current-phase.md`. Este archivo pasa a contener el plan activo de Fase 2.

Objetivo: implementar `code/C_OpenMP_MPI/scoring_openmp.c` (Random Search con paralelismo de
memoria compartida) **equivalente a la Fase 1** (AUC, esquema de benchmark, CLI análoga),
leyendo los `.npy` existentes sin modificarlos (DEC-10) y sin librerías C externas (DEC-06).
Ejecutar las tareas en orden; cada una valida la anterior.

## Hallazgos verificados contra el repo (base del plan)

- **Formato `.npy` real** (inspección `xxd` de `data/n_50/`): versión `1.0`, header ASCII con
  su longitud en un campo little-endian de 2 bytes en el offset 8–9; los datos crudos empiezan
  en `10 + header_len` (128 bytes para N=50).
  - `matrix_A.npy` → `descr='<f4'`, `fortran_order=False`, shape `(10, N)`.
  - `profiles.npy` → `descr='<f4'`, `fortran_order=False`, shape `(N, 3)` (col 0=T, 1=S, 2=F).
  - `labels.npy` → `descr='<i4'`, shape `(10,)`.
- **Toolchain local OK** (mitiga RIESGO-02): `gcc 13.3` con `-fopenmp` compila; `nproc=20`
  (WSL2). No se requiere contenedor.
- **Bug del placeholder (RIESGO-03)**: `compute_auc` solo cuenta pares estrictos
  (`scores[i] > scores[j]`) y **no acredita empates**; `sklearn.roc_auc_score` da **0.5 por
  empate**. Hay que corregir la fórmula (ver tarea 4 / ISSUE-008).
- **Fase 1 es perfectamente separable** (`best_auc = 1.0`, `best_W ≈ [0.337, 0.328, 0.335]`):
  habilita el criterio de equivalencia por valor de AUC (tarea 4).

## 1. Crear `code/C_OpenMP_MPI/npy_io.h` + `npy_io.c` — parser `.npy` mínimo

- API: `float  *npy_load_f32(const char *path, int *rows, int *cols);`
       `int32_t *npy_load_i32(const char *path, int *rows, int *cols);`
       (`cols = 1` para arrays 1-D). Include guard `#ifndef NPY_IO_H`.
- Parser (~100–130 LOC): validar magic `\x93NUMPY` + versión `1 0`; leer `header_len` (2 bytes
  LE, offset 8); `strstr` para validar `descr` (`<f4`/`<i4`), exigir `fortran_order: False` y
  extraer `shape`; `fread` del bloque crudo a buffer `malloc`. Errores → `stderr` + `exit(1)`.
- Sin librerías externas (respeta DEC-06 y la regla de no añadir libs sin Makefile). **No** se
  tocan los `.npy` (DEC-10); solo lectura. Reutilizable por `scoring_mpi.c` (Fase 3).

## 2. Reescribir `code/C_OpenMP_MPI/scoring_openmp.c` (< 400 LOC, lógica separada de `main`)

Estructura (separar cómputo de `main`, `#define` para constantes, sin VLAs — usar
`malloc`/`free`):

- **Carga** (fuera del cronómetro): `npy_load_*` de `data/n_%d/{matrix_A,profiles,labels}.npy`
  con CWD = `code/`. Validar (espejo de `validate_dataset`): `A`=(10,N), `profiles`=(N,3),
  `labels`=(10,) y `labels == [0]*5+[1]*5`. Recordar C-order: `T=profiles[i*3+0]`,
  `S=...+1`, `F=...+2`.
- **RNG (tarea 3)**, **score+AUC (tarea 4)**, **paralelismo (tarea 5)**, **timing (tarea 6)**,
  **CLI (tarea 7)**, **benchmark (tarea 8)**.
- Imprimir `best_W`, `best_auc`, consistencia (ec. 4) y tiempo, como en Fase 1.
- **`--self-test`**: validación de la función AUC (tarea 4.1) sin ejecutar la búsqueda.

## 3. Generación de candidatos W ~ Dirichlet(1,1,1) — reproducible y thread-safe

- Dirichlet(1,1,1) = 3 muestras Exp(1) normalizadas; `Exp(1) = -log(U)`, `U~Uniform(0,1)`.
- **RNG SplitMix64 sembrado por candidato** con `seed + k` → 3 uniformes → 3 exponenciales →
  normalizar. `W_k` depende **solo de `k`** (no del nº de hilos ni del `schedule`):
  `best_auc` queda **determinista y reproducible** para todo P ∈ {1,2,4,8}; el estado del RNG
  vive en la pila de cada iteración → **sin sección crítica en el muestreo**.
- **No replica bit a bit** el `dirichlet` de NumPy (PCG64); no es necesario, el criterio es
  sobre el **valor de AUC**, no sobre `best_W` idéntico (ver DEC-12 y tarea 4).

## 4. Score, AUC y validación de equivalencia (RIESGO-03)

- Cómputo (ecs. 1–2, idéntico a `score_samples`), acumulando en `double`:
  `P_i = W1*T_i + W2*S_i + W3*F_i`; `Score_j = Σ_i A[j][i]*P_i`.
- **AUC corregida (igualar a sklearn)**:
  `AUC = (Σ_{y_i=1,y_j=0} [Score_i>Score_j] + 0.5·[Score_i==Score_j]) / (pos·neg)`.
  El `+0.5·empates` es el arreglo respecto al placeholder (ISSUE-008).
- Consistencia (ec. 4): `theta = mediana(Score)` (de los 10), `TP/5 + TN/5`.
- **Validación en dos capas** (replanteo de "N=3/K=100" de RIESGO-03):
  1. **Función AUC sobre scores idénticos** (`--self-test`), incluyendo un caso con **empate**,
     comparado contra valores conocidos de sklearn (p.ej. `scores=[1,2,2,3]`, `y=[0,0,1,1]`).
  2. **Resultado de la búsqueda**: `|best_auc_C − best_auc_PySeq| < 1e-4` (mismo K, seed). Se
     sostiene por la separabilidad (AUC=1.0): múltiples W alcanzan el máximo aunque el stream
     de candidatos difiera. Probar primero N/K pequeños y luego N=50/K=100k.

## 5. Paralelización OpenMP

- Patrón **best local por hilo + `#pragma omp critical` de fusión** (asocia `best_W` al máximo
  global de forma robusta; cumple la intención de `architecture.md`/`coding-standards.md`):

```c
#pragma omp parallel
{
    double local_best = -1.0, local_W[3];
    #pragma omp for schedule(dynamic, 64) nowait   /* reparte K candidatos */
    for (int k = 0; k < K; k++) {
        double W[3]; sample_dirichlet(W, seed + k);
        double auc = compute_auc(A, profiles, W, y, N);
        if (auc > local_best) { local_best = auc; memcpy(local_W, W, sizeof W); }
    }
    #pragma omp critical                            /* fusiona el óptimo global */
    { if (local_best > best_auc) { best_auc = local_best; memcpy(best_W, local_W, sizeof best_W); } }
}
```

- `omp_set_num_threads(threads)` según CLI. `best_auc` determinista; `best_W` puede variar ante
  empates de AUC (no afecta el criterio). Documentar cada `#pragma` con un comentario de línea.

## 6. Medición de tiempo

- `omp_get_wtime()` **solo alrededor del bloque de búsqueda** (tarea 5), excluyendo carga
  `.npy`, validación y escritura de CSV (ec. 7 / `rules.md`, igual que `perf_counter` en Fase 1).

## 7. CLI

- `getopt_long`: `--n-items` (50), `--k-candidates` (100000), `--seed` (42), `--threads`
  (default `omp_get_max_threads()`), `--self-test`. Barrido P ∈ {1,2,4,8} (`phases.md`).
- Ruta de datos `data/n_%d/` con **CWD = `code/`** (igual que Fase 1 y `run_all.sh`).

## 8. Registro en `results/benchmark.csv` (esquema de 9 columnas, append-only)

| columna | valor en C |
|---|---|
| `implementation` | `C OpenMP` (estilo "Python multicore", DEC-12) |
| `n_items`, `k_candidates`, `workers` | de CLI (`workers` = `threads`) |
| `best_auc`, `time_seconds` | del run (tiempo solo de búsqueda) |
| `candidates_per_second` | `K / time_seconds` |
| `speedup` | `t_seq / time_seconds` |
| `efficiency` | `speedup / threads` |

- `t_seq`: leer en C la última fila `Python secuencial` que coincida en `n_items` y
  `k_candidates` (mini-lector CSV, espejo de `read_sequential_time`). Si no existe → advertir y
  dejar `speedup`/`efficiency` vacíos (como `multicore.py`).
- Append-only: crear cabecera solo si el archivo falta/está vacío; **no** sobrescribir filas.

## 9. Criterios de salida de Fase 2 (no avanzar a Fase 3 sin esto)

- [ ] `--self-test` de AUC en verde, incluido el caso con empate (RIESGO-03).
- [ ] `|best_auc_C − best_auc_PySeq| < 1e-4` (mismo K=100k, seed=42, N=50).
- [ ] `best_auc ∈ [0.5, 1.0]` (se espera 1.0) y consistencia (ec. 4) ≥ 0.8.
- [ ] `speedup ≥ 3×` con **P=4** (RNF-03); reportar curva P ∈ {1,2,4,8}.
- [ ] Filas `C OpenMP` añadidas a `results/benchmark.csv` sin perder filas previas.
- [ ] Sin fugas (`valgrind --leak-check=full`).

## 10. Entorno y comandos

```bash
# Compilar (desde code/C_OpenMP_MPI/)
make openmp     # gcc -O2 -Wall -Wextra -fopenmp scoring_openmp.c npy_io.c -o scoring_openmp -lm

# Ejecutar (CWD = code/)
cd code
./C_OpenMP_MPI/scoring_openmp --self-test
for P in 1 2 4 8; do ./C_OpenMP_MPI/scoring_openmp --n-items 50 --k-candidates 100000 --threads $P --seed 42; done
valgrind --leak-check=full ./C_OpenMP_MPI/scoring_openmp --k-candidates 1000
cat results/benchmark.csv
```

- Toolchain verificado localmente (gcc 13.3 + OpenMP, WSL2) → RIESGO-02 cubierto sin contenedor.
- Ajustar `Makefile`: añadir `npy_io.c` al enlace de `openmp`, `-Wextra` a `CFLAGS`; dejar el
  target `mpi` intacto (Fase 3). Límite de 400 LOC por archivo C respetado (parser separado).

## Restricciones (recordatorio)

Sin MPI/CUDA en esta fase · sin nuevas dependencias Python · sin cambiar firmas canónicas ni el
esquema de `benchmark.csv` de Fase 1 · sin modificar `matrix_A.npy`/`profiles.npy`/`labels.npy`
(solo lectura) · sin librerías C externas (parser propio).
