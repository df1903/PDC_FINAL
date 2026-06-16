# Active Tasks — Fase 3 (C + MPI)

**Estado: PLANIFICADA — 2026-06-15** (plan definido en
`traceability_data/2026_06_15_19-22.md`, iteración 1). Pendiente de implementación.
Fase activa: Fase 3 (C + MPI). Fases 1 (Python Baseline) y 2 (C + OpenMP) **COMPLETADAS**.

> El plan ya ejecutado de Fase 2 (C + OpenMP) se conserva como referencia histórica en
> `traceability_data/2026_06_15_14-24.md`; sus resultados en `context/state/current-phase.md`.
> Fase 3 **reutiliza `npy_io.{h,c}` sin cambios** y **copia literalmente** el cómputo de
> `scoring_openmp.c` (RNG, score, AUC, consistencia, self-test, I/O de benchmark); solo cambia
> el modelo de paralelismo (procesos MPI con memoria distribuida en vez de hilos OpenMP).

Objetivo: implementar `code/C_OpenMP_MPI/scoring_mpi.c` (Random Search con paralelismo de
**memoria distribuida**) **equivalente a las Fases 1 y 2** (mismo AUC, esquema de benchmark de 10
columnas, CLI análoga sin `--threads`), leyendo los `.npy` existentes sin modificarlos (DEC-10),
sin librerías C externas más allá de MPI (DEC-06/`stack.md`) y sin OpenMP ni CUDA. Decisiones de
estrategia formalizadas en **DEC-14**. Ejecutar las tareas en orden; cada una valida la anterior.

## Hallazgos verificados contra el repo (base del plan)

- **`scoring_mpi.c` es un placeholder** (28 LOC): `MPI_Init`/`Comm_rank`/`Comm_size`/`Finalize`
  con un `TODO` de `MPI_Scatter`/`MPI_Reduce` y un `printf`. Reemplazo completo.
- **Cómputo ya validado en Fase 2**: `splitmix64_*`, `sample_dirichlet`, `compute_P`,
  `compute_score`, `compute_auc` (empates a 0.5, RIESGO-03 resuelto), `scoring_consistency`,
  `self_test`, `load_dataset`, `read_last_time`, `append_benchmark` en `scoring_openmp.c` —
  **se copian literalmente** (sin abstracción nueva; no se modifica `scoring_openmp.c`).
- **`npy_io.{h,c}` listo** para reutilizarse sin cambios (DEC-06/10; ya diseñado pensando en
  Fase 3). `npy_load_f32`/`npy_load_i32` cargan `data/n_%d/{matrix_A,profiles,labels}.npy`.
- **`benchmark.csv` en esquema de 10 columnas** (DEC-13): `..., speedup, efficiency,
  speedup_vs_python`. Ya contiene filas de Fase 1 (Python sec/multicore) y Fase 2 (C OpenMP
  P∈{1,2,4,8}). Append-only; no tocar filas previas ni el esquema.
- **DEC-12 hace innecesario el `MPI_Scatter` de candidatos**: como `W_k = sample_dirichlet(seed
  + k)` depende **solo del índice global `k`**, cada proceso regenera localmente los `W_k` de su
  rango → cero coste de transmisión de candidatos y determinismo exacto (ver DEC-14).
- **Dataset separable** (`best_auc = 1.0`): la equivalencia por **valor de AUC** se cumple
  trivialmente para todo P, igual que en Fase 2.
- **Toolchain a verificar (RIESGO-02)**: WSL2 con OpenMPI (`mpicc`/`mpirun`); confirmar
  `mpicc --version` antes de compilar. No se requiere contenedor (gcc/WSL2 ya operativo).

## 1. Ajustar `code/C_OpenMP_MPI/Makefile` (target `mpi` linkea `npy_io.c`)

```make
scoring_mpi: scoring_mpi.c npy_io.c npy_io.h
	$(MPICC) $(CFLAGS) -o scoring_mpi scoring_mpi.c npy_io.c -lm
```

- `CFLAGS = -O2 -Wall -Wextra` ya existe; alias `make mpi` ya existe. Sin `-fopenmp` (esta fase
  no usa OpenMP). No tocar el target `scoring_openmp`.

## 2. Reescribir `code/C_OpenMP_MPI/scoring_mpi.c` (< 400 LOC, lógica separada de `main`)

Estructura (espejo de `scoring_openmp.c`, `#define` para constantes, sin VLAs):

- **Copias literales** desde `scoring_openmp.c`: RNG (tarea 3), score/AUC/consistencia (tarea 5),
  `self_test` (tarea 8), `load_dataset` (tarea 4), `read_last_time`/`append_benchmark` (tarea 10).
- **Nuevo** (lógica MPI): `MPI_Init`, partición de trabajo, `random_search_mpi`, reducción,
  timing y `main` orquestador. Estimación total ≈ **305–340 LOC** (< 400).
- Includes: reemplazar `<omp.h>` por `<mpi.h>`; conservar `<getopt.h>`, `<math.h>`, `<stdint.h>`,
  `"npy_io.h"`. Quitar `omp_*`.

## 3. Generación de candidatos W ~ Dirichlet(1,1,1) — regeneración local (DEC-14)

- Copia literal de `splitmix64_next`/`splitmix64_uniform01`/`sample_dirichlet`.
- Cada proceso, en su rango, llama `sample_dirichlet(W, seed + (uint64_t)k)` con el **`k`
  global**. **Sin `MPI_Scatter` de candidatos** (DEC-14): el conjunto total `{W_0…W_{K-1}}` es
  idéntico al de Fases 1 y 2; solo cambia **qué proceso** evalúa cada `k`. ⇒ `best_auc`
  determinista e idéntico para todo P, sin coste de transmisión de candidatos.

## 4. Carga de datos en root + difusión `MPI_Bcast` (DEC-06/10/14)

- **Solo rank 0** llama `load_dataset(n_items)` (copia literal; valida shapes, T/S∈[0,1],
  F∈{0,1,2}, `labels==[0]*5+[1]*5`). Carga **fuera del cronómetro**.
- Difusión antes de la búsqueda, en orden:
  1. `MPI_Bcast(&n_items, 1, MPI_INT, 0, MPI_COMM_WORLD)`.
  2. Ranks ≠ 0 hacen `malloc` de `A` (`10*n_items` f32), `profiles` (`n_items*3` f32),
     `labels` (10 i32).
  3. `MPI_Bcast` de `A` (`MPI_FLOAT`), `profiles` (`MPI_FLOAT`), `labels` (`MPI_INT`).
- Datos minúsculos (~710 floats); `Bcast` despreciable. Reutiliza `npy_io` sin cambios.

## 5. Distribución de trabajo + Score/AUC

- **Reparto contiguo por bloques**:
  `local_K = K / size; start = rank*local_K; end = (rank==size-1) ? K : start+local_K;`
  (el último rank absorbe el remanente si `K % size != 0`). Evaluación **independiente, sin
  comunicación durante la búsqueda**.
- Cómputo copiado literalmente: `compute_P`, `compute_score`, `compute_auc` (`+0.5·empates`,
  ec. 3), `scoring_consistency` (ec. 4). Acumular en `double`.
- Cada proceso mantiene `local_best_auc` y `local_best_k` (índice **global**; init al primer `k`
  del rango; actualizar solo con `auc > local_best` estricto → desempate consistente).

## 6. Reducción distribuida — `MPI_MAXLOC` con `MPI_DOUBLE_INT` (DEC-14)

- Estrategia elegida: cada proceso empaqueta `{double auc; int k_global;}` y
  `MPI_Reduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD)`.
- El root recibe el AUC máximo **y su `k*` global**; reconstruye `best_W =
  sample_dirichlet(seed + k*)` (1 evaluación) y recalcula `score`/consistencia para imprimir.
- **Por qué `MAXLOC` y no `MPI_MAX`**: `MPI_MAX` da el AUC pero no el `k*`, obligando a re-buscar
  el óptimo (ambiguo con muchos empates en AUC=1.0). `MAXLOC` transporta el índice y resuelve
  empates por **menor `k`** → `best_W` reproducible y fijo respecto a P (mejora menor sobre
  Fase 2, donde `best_W` variaba con P; no afecta el criterio de equivalencia).

## 7. Medición de tiempo — `MPI_Wtime()` + `MPI_Barrier`

```c
MPI_Barrier(MPI_COMM_WORLD);
double t0 = MPI_Wtime();
/* búsqueda local */
MPI_Reduce(..., MPI_MAXLOC, 0, MPI_COMM_WORLD);
double time_seconds = MPI_Wtime() - t0;   /* rank 0 registra */
```

- **`MPI_Barrier` antes de `t0`**: sincroniza el arranque de todos los procesos (que parten en
  instantes distintos por el lanzamiento de `mpirun` y el `Bcast`), de modo que `t1 − t0` mida el
  tiempo del proceso más lento (el real de la fase paralela), no el desfase de arranque. El
  `MPI_Reduce` final ya es punto de sincronización para cerrar `t1`. Excluye carga `.npy` y
  escritura de CSV (ec. 7 / `rules.md`).

## 8. Self-test

- Copia literal de `self_test()` (3 casos de `compute_auc`, incluido empate → AUC=0.875).
- Lo ejecuta **solo rank 0**; luego `MPI_Finalize`. Invocar con
  `mpirun -n 1 ./scoring_mpi --self-test` (`MPI_Init/Finalize` presentes igualmente).

## 9. CLI

- `getopt_long`: `--n-items` (50), `--k-candidates` (100000), `--seed` (42), `--self-test`.
  **Sin `--threads`**: el nº de procesos lo fija `mpirun -n P` (`MPI_Comm_size`).
- El parseo corre en todos los ranks (mismos `argv`); seguro. Ruta `data/n_%d/` con CWD = `code/`.

## 10. Registro en `results/benchmark.csv` (10 columnas, append-only, DEC-13) — solo rank 0

- Copia literal de `read_last_time` y `append_benchmark`. Adaptaciones mínimas:

| columna | valor en C MPI |
|---|---|
| `implementation` | `C MPI` |
| `workers` | `size` (`MPI_Comm_size`) |
| `best_auc`, `time_seconds` | del run (tiempo solo de búsqueda, `MPI_Wtime`) |
| `candidates_per_second` | `K / time_seconds` |
| `speedup` | `T_mpi(workers=1) / T_mpi(workers)` (baseline propio, DEC-13) |
| `efficiency` | `speedup / workers` |
| `speedup_vs_python` | `T_python_secuencial / T_mpi(workers)` |

- `speedup`: con `size==1`, baseline = esta misma corrida (`has_speedup=1`); con `size>1`,
  `read_last_time(BENCHMARK_CSV, "C MPI", n_items, k_candidates, 1, &t_mpi1)`.
- `speedup_vs_python`: `read_last_time(..., "Python secuencial", ..., -1, ...)`.
- **Solo rank 0** abre/escribe el CSV → sin condiciones de carrera. Header solo si falta;
  append-only; no sobrescribir filas previas.

## 11. Criterios de salida de Fase 3 (no avanzar a Fase 4 sin esto)

- [ ] `--self-test` de AUC en verde, incluido el caso con empate (RIESGO-03).
- [ ] `|best_auc_MPI − best_auc_PySeq| < 1e-4` (mismo K=100k, seed=42, N=50); se espera
      `1.0000 vs 1.0000`, `|ΔAUC|=0` (RF-04).
- [ ] `best_auc ∈ [0.5, 1.0]` (se espera 1.0) y consistencia (ec. 4) ≥ 0.8.
- [ ] `speedup ≥ 3×` con **P=4** (RNF-03); reportar curva P ∈ {1,2,4,8}. Si el overhead MPI lo
      impide con K=100k (RIESGO-05), aportar curva K∈{500k,1M} como evidencia de amortización.
- [ ] Filas `C MPI` (P∈{1,2,4,8}) añadidas a `results/benchmark.csv` sin perder filas previas.
- [ ] (Opcional) Sin fugas: `mpirun -n 4 valgrind --leak-check=full ./scoring_mpi --k-candidates
      1000` (se esperan bloques "still reachable" internos de OpenMPI, no del código propio).

## 12. RIESGO-05 — overhead MPI con K pequeño

- Con N=50/K=100k el cómputo P=1 es ~0.03 s (referencia Fase 2); `MPI_Init` (arranque de
  procesos, PMIx/red), `Bcast` y `Reduce` añaden overhead fijo que puede acercar/superar la
  ganancia, sobre todo en P=8. Es posible `speedup(P=4) < 3×` con K=100k.
- **Mitigación**: medir también `K ∈ {500k, 1M}` y reportar la curva speedup vs K (K mínimo
  amortizable). K=100k se mantiene como caso por defecto para comparabilidad con Fases 1–2.
  Alimenta el análisis de Amdahl (Fase 5).

## 13. Entorno y comandos

```bash
# Compilar (desde code/C_OpenMP_MPI/)
make mpi     # mpicc -O2 -Wall -Wextra -o scoring_mpi scoring_mpi.c npy_io.c -lm

# Ejecutar (CWD = code/)
cd code
mpirun -n 1 ./C_OpenMP_MPI/scoring_mpi --self-test
for P in 1 2 4 8; do mpirun -n $P ./C_OpenMP_MPI/scoring_mpi --n-items 50 --k-candidates 100000 --seed 42; done
cat results/benchmark.csv
```

- Correr `-n 1` **primero** (crea la fila baseline `workers=1` para el `speedup` de los demás P).
- Si OpenMPI rechaza P > núcleos físicos: añadir `--oversubscribe`.
- Toolchain a verificar (RIESGO-02): WSL2 + OpenMPI. Límite de 400 LOC por archivo C respetado.

## Restricciones (recordatorio)

Sin OpenMP ni CUDA en esta fase · sin nuevas dependencias más allá de MPI (ya en `stack.md`) ·
sin modificar `npy_io.{h,c}` ni `scoring_openmp.c` · sin modificar
`matrix_A.npy`/`profiles.npy`/`labels.npy` (solo lectura) · sin tocar filas previas ni el esquema
de 10 columnas de `benchmark.csv` · `scoring_mpi.c` < 400 LOC.
