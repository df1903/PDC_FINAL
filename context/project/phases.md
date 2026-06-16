# Phases

## Fase 1 — Python Baseline `[COMPLETADA — 2026-06-14]`

Objetivo: implementación de referencia correcta y validada según el modelo de perfiles (DEC-07), con CLI, medición de tiempo (excluyendo carga de datos) y registro automático en `results/benchmark.csv`.

Plan técnico detallado y tareas concretas: ver `context/state/active-tasks.md`. Resumen ordenado:

- [x] **DEC-11**: actualizar `code/data/generate_data.py` para inyectar señal diferencial en las filas enfermas de `A` y regenerar `data/n_50/` y `data/n_100/` (`seed=42`).
- [x] Crear `code/python/common.py` con funciones compartidas entre `sequential.py` y `multicore.py`: `load_dataset`, `validate_dataset`, `sample_candidates`, `score_samples`, `evaluate_candidates`, `scoring_consistency`, `compute_metrics`, `append_benchmark`, `read_sequential_time`.
- [x] Reescribir `code/python/sequential.py`: `random_search(A, profiles, y, K=100_000, seed=42)`, CLI (`--n-items`, `--k-candidates`, `--seed`), timing con `perf_counter` (excluye carga), registro en benchmark.
- [x] Reescribir `code/python/multicore.py`: `random_search_multicore(A, profiles, y, K=100_000, seed=42, workers=cpu_count())`, candidatos generados en el proceso principal y repartidos con `np.array_split` + `multiprocessing.Pool`, CLI (`--n-items`, `--k-candidates`, `--workers`, `--seed`).
- [x] Migrar `code/results/benchmark.csv` al esquema: `implementation,n_items,k_candidates,workers,best_auc,time_seconds,candidates_per_second,speedup,efficiency` (append-only; crear si no existe; no sobrescribir filas previas).
- [x] Crear pruebas (`code/python/tests/test_baseline.py`): equivalencia AUC secuencial↔multicore (<1e-4, RF-04), AUC∈[0.5,1], consistencia≥0.8, caso chico N=3/K=100 (RIESGO-03). 5 tests, todos OK.
- [x] Validar AUC ∈ [0.5, 1.0] y consistencia ≥ 0.8 con el dataset regenerado (DEC-11). Resultado: AUC=1.0000, consistencia=2.0000.
- [x] Medir T_secuencial como baseline para speedup (S_base = 1.00). Resultado: 84.3913 s (K=100000, n_items=50).
- [x] Ejecutar multicore y verificar speedup ≥ 1.5× (RNF-03); calcular `efficiency = speedup / workers`. Resultado: speedup=3.1698 (workers=4), efficiency=0.7924.
- [x] Registrar ambas filas en `results/benchmark.csv`.

Detalle completo de resultados: `context/state/current-phase.md` y `traceability_data/2026_06_14_17-18.md`.

## Fase 2 — C + OpenMP `[COMPLETADA — 2026-06-15]`

Objetivo: paralelismo de memoria compartida sobre CPU.

Plan técnico detallado y tareas concretas: ver `context/state/active-tasks.md` (definido
2026-06-15, `traceability_data/2026_06_15_14-24.md`). Resumen ordenado:

- [x] Crear `code/C_OpenMP_MPI/npy_io.{h,c}`: parser `.npy` v1.0 mínimo (sin libs externas,
      DEC-06), solo lectura de `matrix_A.npy`/`profiles.npy`/`labels.npy` (DEC-10).
- [x] Implementar carga + validación de datos en `scoring_openmp.c` (espejo de `validate_dataset`).
- [x] RNG SplitMix64 por candidato (`seed+k`) → Dirichlet(1,1,1) reproducible y thread-safe.
- [x] Score `A·P` y **AUC con manejo de empates** (`+0.5·empates`, RIESGO-03/ISSUE-008).
- [x] Random Search con OpenMP (best local por hilo + `#pragma omp critical`).
- [x] CLI (`--n-items/--k-candidates/--seed/--threads/--self-test`) y registro `C OpenMP` en
      `results/benchmark.csv` (9 columnas, append-only).
- [x] Validar equivalencia AUC con Fase 1 (`--self-test` + `|ΔAUC| < 1e-4` vs Python secuencial).
- [x] Medir speedup vs Python secuencial con P ∈ {1, 2, 4, 8} (objetivo `≥ 3×` con P=4).

Resultados (seed=42, n_items=50, K=100000, desde `code/`): `best_auc=1.0000` (=Python
secuencial, `|ΔAUC|=0`), consistencia=2.0000. Speedup vs Python secuencial: P=1 → 2386.10×,
P=2 → 5555.14×, P=4 → 10411.83× (≥3× ✓), P=8 → 14813.67×. `--self-test` de `compute_auc` OK
(3/3, incluido caso con empate=0.875). `valgrind --leak-check=full`: 0 bytes definitely/indirectly
lost (solo "possibly lost"/"still reachable" internos de `libgomp`, no atribuibles al código
propio). Detalle completo: `context/state/current-phase.md` y
`traceability_data/2026_06_15_14-36.md`.

## Fase 3 — C + MPI `[COMPLETADA — 2026-06-15]`

Objetivo: paralelismo de memoria distribuida sobre CPU.

Plan técnico detallado: `context/state/active-tasks.md` (definido 2026-06-15,
`traceability_data/2026_06_15_19-22.md`) y `context/project/decisions.md#DEC-14`.
Implementación: `traceability_data/2026_06_15_19-32.md`.

- [x] Ajustar `code/C_OpenMP_MPI/Makefile`: target `scoring_mpi` linkea `npy_io.c`.
- [x] Reescribir `scoring_mpi.c` (389 LOC, < 400): copias literales de RNG SplitMix64, score/AUC
      con empates, consistencia, `self_test`, `load_dataset`, `read_last_time`/`append_benchmark`.
- [x] Carga solo en root + difusión `MPI_Bcast` de `n_items`/`A`/`profiles`/`labels` (DEC-14).
- [x] Candidatos por regeneración local `sample_dirichlet(seed+k)` sin `MPI_Scatter` (DEC-14).
- [x] Distribución contigua por bloques; reducción `MPI_MAXLOC`/`MPI_DOUBLE_INT`.
- [x] Timing `MPI_Wtime` + `MPI_Barrier`; CLI sin `--threads`; registro `C MPI` (10 cols).
- [x] `--self-test` 3/3 OK (incluido empate); `|ΔAUC| = 0 < 1e-4` vs Python secuencial.
- [x] Speedup P=4: 3.65× ≥ 3× ✓ (RNF-03). Barrido P∈{1,2,4,8} completo.

Resultados (seed=42, n_items=50, K=100000, desde `code/`, OpenMPI 4.1.6, WSL2):

| P | best_auc | time_seconds | candidates_per_second | speedup | efficiency | speedup_vs_python |
|:-:|----------|--------------|------------------------|---------|------------|-------------------|
| 1 | 1.0000   | 0.039927     | 2,504,562              | 1.0000  | 1.0000     | 2411.99×          |
| 2 | 1.0000   | 0.022423     | 4,459,637              | 1.7806  | 0.8903     | 4294.80×          |
| 4 | 1.0000   | 0.010940     | 9,140,351              | 3.6495  | 0.9124     | 8802.50×          |
| 8 | 1.0000   | 0.007415     | 13,485,991             | 5.3846  | 0.6731     | 12987.52×         |

- `best_W = [0.08767833, 0.53756110, 0.37476056]` — idéntico para todo P (MAXLOC determinista).
- Consistencia = 2.0000 en todos los casos (≥ 0.8 ✓).
- `mpicc -O2 -Wall -Wextra`: sin warnings (389 LOC, dentro del límite).
- Detalle completo: `context/state/current-phase.md` y `traceability_data/2026_06_15_19-32.md`.

## Fase 4 — CUDA (notebook en Google Colab) `[PLANIFICADA — 2026-06-16]`

Objetivo: aceleración GPU masiva en `CUDA/scoring_cuda.ipynb`, ejecutado en Colab.

Plan técnico detallado: `context/state/active-tasks.md` (definido 2026-06-16,
`traceability_data/2026_06_16_00-43.md`) y `context/project/decisions.md#DEC-15`. **La ejecución
ocurre en Google Colab** (runtime GPU); el plan se deja planteado para correrse después (no es
ejecutable en el entorno local, RIESGO-01). Resumen ordenado:

- [ ] Crear `scoring_cuda.ipynb` con celdas: setup/GPU → carga `.npy` → `%%writefile
      scoring_kernel.cu` → compilación `SourceModule(-O2)` → orquestación PyCUDA → reducción →
      timing → self-test → validación → registro CSV (DEC-15, tarea 1 de active-tasks).
- [ ] RNG SplitMix64 `__device__` sembrado con `seed+k` (copia de `scoring_openmp.c`); cada hilo
      reconstruye `W_k` desde el índice global — sin transferir `W_pool` (DEC-12/DEC-15).
- [ ] Kernel `scoring_kernel` (un hilo por candidato): `__shared__` cachea `A` y `profiles` por
      bloque (`BLOCK_SIZE=256`, DEC-05); calcula `P=W·[T,S,F]`, `Score=A·P` y AUC con empates +0.5.
- [ ] Reducción con `np.argmax` en host (no kernel de reducción) + reconstrucción de `best_W`
      (DEC-15, justificación).
- [ ] Compilación con `%%writefile` + `pycuda.compiler.SourceModule(-O2)`; `nvcc -O2` como sanity.
- [ ] Orquestación PyCUDA: H2D una vez, lanzamiento grid `ceil(K/256)`, D2H `auc_out`; `CUDA_CHECK`.
- [ ] Timing que excluye carga: `cudaEvent_t` (kernel) + `perf_counter`+`Context.synchronize()`
      (búsqueda de W*).
- [ ] Self-test: empate → 0.875 y caso `n_items=3`/K=100 vs mirror SplitMix64 (`|ΔAUC|≈0`).
- [ ] Subir los `.npy` de `code/data/n_50/` al runtime de Colab (solo lectura, DEC-10).
- [ ] Validar `AUC∈[0.5,1]`, `|ΔAUC|<1e-4` vs Python secuencial, consistencia ≥ 0.8.
- [ ] Registrar fila `CUDA` en `results/benchmark.csv` (10 col, DEC-13): `workers=1`, `speedup=1.0`,
      `efficiency=1.0`, `speedup_vs_python = T_python_secuencial / T_cuda`.
- [ ] Medir `speedup_vs_python ≥ 5×` (RNF-03) y registrar el modelo de GPU usado (RIESGO-06).

## Fase 5 — Benchmarking `[PENDIENTE]`

Objetivo: consolidar métricas y generar reportes.

- [ ] Implementar `run_all.sh`.
- [ ] Generar `results/benchmark.csv` con todas las métricas.
- [ ] Generar gráficas PNG de Speedup y Eficiencia vs P.
- [ ] Calcular f empírico para Ley de Amdahl.

## Fase 6 — Documentación `[PENDIENTE]`

Objetivo: informe técnico final.

- [ ] Redactar informe con estrategias de sincronización por nivel.
- [ ] Incluir análisis de gestión de memoria.
- [ ] Incluir gráficas y análisis de Amdahl con f empírico.
- [ ] Discutir separabilidad de grupos mediante score óptimo.
