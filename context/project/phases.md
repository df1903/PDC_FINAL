# Phases

## Fase 1 — Python Baseline `[PLANIFICADA — lista para implementación]`

Objetivo: implementación de referencia correcta y validada según el modelo de perfiles (DEC-07), con CLI, medición de tiempo (excluyendo carga de datos) y registro automático en `results/benchmark.csv`.

Plan técnico detallado y tareas concretas: ver `context/state/active-tasks.md`. Resumen ordenado:

- [ ] **DEC-11**: actualizar `code/data/generate_data.py` para inyectar señal diferencial en las filas enfermas de `A` y regenerar `data/n_50/` y `data/n_100/` (`seed=42`).
- [ ] Crear `code/python/common.py` con funciones compartidas entre `sequential.py` y `multicore.py`: `load_dataset`, `validate_dataset`, `sample_candidates`, `score_samples`, `evaluate_candidates`, `scoring_consistency`, `compute_metrics`, `append_benchmark`, `read_sequential_time`.
- [ ] Reescribir `code/python/sequential.py`: `random_search(A, profiles, y, K=100_000, seed=42)`, CLI (`--n-items`, `--k-candidates`, `--seed`), timing con `perf_counter` (excluye carga), registro en benchmark.
- [ ] Reescribir `code/python/multicore.py`: `random_search_multicore(A, profiles, y, K=100_000, seed=42, workers=cpu_count())`, candidatos generados en el proceso principal y repartidos con `np.array_split` + `multiprocessing.Pool`, CLI (`--n-items`, `--k-candidates`, `--workers`, `--seed`).
- [ ] Migrar `code/results/benchmark.csv` al esquema: `implementation,n_items,k_candidates,workers,best_auc,time_seconds,candidates_per_second,speedup,efficiency` (append-only; crear si no existe; no sobrescribir filas previas).
- [ ] Crear pruebas (`code/python/tests/test_baseline.py` o `code/tests/`): equivalencia AUC secuencial↔multicore (<1e-4, RF-04), AUC∈[0.5,1], consistencia≥0.8, caso chico N=3/K=100 (RIESGO-03).
- [ ] Validar AUC ∈ [0.5, 1.0] y consistencia ≥ 0.8 con el dataset regenerado (DEC-11).
- [ ] Medir T_secuencial como baseline para speedup (S_base = 1.00).
- [ ] Ejecutar multicore y verificar speedup ≥ 1.5× (RNF-03); calcular `efficiency = speedup / workers`.
- [ ] Registrar ambas filas en `results/benchmark.csv`.

## Fase 2 — C + OpenMP `[PENDIENTE]`

Objetivo: paralelismo de memoria compartida sobre CPU.

- [ ] Implementar carga de datos en `scoring_openmp.c`.
- [ ] Implementar Random Search con `#pragma omp parallel for`.
- [ ] Validar equivalencia AUC con Fase 1.
- [ ] Medir speedup vs Python secuencial con P ∈ {1, 2, 4, 8}.

## Fase 3 — C + MPI `[PENDIENTE]`

Objetivo: paralelismo de memoria distribuida.

- [ ] Implementar `MPI_Scatter` de candidatos desde root.
- [ ] Implementar evaluación local y `MPI_Reduce(MPI_MAX)`.
- [ ] Validar equivalencia AUC.
- [ ] Medir speedup con P ∈ {1, 2, 4, 8} procesos.

## Fase 4 — CUDA (notebook en Google Colab) `[PENDIENTE]`

Objetivo: aceleración GPU masiva en `CUDA/scoring_cuda.ipynb`, ejecutado en Colab.

- [ ] Crear `scoring_cuda.ipynb` con celdas: setup/GPU → `%%writefile scoring_kernel.cu` → compilación `nvcc` → orquestación PyCUDA.
- [ ] Implementar kernel `scoring_kernel` (un hilo por candidato W_k; calcula `P=W·[T,S,F]` y `Score=A·P`).
- [ ] Usar `__shared__` memory para filas de A por bloque.
- [ ] Implementar reduction kernel para AUC máximo.
- [ ] Subir los `.npy` de `code/data/` al runtime de Colab.
- [ ] Medir speedup vs Python secuencial y registrar el modelo de GPU usado.

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
