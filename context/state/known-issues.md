# Known Issues

## ISSUE-001 — `run_all.sh` no existe

**Descripción**: el script maestro de benchmark no ha sido creado.

**Impacto**: no se puede ejecutar el benchmark completo ni generar `results/benchmark.csv` automáticamente.

**Estado**: pendiente (Fase 5).

---

## ISSUE-002 — Carga de datos no implementada en scoring_openmp.c `[RESUELTO — 2026-06-15]`

**Archivo**: `code/C_OpenMP_MPI/scoring_openmp.c` línea 29

**Descripción**: el `main()` imprime un placeholder y retorna sin cargar datos ni ejecutar la búsqueda.

**Estado**: resuelto en Fase 2 (2026-06-15, `context/state/active-tasks.md`). Implementados `code/C_OpenMP_MPI/npy_io.{h,c}` (parser `.npy` v1.0 propio, sin libs externas) y `load_dataset`/`free_dataset` en `scoring_openmp.c` (espejo de `validate_dataset`), Random Search OpenMP completo y registro en `results/benchmark.csv`. Detalle: `traceability_data/2026_06_15_14-36.md`.

---

## ISSUE-003 — scoring_mpi.c placeholder y notebook CUDA inexistente

**Archivos**: `code/C_OpenMP_MPI/scoring_mpi.c` (placeholder), `code/CUDA/` (falta `scoring_cuda.ipynb`; `scoring_kernel.cu`/`scoring_pycuda.py` son placeholders).

**Nota**: la Fase 4 se entrega como `CUDA/scoring_cuda.ipynb` para Google Colab (DEC-09).

**Estado**: pendiente (Fases 3 y 4).

---

## ISSUE-004 — `sequential.py` y `multicore.py` eran placeholders no funcionales

**Archivos**: `code/python/sequential.py`, `code/python/multicore.py`.

**Descripción**: ambos cargaban rutas planas antiguas (`../data/matrix_A.npy`, `../data/labels.npy`, no `data/n_{n_items}/`), no cargaban `profiles.npy`, e implementaban `P = A @ W[...]` / `A.mean(axis=1)` en vez del modelo DEC-07 (`P=W1*T+W2*S+W3*F`, `Score=A@P`). Sin CLI, sin timing, sin registro en benchmark.

**Estado**: resuelto en Fase 1 (2026-06-14, tareas 3-4 de `context/state/active-tasks.md`). Reescritos con `common.py`, firmas canónicas, CLI, timing con `perf_counter` y registro en `results/benchmark.csv`.

---

## ISSUE-005 — `generate_data.py` no inyectaba señal diferencial (RIESGO-04)

**Archivo**: `code/data/generate_data.py`.

**Descripción**: `A` era Dirichlet puro y los perfiles `T/S/F` eran aleatorios sin correlación con `A` ni con `y`. `P` resultaba común a todas las muestras y ningún `W` del simplex producía AUC > 0.5 ni consistencia ≥ 0.8.

**Estado**: resuelto en Fase 1 (2026-06-14, DEC-11, tarea 1 de `context/state/active-tasks.md`). Filas enfermas (5-9) de `A` ahora usan Dirichlet sesgada por `importance = T + F` (`SIGNAL_STRENGTH = 8.0`); `data/n_50/` y `data/n_100/` regenerados con `seed=42`. AUC resultante = 1.0000, consistencia = 2.0000.

---

## ISSUE-006 — `benchmark.csv` con esquema antiguo

**Archivo**: `code/results/benchmark.csv`.

**Descripción**: cabecera anterior `implementacion,T_s,speedup,eficiencia,AUC` con filas plantilla (sin datos reales), distinta del esquema requerido por Fase 1: `implementation,n_items,k_candidates,workers,best_auc,time_seconds,candidates_per_second,speedup,efficiency`.

**Estado**: resuelto en Fase 1 (2026-06-14, tarea 5 de `context/state/active-tasks.md`). Cabecera migrada al esquema nuevo (no se perdieron resultados reales, las filas anteriores eran plantilla vacía); ya contiene las filas reales de "Python secuencial" y "Python multicore".

---

## ISSUE-007 — `code/pyproject.toml` no existía `[RESUELTO — ya existía]`

**Descripción**: `context/.IA/stack.md` y `context/.IA/directory-structure.md` mencionaban `code/pyproject.toml` y `code/uv.lock` como inexistentes.

**Estado**: al verificar en Fase 1 (2026-06-14, tarea 6 de `context/state/active-tasks.md`) se encontró que `code/pyproject.toml` y `code/uv.lock` **ya existían**, con `numpy`/`scikit-learn`/`matplotlib` declarados y `pytest`/`ruff` como `dependency-groups.dev`; `code/.venv/` ya tenía estos paquetes instalados. No fue necesario crear nada; solo se agregó `[tool.pytest.ini_options] pythonpath = ["python"]` para los tests de `code/python/tests/test_baseline.py`.

---

## ISSUE-009 — `speedup`/`efficiency` de "C OpenMP" comparaban contra Python, no paralelismo `[RESUELTO — 2026-06-15]`

**Archivo**: `code/results/benchmark.csv`, `code/python/common.py`, `code/C_OpenMP_MPI/scoring_openmp.c`.

**Descripción**: el esquema de Fase 2 (DEC-12, ISSUE-008) calculaba `speedup_openmp =
T_python_secuencial / T_openmp` para todas las filas "C OpenMP", produciendo valores de miles
(2386×–14813×) que mezclan la diferencia de rendimiento C vs Python con el speedup paralelo de
OpenMP; `efficiency = speedup/threads` resultante (1851×–2777×) no era interpretable como
eficiencia de paralelización clásica.

**Estado**: resuelto (2026-06-15, DEC-13, `traceability_data/2026_06_15_15-24.md`). `speedup`/
`efficiency` ahora son relativos al baseline propio `workers=1` de cada implementación
(`speedup_openmp(P)=T_openmp(1)/T_openmp(P)`); la comparación contra Python vive en la nueva
columna `speedup_vs_python`. Esquema de `benchmark.csv` 9→10 columnas; `compute_metrics` en
`common.py` y `read_last_time`/`append_benchmark` en `scoring_openmp.c` actualizados; 3 tests
nuevos en `test_baseline.py` (8/8 OK); `benchmark.csv` regenerado: `speedup(P=1..8) = 1.00,
2.04, 3.84, 5.40`.

---

## ISSUE-008 — `compute_auc` del placeholder C no acredita empates (RIESGO-03) `[RESUELTO — 2026-06-15]`

**Archivo**: `code/C_OpenMP_MPI/scoring_openmp.c` líneas 13-26.

**Descripción**: el `compute_auc` del placeholder solo cuenta pares estrictos (`scores[i] > scores[j]`) y **no acredita los empates**, mientras que `sklearn.roc_auc_score` (usado por la Fase 1) otorga **0.5 por empate**. En datasets con scores empatados esto produce divergencia de AUC entre C y Python (RIESGO-03).

**Estado**: resuelto en Fase 2 (2026-06-15, tarea 4 de `context/state/active-tasks.md`). `compute_auc` ahora calcula `AUC = (concordantes + 0.5·empates) / (pos·neg)`; validado con `--self-test` (3 casos, incluido `scores=[1,2,2,3]`/`y=[0,0,1,1]` → AUC=0.875, igual a `sklearn.roc_auc_score`). Detalle: `traceability_data/2026_06_15_14-36.md`.
