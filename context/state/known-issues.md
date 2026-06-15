# Known Issues

## ISSUE-001 — `run_all.sh` no existe

**Descripción**: el script maestro de benchmark no ha sido creado.

**Impacto**: no se puede ejecutar el benchmark completo ni generar `results/benchmark.csv` automáticamente.

**Estado**: pendiente (Fase 5).

---

## ISSUE-002 — Carga de datos no implementada en scoring_openmp.c

**Archivo**: `code/C_OpenMP_MPI/scoring_openmp.c` línea 29

**Descripción**: el `main()` imprime un placeholder y retorna sin cargar datos ni ejecutar la búsqueda.

**Estado**: pendiente (Fase 2). **Plan definido** (2026-06-15) en `context/state/active-tasks.md`: carga vía parser `.npy` mínimo (`npy_io.{h,c}`, sin libs externas), Random Search OpenMP y registro en `benchmark.csv`.

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

## ISSUE-008 — `compute_auc` del placeholder C no acredita empates (RIESGO-03)

**Archivo**: `code/C_OpenMP_MPI/scoring_openmp.c` líneas 13-26.

**Descripción**: el `compute_auc` del placeholder solo cuenta pares estrictos (`scores[i] > scores[j]`) y **no acredita los empates**, mientras que `sklearn.roc_auc_score` (usado por la Fase 1) otorga **0.5 por empate**. En datasets con scores empatados esto produce divergencia de AUC entre C y Python (RIESGO-03).

**Estado**: pendiente (Fase 2). **Plan definido** (2026-06-15, tarea 4 de `context/state/active-tasks.md`): corregir a `AUC = (concordantes + 0.5·empates) / (pos·neg)` y validar con `--self-test` (incluido un caso con empate) contra valores conocidos de sklearn antes de escalar.
