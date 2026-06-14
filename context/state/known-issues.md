# Known Issues

## ISSUE-001 — `run_all.sh` no existe

**Descripción**: el script maestro de benchmark no ha sido creado.

**Impacto**: no se puede ejecutar el benchmark completo ni generar `results/benchmark.csv` automáticamente.

**Estado**: pendiente (Fase 5).

---

## ISSUE-002 — Carga de datos no implementada en scoring_openmp.c

**Archivo**: `code/C_OpenMP_MPI/scoring_openmp.c` línea 29

**Descripción**: el `main()` imprime un placeholder y retorna sin cargar datos ni ejecutar la búsqueda.

**Estado**: pendiente (Fase 2).

---

## ISSUE-003 — scoring_mpi.c placeholder y notebook CUDA inexistente

**Archivos**: `code/C_OpenMP_MPI/scoring_mpi.c` (placeholder), `code/CUDA/` (falta `scoring_cuda.ipynb`; `scoring_kernel.cu`/`scoring_pycuda.py` son placeholders).

**Nota**: la Fase 4 se entrega como `CUDA/scoring_cuda.ipynb` para Google Colab (DEC-09).

**Estado**: pendiente (Fases 3 y 4).

---

## ISSUE-004 — `sequential.py` y `multicore.py` son placeholders no funcionales

**Archivos**: `code/python/sequential.py`, `code/python/multicore.py`.

**Descripción**: ambos cargan rutas planas antiguas (`../data/matrix_A.npy`, `../data/labels.npy`, no `data/n_{n_items}/`), no cargan `profiles.npy`, e implementan `P = A @ W[...]` / `A.mean(axis=1)` en vez del modelo DEC-07 (`P=W1*T+W2*S+W3*F`, `Score=A@P`). Sin CLI, sin timing, sin registro en benchmark.

**Estado**: a resolver en Fase 1 (ver `context/state/active-tasks.md`, tareas 3-4).

---

## ISSUE-005 — `generate_data.py` no inyecta señal diferencial (RIESGO-04)

**Archivo**: `code/data/generate_data.py`.

**Descripción**: `A` es Dirichlet puro y los perfiles `T/S/F` son aleatorios sin correlación con `A` ni con `y`. `P` resulta común a todas las muestras y ningún `W` del simplex produce AUC > 0.5 ni consistencia ≥ 0.8.

**Estado**: resuelto vía DEC-11 — a implementar en Fase 1 (tarea 1 de `context/state/active-tasks.md`).

---

## ISSUE-006 — `benchmark.csv` con esquema antiguo

**Archivo**: `code/results/benchmark.csv`.

**Descripción**: cabecera actual `implementacion,T_s,speedup,eficiencia,AUC` con filas plantilla (sin datos reales), distinta del esquema requerido por Fase 1: `implementation,n_items,k_candidates,workers,best_auc,time_seconds,candidates_per_second,speedup,efficiency`.

**Estado**: a migrar en Fase 1 (tarea 5 de `context/state/active-tasks.md`). No se pierden resultados reales (las filas actuales están vacías).

---

## ISSUE-007 — `code/pyproject.toml` no existe

**Descripción**: `context/.IA/stack.md` y `context/.IA/directory-structure.md` mencionan `code/pyproject.toml` y `code/uv.lock`, pero ninguno existe en el repo. `code/.env/` (probable venv) está vacío. `rules.md` exige declarar dependencias Python en `pyproject.toml`.

**Impacto**: no está claro qué entorno/dependencias usa `python python/sequential.py` actualmente (numpy, scikit-learn ya se usan en el código existente sin estar declarados en ningún lado).

**Estado**: a verificar al inicio de la implementación de Fase 1 (tarea 6 de `context/state/active-tasks.md`); si falta, crear `pyproject.toml` declarando numpy/scikit-learn/multiprocessing(stdlib)/pytest — confirmar con el usuario antes de crearlo.
