# Rules

Reglas que el agente NO puede violar sin confirmación explícita del usuario.

## Datos y reproducibilidad

- Dataset estándar en `code/data/n_{n_items}/` (un subdirectorio por tamaño de problema, ver DEC-10):
  - `matrix_A.npy` — matriz de contribución, shape 10×N, float32.
  - `profiles.npy` — perfiles por ítem, shape (N, 3), float32:
    - columna 0 = `T` — perfil taxonómico, float en [0, 1].
    - columna 1 = `S` — perfil socioeconómico, float en [0, 1].
    - columna 2 = `F` — perfil funcional, entero en **{0, 1, 2}** (almacenado como float32).
  - `labels.npy` — etiquetas, shape (10,), int32.
- `T`, `S`, `F` son perfiles **independientes por ítem**; no se derivan particionando columnas de `A`.
- **Semilla fija: `seed=42`** en todas las implementaciones (Python, C, CUDA).
- N_ITEMS por defecto: **50** (`data/n_50/`). Para otros tamaños, usar/crear `data/n_{n_items}/`.
- K_CANDIDATES por defecto: **100 000** por búsqueda. Configurable vía argumento.
- El vector W = (W₁, W₂, W₃) debe satisfacer el simplex 3D: W₁ + W₂ + W₃ = 1, Wᵢ ≥ 0 (independiente de N).
- No modificar `matrix_A.npy`, `profiles.npy` ni `labels.npy` sin regenerar también todos los resultados de referencia.

## Correctitud

- Toda implementación (C, CUDA) debe producir `AUC ≥ AUC_Python_secuencial − 1e-4` (mismo K, misma semilla).
- AUC válido está en [0.5, 1.0]. Valores fuera de este rango indican bug grave.
- Consistencia de scoring ≥ 0.8 (ec. 4 del proyecto) es criterio de aceptación adicional.

## Código

- Python: máximo **200 LOC** por archivo, type hints obligatorios en firmas públicas.
- C: máximo **400 LOC** por archivo; lógica de cómputo separada de `main()`.
- CUDA: un archivo `.cu` por kernel; separar `host_main` de `__global__` kernels.
- No cambiar las firmas públicas de `random_search` / `random_search_multicore` sin actualizar el benchmark.
- No agregar dependencias Python sin declarar en `pyproject.toml`.
- No usar librerías externas en C sin agregar al `Makefile` correspondiente.

## Benchmark

- Tiempo medido: desde inicio de la búsqueda hasta obtener W* (excluye carga de datos).
- `results/benchmark.csv` es la fuente de verdad. No editar manualmente.
- Speedup siempre calculado respecto a **Python secuencial** (S_base = 1.00).
- Esquema de columnas (desde Fase 1, DEC-11 / `context/state/active-tasks.md`):
  `implementation,n_items,k_candidates,workers,best_auc,time_seconds,candidates_per_second,speedup,efficiency`.
  Append-only; no sobrescribir filas anteriores.

## Firmas canónicas Fase 1 (Python)

- `random_search(A, profiles, y, K=100_000, seed=42) -> tuple[np.ndarray, float]` en `sequential.py`.
- `random_search_multicore(A, profiles, y, K=100_000, seed=42, workers=cpu_count()) -> tuple[np.ndarray, float]` en `multicore.py`.
- `profiles` es el array `(N, 3)` de DEC-10 (`T=profiles[:,0]`, `S=profiles[:,1]`, `F=profiles[:,2]`).
- Lógica compartida entre ambos scripts vive en `code/python/common.py` (ver `context/state/active-tasks.md`).
- CLI obligatoria: `--n-items` (50), `--k-candidates` (100000), `--seed` (42); `multicore.py` además `--workers` (cpu_count()).

## Nomenclatura (no negociable)

| Implementación | Archivo |
|----------------|---------|
| Python secuencial | `python/sequential.py` |
| Python multicore | `python/multicore.py` |
| C + OpenMP | `C_OpenMP_MPI/scoring_openmp.c` |
| C + MPI | `C_OpenMP_MPI/scoring_mpi.c` |
| CUDA (Fase 4, entregable Colab) | `CUDA/scoring_cuda.ipynb` |
| CUDA kernel (fuente derivada) | `CUDA/scoring_kernel.cu` |
| CUDA Python (fuente derivada) | `CUDA/scoring_pycuda.py` |
| Generador de datos | `data/generate_data.py` |
| Script benchmark | `run_all.sh` (raíz de `code/`) |
