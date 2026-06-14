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
