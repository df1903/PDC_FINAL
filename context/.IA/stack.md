# Stack

## Lenguaje y runtime

- **Python 3.12+** (declarado en `code/pyproject.toml`)
- Gestor de dependencias Python: **uv** (`code/uv.lock` presente)
- **C11** con GCC; **CUDA C** con nvcc

## Dependencias Python (desde `code/pyproject.toml`)

| Paquete | Uso |
|---------|-----|
| `numpy` | Operaciones matriciales (P=W·[T,S,F], Score=A·P), generación Dirichlet, `.npy` I/O |
| `scikit-learn` | `roc_auc_score` para cálculo de AUC en Python |
| `multiprocessing` | stdlib; paralelismo multicore en `multicore.py` |
| `matplotlib` | Gráficas de speedup y eficiencia en `results/plots/` |
| `pycuda` | Interfaz Python para el kernel CUDA dentro de `CUDA/scoring_cuda.ipynb` (Colab) |
| `jupyter` / `notebook` | Entorno de la Fase 4: `scoring_cuda.ipynb` ejecutado en Google Colab |

## Toolchain C/CUDA

| Herramienta | Versión mínima | Uso |
|-------------|---------------|-----|
| GCC | ≥ 9.0 | Compilar `scoring_openmp.c` (`-fopenmp -O2`) |
| OpenMPI o MPICH | ≥ 3.0 | Compilar `scoring_mpi.c` (`mpicc -O2`) |
| CUDA Toolkit | ≥ 11.0 (provisto por Colab) | Compilar el kernel con `nvcc -O2` dentro de `scoring_cuda.ipynb` |
| Google Colab (runtime GPU) | T4 o superior | Entorno de ejecución de la Fase 4 (notebook) |

## Compilación

```bash
# C + OpenMP
gcc -O2 -fopenmp -o scoring_openmp scoring_openmp.c -lm

# C + MPI
mpicc -O2 -o scoring_mpi scoring_mpi.c -lm

# CUDA (Fase 4) — dentro de CUDA/scoring_cuda.ipynb en Google Colab:
#   1) celda con %%writefile scoring_kernel.cu  (código del kernel)
#   2) !nvcc -O2 ... scoring_kernel.cu          (o compilación vía PyCUDA SourceModule)

# Con Makefile (desde C_OpenMP_MPI/)
make openmp
make mpi
```

## Ejecución

```bash
# Python (desde code/)
python python/sequential.py
python python/multicore.py

# C (desde code/)
./C_OpenMP_MPI/scoring_openmp
mpirun -np 4 ./C_OpenMP_MPI/scoring_mpi

# CUDA (Fase 4): abrir CUDA/scoring_cuda.ipynb en Google Colab (runtime GPU) y
# ejecutar todas las celdas. Subir previamente los .npy de code/data/ al entorno Colab.

# Benchmark completo (niveles CPU)
bash run_all.sh
```

## Herramientas de desarrollo

- **pytest**: pruebas de correctitud del baseline Python.
- **ruff**: linter Python.
- **valgrind**: detección de memory leaks en binarios C.
- **nvprof / Nsight**: profiling de kernels CUDA.
