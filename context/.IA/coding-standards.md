# Coding Standards

## Python

| Elemento | Convención | Ejemplo |
|----------|-----------|---------|
| Funciones | `snake_case` | `random_search`, `evaluate_batch` |
| Variables | `snake_case` | `best_auc`, `n_workers`, `candidates` |
| Constantes de módulo | `UPPER_SNAKE` | `K_DEFAULT`, `SEED` |

- **Type hints** obligatorios en toda firma pública.
- Máximo **200 LOC** por archivo (sin docstrings ni comentarios).
- Imports: orden PEP 8 — stdlib → third-party → local.
- Vectorizar con NumPy; evitar bucles Python sobre arrays grandes.
- Cómputo del score: `P = W[0]*T + W[1]*S + W[2]*F` (vector dim N) y `Score = A @ P` (dim 10). `T`, `S`, `F` se cargan como `.npy` independientes; nunca particionar columnas de `A`.
- `time.perf_counter()` para medir tiempo (resolución nanosegundos).

```python
def random_search(
    A: np.ndarray,
    y: np.ndarray,
    K: int = 100_000,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    ...
```

## C

- `#define` para todas las constantes (`N_SAMPLES`, `K_CANDIDATES`, `BLOCK_SIZE`).
- Include guard en todo header: `#ifndef SCORING_H … #endif`.
- Documentar cada `#pragma omp` con un comentario de una línea explicando el patrón.
- `omp_get_wtime()` para medir tiempo en C.
- Nunca usar VLAs (Variable Length Arrays); usar `malloc`/`free` explícitos.
- Compilar siempre con `-Wall -Wextra -O2`.

```c
/* Reducción paralela: cada hilo actualiza su máximo local */
#pragma omp parallel for reduction(max:best_auc) schedule(dynamic, 64)
for (int k = 0; k < K_CANDIDATES; k++) { ... }
/* Sección crítica para actualizar best_W */
#pragma omp critical
{ if (local_auc > best_auc) { best_auc = local_auc; memcpy(best_W, W_k, 3*sizeof(double)); } }
```

## CUDA (notebook `.ipynb` en Google Colab)

- La Fase 4 vive en `CUDA/scoring_cuda.ipynb`. Organización de celdas recomendada:
  1. Setup: verificar GPU (`!nvidia-smi`), instalar/`import pycuda`, cargar `A, T, S, F, y` (`.npy`).
  2. `%%writefile scoring_kernel.cu` con el código del kernel.
  3. Compilación: `!nvcc -O2 ...` (o `SourceModule` de PyCUDA).
  4. Orquestación: transferencias H→D, lanzamiento del kernel, reduction.
  5. Medición y registro de T/speedup.
- Nombrar kernels: `__global__ void scoring_kernel(...)`.
- `__shared__` para datos reutilizados por todos los hilos del bloque (filas de A).
- Transferencias H→D realizadas **una sola vez** antes del kernel.
- Usar CUDA events (`cudaEvent_t`) para medir tiempo de kernel.
- `BLOCK_SIZE = 256` como constante de compilación.
- Verificar errores CUDA con macro `CUDA_CHECK`.

```c
#define BLOCK_SIZE 256
#define CUDA_CHECK(e) { cudaError_t r=(e); if(r!=cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(r)); exit(1); } }
```

## Benchmark / scripts

- `run_all.sh`: ejecutar todos los niveles y volcar métricas a `results/benchmark.csv`.
- Columnas mínimas de `benchmark.csv`: `impl,T_s,S,E,AUC,K,n_threads`.
- Gráficas en `results/plots/` como PNG (matplotlib en Python).
- Encabezado del CSV generado automáticamente si no existe; no sobreescribir si ya tiene datos.
