# Scoring Metagenómico HPC — `code/`

Implementación del proyecto de Scoring Metagenómico en distintos niveles de
paralelismo: Python secuencial/multicore (Fase 1), C con OpenMP/MPI (Fase 2-3)
y CUDA (Fase 4, vía notebook en Google Colab).

## Requisitos

- **Python ≥ 3.12**
- **[uv](https://docs.astral.sh/uv/)** para gestionar el entorno y las dependencias
- **GCC ≥ 9** (para `C_OpenMP_MPI/`, target `openmp`)
- **OpenMPI o MPICH** (para `C_OpenMP_MPI/`, target `mpi`)
- **CUDA Toolkit** — solo necesario dentro de Google Colab para `CUDA/scoring_cuda.ipynb`

## Entorno Python (uv)

Desde `code/`:

```bash
# Crea .venv e instala dependencias (numpy, scikit-learn, matplotlib, pytest, ruff)
uv sync
```

Esto genera `.venv/` (no versionado) a partir de `pyproject.toml` y `uv.lock`.
No es necesario activar el entorno manualmente: usa `uv run <comando>` para
ejecutar cualquier script o herramienta dentro del entorno.

Para agregar una nueva dependencia:

```bash
uv add <paquete>            # dependencia principal
uv add --dev <paquete>      # dependencia de desarrollo (tests/lint)
```

## Generar/cargar datos

```bash
uv run python data/generate_data.py
```

Genera (o reutiliza si ya existen) `matrix_A.npy`, `profiles.npy` y
`labels.npy` dentro de `data/n_{n_items}/` (por defecto `n_items=50`).

## Ejecutar el baseline Python

```bash
# Nivel 1 — secuencial
uv run python python/sequential.py

# Nivel 1 — multicore (multiprocessing)
uv run python python/multicore.py
```

> Nota: `sequential.py` y `multicore.py` están siendo migrados (Fase 1) a un
> CLI con `--n-items`, `--k-candidates`, `--workers`, `--seed` y registro
> automático en `results/benchmark.csv` (ver `context/state/active-tasks.md`).
> Mientras eso se completa, los scripts pueden ejecutarse sin argumentos.

## Tests y lint

```bash
uv run pytest python/tests/ -q
uv run ruff check python/
```

## C + OpenMP / MPI (Fase 2-3)

Desde `code/C_OpenMP_MPI/`:

```bash
make openmp       # gcc -O2 -fopenmp
make mpi          # mpicc -O2
make clean

./scoring_openmp
mpirun -np 4 ./scoring_mpi
```

## CUDA (Fase 4)

La Fase 4 se desarrolla en `CUDA/scoring_cuda.ipynb`, ejecutado en
**Google Colab** con runtime GPU (T4 o superior). Antes de correrlo, sube los
`.npy` generados en `data/n_{n_items}/` al entorno de Colab.

## Estructura

```
code/
├── pyproject.toml / uv.lock   # Dependencias Python (uv)
├── data/                       # Generación/carga de datasets (.npy)
├── python/                     # Nivel 1: secuencial y multicore
│   └── tests/                  # Pruebas pytest
├── C_OpenMP_MPI/                # Nivel 2: OpenMP y MPI
├── CUDA/                        # Nivel 3 (Fase 4): notebook + kernel CUDA
└── results/                     # benchmark.csv y plots/
```
