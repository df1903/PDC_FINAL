# Scoring Metagenómico HPC — `code/`

Implementación del proyecto de Scoring Metagenómico en distintos niveles de
paralelismo: Python secuencial/multicore (Fase 1 — completada), C con OpenMP
(Fase 2 — completada) / MPI (Fase 3 — en progreso, placeholder) y CUDA
(Fase 4, vía notebook en Google Colab).

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

## Ejecutar el baseline Python (Fase 1)

Desde `code/` (los scripts leen `data/n_{n_items}/` y escriben en
`results/benchmark.csv`, ambos relativos al directorio de ejecución):

```bash
# Nivel 1 — secuencial
uv run python python/sequential.py --n-items 50 --k-candidates 100000 --seed 42

# Nivel 1 — multicore (multiprocessing)
uv run python python/multicore.py --n-items 50 --k-candidates 100000 --workers 4 --seed 42
```

Ambos aceptan `--n-items` (50), `--k-candidates` (100000) y `--seed` (42) por
defecto; `multicore.py` además acepta `--workers` (por defecto
`cpu_count()`). Cada ejecución agrega una fila a `results/benchmark.csv`
(`best_auc`, `time_seconds`, `candidates_per_second`, `speedup`,
`efficiency`); `speedup`/`efficiency` de `multicore.py` se calculan contra la
última fila `Python secuencial` del CSV con el mismo `n_items`/`k_candidates`
(ejecuta primero `sequential.py` con esos mismos valores).

## Tests y lint

```bash
uv run pytest python/tests/ -q
uv run ruff check python/
```

## C + OpenMP (Fase 2 — completada)

Compilar desde `code/C_OpenMP_MPI/`:

```bash
make clean
make openmp       # gcc -O2 -Wall -Wextra -fopenmp scoring_openmp.c npy_io.c -o scoring_openmp -lm
```

El binario `scoring_openmp` lee `data/n_{n_items}/{matrix_A,profiles,labels}.npy`
y registra en `results/benchmark.csv`, ambos relativos al directorio de
ejecución: **ejecutar desde `code/`**, igual que los scripts de Python.

```bash
cd code   # si vienes de C_OpenMP_MPI/, usa `cd ..`

# Self-test de la función AUC (incluye caso con empate, ver ISSUE-008)
./C_OpenMP_MPI/scoring_openmp --self-test

# Run individual
./C_OpenMP_MPI/scoring_openmp --n-items 50 --k-candidates 100000 --seed 42 --threads 4

# Barrido de hilos (resultados reportados en context/state/current-phase.md)
for P in 1 2 4 8; do
  ./C_OpenMP_MPI/scoring_openmp --n-items 50 --k-candidates 100000 --seed 42 --threads $P
done
```

Flags: `--n-items` (50), `--k-candidates` (100000), `--seed` (42), `--threads`
(por defecto `omp_get_max_threads()`), `--self-test`. Cada run agrega una fila
`C OpenMP` a `results/benchmark.csv`; `speedup`/`efficiency` se calculan
contra la última fila `Python secuencial` con el mismo `n_items`/`k_candidates`.

## C + MPI (Fase 3 — en progreso)

`scoring_mpi.c` es por ahora un placeholder (`MPI_Init`/`MPI_Finalize` sin
lógica de scoring). Pendiente: `MPI_Scatter` de candidatos W desde el proceso
raíz, evaluación local de AUC y `MPI_Reduce(MPI_MAX)`, reutilizando
`npy_io.{h,c}` (ver "Próxima acción" en `context/state/current-phase.md`).

```bash
make mpi          # mpicc -O2 -Wall -Wextra scoring_mpi.c -o scoring_mpi -lm
mpirun -np 4 ./scoring_mpi
```

## CUDA (Fase 4)

La Fase 4 se desarrolla en `CUDA/scoring_cuda.ipynb`, ejecutado en
**Google Colab** con runtime GPU (T4 o superior). Antes de correrlo, sube los
`.npy` generados en `data/n_{n_items}/` al entorno de Colab.

## Resultados

Cada ejecución (Python o C OpenMP) agrega una fila a `results/benchmark.csv`
(esquema de 9 columnas, append-only). Los resultados de Fase 1 y Fase 2 ya
registrados, junto con su análisis, están resumidos en
`context/state/current-phase.md`.

## Estructura

```
code/
├── pyproject.toml / uv.lock   # Dependencias Python (uv)
├── data/                       # Generación/carga de datasets (.npy)
├── python/                     # Nivel 1: secuencial y multicore
│   └── tests/                  # Pruebas pytest
├── C_OpenMP_MPI/                # Nivel 2-3: OpenMP (listo) y MPI (placeholder)
│   ├── npy_io.{h,c}             # Parser .npy propio (sin libs externas)
│   ├── scoring_openmp.c          # Random Search + OpenMP (Fase 2)
│   └── scoring_mpi.c              # Random Search + MPI (Fase 3, pendiente)
├── CUDA/                        # Nivel 3 (Fase 4): notebook + kernel CUDA
└── results/                     # benchmark.csv y plots/
```
