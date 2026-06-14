# Directory Structure

Estructura verificada del repositorio `PDC_FINAL/`.

```
PDC_FINAL/
├── code/                              # Todo el código del proyecto
│   ├── data/
│   │   ├── generate_data.py          # Genera/carga matrix_A.npy, profiles.npy y labels.npy por n_items
│   │   └── n_{n_items}/               # Un subdirectorio por tamaño de problema (ej. n_50/, n_100/)
│   │       ├── matrix_A.npy          # A ∈ ℝ^{10×N}, float32 — matriz de contribución (seed=42)
│   │       ├── profiles.npy          # (N,3) float32 — columnas T, S, F (perfiles por ítem)
│   │       └── labels.npy            # y ∈ {0,1}^{10}, int32
│   ├── python/
│   │   ├── common.py                 # Nivel 1: funciones compartidas (Fase 1, ver context/state/active-tasks.md)
│   │   ├── sequential.py             # Nivel 1: Random Search secuencial
│   │   ├── multicore.py              # Nivel 1: Random Search multicore (Pool)
│   │   └── tests/
│   │       └── test_baseline.py      # Equivalencia AUC, validaciones AUC/consistencia (Fase 1)
│   ├── C_OpenMP_MPI/
│   │   ├── scoring_openmp.c          # Nivel 2a: C + OpenMP
│   │   ├── scoring_mpi.c             # Nivel 2b: C + MPI
│   │   └── Makefile                  # Targets: openmp, mpi, clean
│   ├── CUDA/
│   │   ├── scoring_cuda.ipynb        # Nivel 3 (Fase 4): notebook principal para Google Colab
│   │   ├── scoring_kernel.cu         # Fuente del kernel (derivada/embebida en el notebook)
│   │   ├── scoring_pycuda.py         # Orquestación PyCUDA (fuente derivada del notebook)
│   │   └── Makefile                  # Target: cuda, clean
│   ├── results/
│   │   ├── benchmark.csv             # Métricas consolidadas (generado por run_all.sh)
│   │   └── plots/                    # PNGs de speedup y eficiencia
│   ├── report/                       # Informe técnico (PDF final)
│   ├── run_all.sh                    # [PENDIENTE] Script maestro de benchmark
│   ├── pyproject.toml                # Dependencias Python (uv)
│   └── uv.lock
│
├── context/                          # Base de conocimiento para el agente IA
│   ├── .IA/                          # Conocimiento fijo (arquitectura, reglas, stack)
│   │   ├── instructions.md
│   │   ├── rules.md
│   │   ├── architecture.md
│   │   ├── coding-standards.md
│   │   ├── constraints.md
│   │   ├── directory-structure.md
│   │   └── stack.md
│   ├── project/                      # Visión global del proyecto
│   │   ├── requirements.md
│   │   ├── phases.md
│   │   ├── decisions.md
│   │   └── risks.md
│   ├── handoffs/                     # Notas de traspaso entre sesiones
│   │   └── 0.md
│   └── state/                        # Estado actual del proyecto
│       ├── current-phase.md
│       ├── progress.md
│       ├── active-tasks.md
│       └── known-issues.md
│
├── docs/                             # Documentación LaTeX (manuales)
├── others/                           # Insumos: proyecto_final.md, proyecto.pdf
├── traceability_data/                # Trazabilidad de conversaciones IA
└── instruction.md                    # Instrucciones para el agente IA
```

## Notas sobre rutas

- Los datos (`matrix_A.npy`, `profiles.npy`, `labels.npy`) se generan/cargan desde `code/data/n_{n_items}/` (vía `get_data_directory(n_items)`) y se leen con rutas relativas `../data/n_{n_items}/` desde los scripts de `python/`.
- Los ejecutables C se generan en `code/C_OpenMP_MPI/` al correr `make`.
- `run_all.sh` debe ejecutarse desde `code/` como directorio de trabajo (cubre los niveles CPU; la Fase 4 CUDA corre en Colab).
- Para la Fase 4 hay que **subir los `.npy` de `code/data/` al runtime de Colab** antes de ejecutar `scoring_cuda.ipynb`.
- Los archivos `.npy` **no se versionan** si son muy grandes; regenerarlos con `generate_data.py` (se generan únicamente los faltantes en `data/n_{n_items}/`).
- `pyproject.toml` / `uv.lock` aparecen en este listado como referencia de la convención de dependencias, pero **no existen aún** en el repo (ver `context/state/known-issues.md` ISSUE-007). Verificar/crear al implementar la Fase 1.
