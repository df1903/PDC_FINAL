# Directory Structure

Estructura verificada del repositorio `PDC_FINAL/`.

```
PDC_FINAL/
├── code/                              # Todo el código del proyecto
│   ├── data/
│   │   ├── generate_data.py          # Genera matrix_A.npy, T/S/F.npy y labels.npy
│   │   ├── matrix_A.npy              # A ∈ ℝ^{10×N}, float32 — matriz de contribución (seed=42)
│   │   ├── T.npy                     # Perfil taxonómico, (N,) float32
│   │   ├── S.npy                     # Perfil socioeconómico, (N,) float32
│   │   ├── F.npy                     # Perfil funcional, (N,) int32 ∈ {0,1,2}
│   │   └── labels.npy                # y ∈ {0,1}^{10}, int32
│   ├── python/
│   │   ├── sequential.py             # Nivel 1: Random Search secuencial
│   │   └── multicore.py              # Nivel 1: Random Search multicore (Pool)
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

- Los datos (`matrix_A.npy`, `T.npy`, `S.npy`, `F.npy`, `labels.npy`) se generan desde `code/data/` y se leen con rutas relativas `../data/` desde los scripts de `python/`.
- Los ejecutables C se generan en `code/C_OpenMP_MPI/` al correr `make`.
- `run_all.sh` debe ejecutarse desde `code/` como directorio de trabajo (cubre los niveles CPU; la Fase 4 CUDA corre en Colab).
- Para la Fase 4 hay que **subir los `.npy` de `code/data/` al runtime de Colab** antes de ejecutar `scoring_cuda.ipynb`.
- Los archivos `.npy` **no se versionan** si son muy grandes; regenerarlos con `generate_data.py`.
