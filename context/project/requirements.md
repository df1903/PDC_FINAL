# Requirements

## Requisitos funcionales

### RF-01 — Generación de datos
El sistema debe generar, con semilla configurable (`data/generate_data.py`):
- `A ∈ ℝ^{10×N}` (float32) — matriz de contribución.
- `T ∈ ℝ^N` (float32) — perfil taxonómico (abundancias de microorganismos y su diferencia).
- `S ∈ ℝ^N` (float32) — perfil socioeconómico.
- `F ∈ {0,1,2}^N` (int32) — perfil funcional (conteo de genes beneficiosos por ítem).
- `y ∈ {0,1}^{10}` (int32) — etiquetas (filas 0–4 sanas, 5–9 enfermas).

`T`, `S`, `F` son perfiles independientes por ítem (no se derivan de columnas de `A`). El generador debe inyectar señal diferencial en las filas enfermas de `A` (correlación con ítems de alto F/T) para que exista un W con AUC alto.

### RF-02 — Búsqueda de pesos óptimos
El sistema debe encontrar W* = (W₁, W₂, W₃) que maximice AUC(y, Score(W)) mediante Random Search sobre K candidatos muestreados uniformemente del simplex {W₁+W₂+W₃=1, Wᵢ≥0}.

El score por ítem se calcula como: P_i = W₁·T_i + W₂·S_i + W₃·F_i  (P ∈ ℝ^N)  
El score por muestra: Score = A · P  (Score ∈ ℝ^{10})  

### RF-03 — Implementaciones múltiples
La búsqueda debe implementarse en:
- Python secuencial (`python/sequential.py`)
- Python multicore con `multiprocessing.Pool` (`python/multicore.py`)
- C con OpenMP (`C_OpenMP_MPI/scoring_openmp.c`)
- C con MPI (`C_OpenMP_MPI/scoring_mpi.c`)
- CUDA en notebook para Google Colab (`CUDA/scoring_cuda.ipynb`: kernel CUDA C + orquestación PyCUDA)

### RF-04 — Equivalencia de resultados
Todas las implementaciones deben producir AUC dentro de 1e-4 del Python secuencial dado el mismo K y semilla.

### RF-05 — Métricas de rendimiento
El sistema debe reportar para cada implementación: T(s), speedup S, eficiencia E, AUC obtenido y número de hilos/procesos P.

### RF-06 — Benchmark automatizado
Un script `run_all.sh` debe ejecutar todas las implementaciones, medir tiempos y generar `results/benchmark.csv` y gráficas PNG en `results/plots/`.

## Requisitos no funcionales

### RNF-01 — Reproducibilidad
Resultados idénticos con la misma semilla en cualquier ejecución.

### RNF-02 — Correctitud del AUC
El AUC del W* óptimo debe estar en [0.5, 1.0] y la consistencia de scoring (ec. 4) ≥ 0.8.

### RNF-03 — Rendimiento
- Python multicore: speedup ≥ 1.5× vs secuencial.
- C + OpenMP / MPI (4 hilos/procesos): speedup ≥ 3×.
- CUDA: speedup ≥ 5×.

### RNF-04 — Portabilidad
- Python: compatible con 3.12+, dependencias en `pyproject.toml`.
- C: estándar C11, compilable con GCC en Linux/WSL2.
- CUDA: ejecutable en Google Colab (runtime GPU, CUDA Toolkit ≥ 11.0); no requiere GPU local.
