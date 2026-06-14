# Constraints

## Dataset

- Matriz de contribución: `A ∈ ℝ^{10×N}` (float32, N=50 por defecto).
- Perfiles por ítem independientes: `T ∈ ℝ^N` (float32), `S ∈ ℝ^N` (float32), `F ∈ {0,1,2}^N` (int32).
- Etiquetas: `y ∈ {0,1}^{10}`; filas 0–4 sanas (y=0), filas 5–9 enfermas (y=1).
- Generado con `seed=42` vía `data/generate_data.py`, que inyecta señal diferencial en las filas enfermas de `A` (correlación con ítems de alto F/T) para garantizar separabilidad.
- `T`, `S`, `F` NO se obtienen particionando columnas de `A`; son entradas independientes del modelo.
- Para pruebas de escalabilidad de N: regenerar con `n_items` configurable (A, T, S, F escalan con N) y documentar.

## Vector de pesos W

- Espacio de búsqueda: simplex 3D → W₁ + W₂ + W₃ = 1, Wᵢ ≥ 0.
- Muestreo vía distribución Dirichlet(1,1,1) — uniforme sobre el simplex.
- `K_CANDIDATES = 100_000` por defecto; configurable para pruebas de escalabilidad.

## Correctitud y validación

- AUC de referencia: calculado por Python secuencial con `seed=42`, K=100k.
- Tolerancia de equivalencia entre implementaciones: |ΔAUC| < 1e-4.
- Consistencia de scoring ≥ 0.8 (ecuación 4 del proyecto) para el W* óptimo.
- AUC fuera de [0.5, 1.0] indica bug; corregir antes de continuar.

## Rendimiento objetivo

| Implementación | Speedup mínimo aceptable |
|----------------|--------------------------|
| Python multicore | ≥ 1.5× vs secuencial |
| C + OpenMP (4 hilos) | ≥ 3× vs Python secuencial |
| C + MPI (4 procesos) | ≥ 3× vs Python secuencial |
| CUDA | ≥ 5× vs Python secuencial |

## Entorno Fase 4 (CUDA)

- La Fase 4 se desarrolla en un **notebook `.ipynb` ejecutado en Google Colab** con runtime GPU (típicamente NVIDIA T4).
- El kernel CUDA C se materializa con `%%writefile` y se compila con `nvcc` dentro del notebook; la orquestación usa PyCUDA.
- No se asume GPU local: el entregado y la medición de speedup de CUDA provienen del runtime de Colab.

## Código

- Python: `requires-python = ">=3.12"` (pyproject.toml).
- C: estándar C11, compilar con GCC `-O2 -fopenmp` y `mpicc -O2`.
- CUDA: CUDA Toolkit ≥ 11.0 (el de Colab), compilar con `nvcc -O2`.
- No usar librerías externas en C sin agregar entrada al Makefile y justificar.

## Salidas

- `results/benchmark.csv` — fuente de verdad para métricas; generado por `run_all.sh`.
- `results/plots/` — gráficas PNG de speedup y eficiencia vs P.
- `report/informe_tecnico.pdf` — informe final.
