# Phases

## Fase 1 — Python Baseline `[PENDIENTE]`

Objetivo: implementación de referencia correcta y validada según el modelo de perfiles (DEC-07).

- [ ] Regenerar datos con `generate_data.py`: `A (10×N)`, `T`, `S`, `F (∈{0,1,2})`, `y`, con señal diferencial.
- [ ] Reimplementar `python/sequential.py` con `P = W₁T+W₂S+W₃F` y `Score = A·P` (carga T/S/F desde `.npy`).
- [ ] Reimplementar `python/multicore.py` con el mismo modelo (Pool sobre candidatos W).
- [ ] Validar AUC ∈ [0.5, 1.0] y consistencia ≥ 0.8.
- [ ] Medir T_secuencial como baseline para speedup.
- [ ] Registrar métricas en `results/benchmark.csv`.

## Fase 2 — C + OpenMP `[PENDIENTE]`

Objetivo: paralelismo de memoria compartida sobre CPU.

- [ ] Implementar carga de datos en `scoring_openmp.c`.
- [ ] Implementar Random Search con `#pragma omp parallel for`.
- [ ] Validar equivalencia AUC con Fase 1.
- [ ] Medir speedup vs Python secuencial con P ∈ {1, 2, 4, 8}.

## Fase 3 — C + MPI `[PENDIENTE]`

Objetivo: paralelismo de memoria distribuida.

- [ ] Implementar `MPI_Scatter` de candidatos desde root.
- [ ] Implementar evaluación local y `MPI_Reduce(MPI_MAX)`.
- [ ] Validar equivalencia AUC.
- [ ] Medir speedup con P ∈ {1, 2, 4, 8} procesos.

## Fase 4 — CUDA (notebook en Google Colab) `[PENDIENTE]`

Objetivo: aceleración GPU masiva en `CUDA/scoring_cuda.ipynb`, ejecutado en Colab.

- [ ] Crear `scoring_cuda.ipynb` con celdas: setup/GPU → `%%writefile scoring_kernel.cu` → compilación `nvcc` → orquestación PyCUDA.
- [ ] Implementar kernel `scoring_kernel` (un hilo por candidato W_k; calcula `P=W·[T,S,F]` y `Score=A·P`).
- [ ] Usar `__shared__` memory para filas de A por bloque.
- [ ] Implementar reduction kernel para AUC máximo.
- [ ] Subir los `.npy` de `code/data/` al runtime de Colab.
- [ ] Medir speedup vs Python secuencial y registrar el modelo de GPU usado.

## Fase 5 — Benchmarking `[PENDIENTE]`

Objetivo: consolidar métricas y generar reportes.

- [ ] Implementar `run_all.sh`.
- [ ] Generar `results/benchmark.csv` con todas las métricas.
- [ ] Generar gráficas PNG de Speedup y Eficiencia vs P.
- [ ] Calcular f empírico para Ley de Amdahl.

## Fase 6 — Documentación `[PENDIENTE]`

Objetivo: informe técnico final.

- [ ] Redactar informe con estrategias de sincronización por nivel.
- [ ] Incluir análisis de gestión de memoria.
- [ ] Incluir gráficas y análisis de Amdahl con f empírico.
- [ ] Discutir separabilidad de grupos mediante score óptimo.
