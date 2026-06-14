# Instructions

Eres el agente de código para el proyecto **Scoring Metagenómico HPC**, un sistema de optimización de alto rendimiento para determinar un vector de pesos **W** = (W₁, W₂, W₃) que maximice el AUC en clasificación binaria de muestras metagenómicas.

## Rol y enfoque

- Trabajas sobre un repositorio con implementaciones a tres niveles:
  - **Nivel 1** — Python (`sequential.py`, `multicore.py`): baseline y validación.
  - **Nivel 2** — C (`scoring_openmp.c`, `scoring_mpi.c`): paralelismo de CPU.
  - **Nivel 3** — CUDA (`scoring_kernel.cu`, `scoring_pycuda.py`): aceleración GPU.
- Antes de implementar, lee siempre: `context/.IA/rules.md`, `context/.IA/architecture.md`, `context/.IA/stack.md` y `context/.IA/constraints.md`.
- El código existente es la fuente de verdad. Si hay conflicto entre este contexto y el código real, prioriza el código.

## Flujo de trabajo por fases

El proyecto sigue un flujo secuencial. No saltar fases:

1. **Fase 1 — Python Baseline**: implementar y validar `sequential.py` y `multicore.py`.
2. **Fase 2 — C + OpenMP**: implementar `scoring_openmp.c` con paralelismo de memoria compartida.
3. **Fase 3 — C + MPI**: implementar `scoring_mpi.c` con memoria distribuida.
4. **Fase 4 — CUDA**: implementar `scoring_kernel.cu` y `scoring_pycuda.py`.
5. **Fase 5 — Benchmarking**: ejecutar `run_all.sh`, generar `results/benchmark.csv` y gráficas.
6. **Fase 6 — Documentación**: completar informe técnico en PDF (`report/`).

## Modo de trabajo

1. **Explorar antes de modificar**: leer los archivos relevantes antes de proponer cambios.
2. **Validación numérica**: toda nueva implementación debe producir AUC dentro de 1e-4 del Python secuencial con la misma semilla.
3. **Reproducibilidad**: usar `seed=42` en todas las implementaciones.
4. **No saltar fases**: cada nivel debe estar validado antes de avanzar.
5. **Commits atómicos**: `feat(python): ...`, `feat(openmp): ...`, `feat(mpi): ...`, `feat(cuda): ...`, `fix: ...`, `bench: ...`.
6. **No romper el baseline**: los archivos Python en `python/` son la referencia para correctitud.

## Respuestas

- Responde en español.
- Código Python conciso y tipado; C/CUDA documentado con comentarios de bloque.
- Confirma antes de modificar datos compartidos (`data/`) o el script de benchmark.
