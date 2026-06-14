# Current Phase

**Fase activa**: Fase 1 — Python Baseline (`context/project/phases.md`).

**Estado**: planificación técnica completada y confirmada por el usuario (2026-06-14). El código en `code/` permanece **sin modificar** todavía; la implementación queda pendiente para la próxima conversación.

## Resumen del plan confirmado

- **DEC-11** (nueva): se autoriza modificar `code/data/generate_data.py` para inyectar señal diferencial en las filas enfermas de `A` y regenerar `data/n_50/` y `data/n_100/` (ver `context/project/decisions.md`).
- Crear `code/python/common.py` (módulo compartido) y reescribir `code/python/sequential.py` y `code/python/multicore.py` (actualmente son placeholders no funcionales — ver `context/state/known-issues.md`).
- Migrar `code/results/benchmark.csv` al esquema de 9 columnas: `implementation,n_items,k_candidates,workers,best_auc,time_seconds,candidates_per_second,speedup,efficiency`.
- Firmas canónicas confirmadas: `random_search(A, profiles, y, K=100_000, seed=42)` y `random_search_multicore(A, profiles, y, K=100_000, seed=42, workers=cpu_count())`.
- CLI: `--n-items` (50), `--k-candidates` (100000), `--workers` (cpu_count()), `--seed` (42).

Detalle completo de tareas: `context/state/active-tasks.md`.

## Próxima acción

Ejecutar la implementación de la Fase 1 siguiendo `context/state/active-tasks.md`, en el orden ahí indicado, validando cada paso (AUC ∈ [0.5,1], consistencia ≥ 0.8, equivalencia secuencial↔multicore < 1e-4, speedup ≥ 1.5×) antes de avanzar a la Fase 2.
