# Progress

## Completado

- [x] Estructura de repositorio creada (`code/`, `context/`, `docs/`, etc.)
- [x] `context/` documentado con el modelo de perfiles confirmado (T/S/F independientes, A matriz de contribución; ver `context/project/decisions.md`).
- [x] Esqueletos de código presentes en `code/` (Python, C/OpenMP/MPI, CUDA) según `context/.IA/directory-structure.md`.
- [x] Fase 1 (Python Baseline) implementada y validada (2026-06-14, ver `context/state/current-phase.md` y `traceability_data/2026_06_14_17-18.md`): DEC-11 (señal diferencial + regeneración `data/n_50/`/`data/n_100/`), `code/python/common.py`, `sequential.py`, `multicore.py` reescritos, `results/benchmark.csv` migrado al esquema de 9 columnas, `code/python/tests/test_baseline.py` (5 tests OK), `ruff check` sin errores.

## En progreso

- [ ] Fase 2 (C + OpenMP): no iniciada. Próximo paso: definir plan detallado en `context/state/active-tasks.md`.

## Pendiente

- [ ] Fases 2 a 6 — pendientes, ver `context/project/phases.md`.
