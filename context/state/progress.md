# Progress

## Completado

- [x] Estructura de repositorio creada (`code/`, `context/`, `docs/`, etc.)
- [x] `context/` documentado con el modelo de perfiles confirmado (T/S/F independientes, A matriz de contribución; ver `context/project/decisions.md`).
- [x] Esqueletos de código presentes en `code/` (Python, C/OpenMP/MPI, CUDA) según `context/.IA/directory-structure.md`.
- [x] Fase 1 (Python Baseline) implementada y validada (2026-06-14, ver `context/state/current-phase.md` y `traceability_data/2026_06_14_17-18.md`): DEC-11 (señal diferencial + regeneración `data/n_50/`/`data/n_100/`), `code/python/common.py`, `sequential.py`, `multicore.py` reescritos, `results/benchmark.csv` migrado al esquema de 9 columnas, `code/python/tests/test_baseline.py` (5 tests OK), `ruff check` sin errores.
- [x] Fase 2 (C + OpenMP) implementada y validada (2026-06-15, ver `context/state/current-phase.md` y `traceability_data/2026_06_15_14-36.md`): `npy_io.{h,c}` (parser `.npy` propio), `scoring_openmp.c` reescrito (RNG SplitMix64 por candidato, AUC con empates, Random Search OpenMP, CLI, `--self-test`), `Makefile` ajustado. `best_auc=1.0000` (=Fase 1, `|ΔAUC|=0`), consistencia=2.0000, `valgrind` sin fugas propias.
- [x] DEC-13 (2026-06-15, `traceability_data/2026_06_15_15-24.md`, ISSUE-009): corregidas las métricas `speedup`/`efficiency` de "C OpenMP" para que sean relativas al baseline propio `workers=1` (`speedup(P=4)=3.84× ≥ 3×`), y se agregó la columna `speedup_vs_python` (3136×–16930×) para la comparación C vs Python que antes ocupaba indebidamente `speedup`. `benchmark.csv` 9→10 columnas, `scoring_openmp.c` exactamente 400 LOC, `test_baseline.py` 8/8 OK.

## En progreso

- [ ] Fase 3 (C + MPI): pendiente de planificar/implementar `scoring_mpi.c` (`MPI_Scatter`/`MPI_Reduce`), reutilizando `npy_io.{h,c}` de Fase 2.

## Pendiente

- [ ] Fases 3 a 6 — pendientes, ver `context/project/phases.md`.
