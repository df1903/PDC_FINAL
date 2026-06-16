# Progress

## Completado

- [x] Estructura de repositorio creada (`code/`, `context/`, `docs/`, etc.)
- [x] `context/` documentado con el modelo de perfiles confirmado (T/S/F independientes, A matriz de contribución; ver `context/project/decisions.md`).
- [x] Esqueletos de código presentes en `code/` (Python, C/OpenMP/MPI, CUDA) según `context/.IA/directory-structure.md`.
- [x] Fase 1 (Python Baseline) implementada y validada (2026-06-14, ver `context/state/current-phase.md` y `traceability_data/2026_06_14_17-18.md`): DEC-11 (señal diferencial + regeneración `data/n_50/`/`data/n_100/`), `code/python/common.py`, `sequential.py`, `multicore.py` reescritos, `results/benchmark.csv` migrado al esquema de 9 columnas, `code/python/tests/test_baseline.py` (5 tests OK), `ruff check` sin errores.
- [x] Fase 2 (C + OpenMP) implementada y validada (2026-06-15, ver `context/state/current-phase.md` y `traceability_data/2026_06_15_14-36.md`): `npy_io.{h,c}` (parser `.npy` propio), `scoring_openmp.c` reescrito (RNG SplitMix64 por candidato, AUC con empates, Random Search OpenMP, CLI, `--self-test`), `Makefile` ajustado. `best_auc=1.0000` (=Fase 1, `|ΔAUC|=0`), consistencia=2.0000, `valgrind` sin fugas propias.
- [x] DEC-13 (2026-06-15, `traceability_data/2026_06_15_15-24.md`, ISSUE-009): corregidas las métricas `speedup`/`efficiency` de "C OpenMP" para que sean relativas al baseline propio `workers=1` (`speedup(P=4)=3.84× ≥ 3×`), y se agregó la columna `speedup_vs_python` (3136×–16930×) para la comparación C vs Python que antes ocupaba indebidamente `speedup`. `benchmark.csv` 9→10 columnas, `scoring_openmp.c` exactamente 400 LOC, `test_baseline.py` 8/8 OK.
- [x] Fase 3 (C + MPI) implementada y validada (2026-06-15, ver `context/state/current-phase.md` y `traceability_data/2026_06_15_19-32.md`): `scoring_mpi.c` (389 LOC, copia literal del cómputo de Fase 2; carga en root + `MPI_Bcast`; regeneración local de candidatos; `MPI_MAXLOC`; timing `MPI_Wtime`+`MPI_Barrier`), `Makefile` actualizado, 4 filas `C MPI` en `benchmark.csv`. `best_auc=1.0000` (`|ΔAUC|=0`), consistencia=2.0000, `speedup(P=4)=3.65× ≥ 3×`.
- [x] Fase 4 (CUDA) implementada — **código completo, corrida en Colab pendiente** (2026-06-16, ver `context/state/current-phase.md#Fase 4` y `traceability_data/2026_06_16_00-57.md`): `scoring_kernel.cu` (125 LOC, kernel real + RNG SplitMix64 `__device__`), `scoring_pycuda.py` (160 LOC, orquestación reusable), `scoring_cuda.ipynb` (28 celdas, entregable Colab). Validado localmente **sin GPU**: LOC, sintaxis, RNG bit-idéntico a `scoring_openmp.c`, helpers de host sobre `data/n_50/` → AUC=1.0/consistencia=2.0/empate=0.875.

## En progreso

- [ ] Fase 4 (CUDA): **ejecutar `scoring_cuda.ipynb` en Google Colab** (runtime GPU), registrar el modelo de GPU y `t_search`, verificar `speedup_vs_python ≥ 5×` (RNF-03), y transcribir la fila `CUDA` a `results/benchmark.csv` (append-only) para marcar la fase como COMPLETADA.

## Pendiente

- [ ] Fases 5 y 6 — pendientes, ver `context/project/phases.md` (Benchmarking y Documentación).
