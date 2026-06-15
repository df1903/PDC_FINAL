# Current Phase

**Fase activa**: Fase 2 — C + OpenMP (`context/project/phases.md`). **Plan técnico definido**
(2026-06-15, ver `context/state/active-tasks.md` y `traceability_data/2026_06_15_14-24.md`);
pendiente de implementar.

**Fase 1 — Python Baseline**: `COMPLETADA` (2026-06-14). Implementada y validada según
`context/state/active-tasks.md`; detalle en `traceability_data/2026_06_14_17-18.md`.

## Resultados Fase 1 (seed=42, n_items=50, K=100000, desde `code/`)

| Implementación        | best_auc | time_seconds | candidates_per_second | speedup | efficiency |
|------------------------|----------|--------------|------------------------|---------|------------|
| Python secuencial       | 1.0000   | 84.3913      | 1184.96                | 1.00    | 1.00       |
| Python multicore (w=4)  | 1.0000   | 26.6237      | 3756.06                | 3.1698  | 0.7924     |

- `best_W = [0.33742523, 0.32787895, 0.33469582]` — idéntico en ambas implementaciones.
- Consistencia (ec. 4) = 2.0000 en ambas (≥ 0.8 ✓).
- `|auc_multicore − auc_secuencial| = 0 < 1e-4` ✓ (RF-04).
- `speedup_multicore = 3.1698× ≥ 1.5×` ✓ (RNF-03).
- `pytest python/tests/ -q` → 5 passed. `ruff check python/` y `ruff check data/` → sin errores.
- `data/n_50/` y `data/n_100/` regenerados con la señal diferencial de DEC-11 (seed=42);
  verificado AUC > 0.6 (de hecho 1.0000) para ambos tamaños antes de escribir el código de
  Fase 1.

## Cambios relevantes

- `code/data/generate_data.py`: DEC-11 implementado — filas enfermas (5-9) de `A` usan
  Dirichlet sesgada por `importance = T + F` (`SIGNAL_STRENGTH = 8.0`); filas sanas usan
  Dirichlet uniforme. `generate_profiles` sin cambios.
- Nuevo `code/python/common.py` (140 LOC) con las 9 funciones compartidas de
  `active-tasks.md`.
- `code/python/sequential.py` y `code/python/multicore.py` reescritos con las firmas
  canónicas (`rules.md`) y CLI requerida. CLI de `sequential.py` no incluye `--workers`
  (no está en la firma canónica ni en la CLI obligatoria de `rules.md`; `workers=1` fijo).
- `code/results/benchmark.csv` migrado al esquema de 9 columnas (append-only); filas
  plantilla antiguas (sin datos reales) reemplazadas.
- `code/python/tests/test_baseline.py` (61 LOC, 5 tests) + `pyproject.toml` actualizado con
  `[tool.pytest.ini_options] pythonpath = ["python"]` para que los tests importen
  `common`/`sequential`/`multicore` sin hacks de `sys.path` (evita E402 en ruff).
- ISSUE-007: `code/pyproject.toml` y `code/uv.lock` ya existían (con numpy/scikit-learn/
  matplotlib + pytest/ruff como dev deps) — no fue necesario crearlos, solo se agregó la
  configuración de pytest.

## Plan de Fase 2 (definido 2026-06-15)

Detalle completo en `context/state/active-tasks.md`. Resumen:

- **Archivos**: crear `code/C_OpenMP_MPI/npy_io.{h,c}` (parser `.npy` mínimo, sin libs externas,
  DEC-06/DEC-10); reescribir `scoring_openmp.c` (< 400 LOC); ajustar `Makefile` (añadir
  `npy_io.c`, `-Wextra`; target `mpi` intacto).
- **Carga**: parser binario directo de `data/n_{N}/{matrix_A,profiles,labels}.npy` (formato v1.0
  verificado con `xxd`), CWD = `code/`. No se modifican los `.npy`.
- **RNG**: SplitMix64 sembrado por candidato (`seed+k`) → Dirichlet(1,1,1); `W_k` depende solo
  de `k` → `best_auc` determinista para P ∈ {1,2,4,8}, sin sección crítica en el muestreo.
- **AUC corregida** (RIESGO-03/ISSUE-008): `(concordantes + 0.5·empates)/(pos·neg)` para igualar
  `roc_auc_score`; validación en dos capas (`--self-test` de la función AUC + equivalencia de
  `best_auc` vs Python secuencial, viable por separabilidad AUC=1.0).
- **OpenMP**: best local por hilo + `#pragma omp critical` de fusión; `omp_get_wtime()` solo
  sobre la búsqueda (excluye carga).
- **CLI**: `--n-items/--k-candidates/--seed/--threads/--self-test`. Registro `C OpenMP` en
  `results/benchmark.csv` (9 columnas, append-only; `t_seq` leído de la fila `Python secuencial`).

## Próxima acción

Implementar el plan de `context/state/active-tasks.md` (Fase 2). Criterios de salida:
`--self-test` OK, `|ΔAUC| < 1e-4` vs Python secuencial (seed=42, K=100k), `best_auc ∈ [0.5,1.0]`,
consistencia ≥ 0.8 y `speedup ≥ 3×` con P=4 (RNF-03), con curva P ∈ {1,2,4,8}.
