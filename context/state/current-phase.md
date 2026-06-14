# Current Phase

**Fase activa**: Fase 2 — C + OpenMP (`context/project/phases.md`), aún sin iniciar.

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

## Próxima acción

Iniciar Fase 2 (`context/project/phases.md`): implementar `code/C_OpenMP_MPI/scoring_openmp.c`
(carga de datos, Random Search con `#pragma omp parallel for`), validar equivalencia AUC con
Fase 1 (|ΔAUC| < 1e-4, mismo seed=42/K) y medir speedup vs Python secuencial con P ∈ {1,2,4,8}.
Definir el plan detallado de Fase 2 en `context/state/active-tasks.md` antes de implementar.
