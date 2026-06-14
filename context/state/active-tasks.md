# Active Tasks — Fase 1 (Python Baseline)

Plan confirmado el 2026-06-14 (ver traceability `traceability_data/2026_06_14_16-42.md`). Ejecutar en este orden; cada tarea valida la anterior.

## 1. DEC-11 — Señal diferencial en `generate_data.py`

- Modificar `code/data/generate_data.py` (`generate_data` y, si aplica, `generate_profiles`) para que las filas enfermas (índices 5-9) de `A` correlacionen con ítems de `T`/`F` altos, de modo que exista un `W` en el simplex con AUC alto.
- Mantener shapes/dtypes (DEC-10): `matrix_A.npy` (10,N) f32, `profiles.npy` (N,3) f32, `labels.npy` (10,) i32.
- Regenerar (no reutilizar) `data/n_50/` y `data/n_100/` con `seed=42`. Documentar el cambio en el docstring del módulo.
- Verificar manualmente tras regenerar: con algún `W` del simplex, `roc_auc_score(y, A @ (W1*T+W2*S+W3*F)) > 0.6` (si no, intensificar señal — RIESGO-04).

## 2. Crear `code/python/common.py`

Módulo compartido entre `sequential.py` y `multicore.py` (máx. 200 LOC, type hints en firmas públicas). Importa los loaders de `code/data/generate_data.py`.

Funciones:

```python
def load_dataset(n_items: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # retorna (A, profiles, y) vía load_or_generate_dataset / load_or_generate_profiles

def validate_dataset(A, profiles, y, n_items: int) -> None
    # valida shapes, dtypes, y == [0]*5+[1]*5, F en {0,1,2}, T/S en [0,1]

def sample_candidates(K: int, seed: int = 42) -> np.ndarray
    # rng.dirichlet(np.ones(3), size=K)

def score_samples(A, profiles, W) -> np.ndarray
    # P = W[0]*T + W[1]*S + W[2]*F ; return A @ P  (Score, dim 10)

def evaluate_candidates(A, profiles, y, candidates) -> tuple[np.ndarray, float]
    # itera candidatos, retorna (best_W, best_auc)

def scoring_consistency(y, scores, theta: float | None = None) -> float
    # ecuación 4 del enunciado; theta por defecto = mediana(scores)

def compute_metrics(K: int, time_s: float, workers: int, t_seq: float | None = None) -> dict
    # candidates_per_second, speedup, efficiency

def append_benchmark(row: dict, csv_path) -> None
    # crea cabecera si no existe; append-only

def read_sequential_time(csv_path, n_items: int, k_candidates: int) -> float | None
    # última fila "Python secuencial" que coincida en n_items/k_candidates
```

## 3. Reescribir `code/python/sequential.py`

- `random_search(A, profiles, y, K=100_000, seed=42) -> tuple[np.ndarray, float]` usando `common.py`.
- CLI (`argparse`): `--n-items` (50), `--k-candidates` (100000), `--seed` (42). (`--workers` se acepta pero se ignora o fuerza a 1, para CLI uniforme — confirmar si se incluye.)
- Cargar dataset (`load_dataset` + `validate_dataset`) **fuera** del cronómetro.
- Medir `time_seconds` con `perf_counter()` solo alrededor de `random_search` (muestreo + evaluación).
- Calcular métricas: `implementation="Python secuencial"`, `workers=1`, `speedup=1.00`, `efficiency=1.00`, `candidates_per_second=K/time_seconds`.
- `append_benchmark(...)` a `results/benchmark.csv`.
- Imprimir `best_W`, `best_auc`, consistencia.

## 4. Reescribir `code/python/multicore.py`

- `random_search_multicore(A, profiles, y, K=100_000, seed=42, workers=cpu_count()) -> tuple[np.ndarray, float]`.
- Generar `candidates = sample_candidates(K, seed)` **una sola vez** en el proceso principal (determinismo independiente de `workers`).
- `np.array_split(candidates, workers)` + `multiprocessing.Pool(workers)` evaluando cada chunk con una función top-level picklable (necesario en Windows/spawn); reducir con `max(..., key=auc)`.
- CLI: `--n-items` (50), `--k-candidates` (100000), `--workers` (cpu_count()), `--seed` (42).
- Timing igual que en `sequential.py` (excluye carga).
- `t_seq = read_sequential_time(csv, n_items, K)`; `speedup = t_seq/time_seconds` si existe, si no deja vacío y avisa por consola.
- `efficiency = speedup / workers`. `append_benchmark(...)`.

## 5. Migrar `code/results/benchmark.csv`

- Esquema nuevo: `implementation,n_items,k_candidates,workers,best_auc,time_seconds,candidates_per_second,speedup,efficiency`.
- El archivo actual tiene cabecera/filas antiguas (`implementacion,T_s,speedup,eficiencia,AUC`) que son plantilla, no resultados reales — se reemplaza la cabecera por la nueva (append-only desde ese punto en adelante).

## 6. Pruebas (`pytest`)

- `code/python/tests/test_baseline.py` (o `code/tests/`):
  - Dataset chico (`n_items=3`, `K=100`): AUC secuencial == AUC multicore (exacto, mismos candidatos).
  - `best_auc ∈ [0.5, 1.0]` para `n_items=50`.
  - `scoring_consistency(...) >= 0.8` para el `best_W` de `n_items=50`.
  - Validar `best_W.sum() ≈ 1` y `best_W >= 0`.
- **Nota (ISSUE-007)**: no existe `code/pyproject.toml`. Verificar si `numpy`/`scikit-learn`/`pytest` están disponibles en el entorno antes de añadir tests; si falta `pyproject.toml`, crearlo declarando las dependencias ya usadas (no son "nuevas", solo había que declararlas — confirmar con el usuario antes de crear el archivo).

## 7. Verificación final (desde `code/`)

```bash
python data/generate_data.py
python python/sequential.py --n-items 50 --k-candidates 100000 --seed 42
python python/multicore.py --n-items 50 --k-candidates 100000 --workers 4 --seed 42
pytest python/tests/ -q
ruff check python/
cat results/benchmark.csv
```

Criterios de salida de Fase 1 (no avanzar a Fase 2 sin esto):
- `best_auc ∈ [0.5, 1.0]` y consistencia ≥ 0.8 en ambas implementaciones.
- `|auc_multicore - auc_secuencial| < 1e-4` (idealmente exacto).
- `speedup_multicore ≥ 1.5×` (RNF-03). Si no se alcanza, documentar en `context/state/known-issues.md` y discutir mitigación (no bloquea necesariamente avanzar, pero debe quedar registrado).
- `results/benchmark.csv` contiene ambas filas con el esquema nuevo, sin perder filas previas relevantes.
