"""Funciones compartidas entre `sequential.py` y `multicore.py` (Fase 1).

Carga/validación de datos, muestreo de candidatos W sobre el simplex,
cálculo de Score/AUC/consistencia (ecs. 1-4 de `others/proyecto_final.md`)
y registro de métricas en `results/benchmark.csv` (esquema de 9 columnas,
ver `context/.IA/rules.md`).
"""

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if str(_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_DIR))

_generate_data = importlib.import_module("generate_data")
load_or_generate_dataset = _generate_data.load_or_generate_dataset
load_or_generate_profiles = _generate_data.load_or_generate_profiles

BENCHMARK_COLUMNS = [
    "implementation",
    "n_items",
    "k_candidates",
    "workers",
    "best_auc",
    "time_seconds",
    "candidates_per_second",
    "speedup",
    "efficiency",
]


def load_dataset(n_items: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga (o genera) A, profiles e y para `n_items` y `seed` dados."""
    A, y = load_or_generate_dataset(n_items, seed)
    profiles = load_or_generate_profiles(n_items, seed)
    return A, profiles, y


def validate_dataset(A: np.ndarray, profiles: np.ndarray, y: np.ndarray, n_items: int) -> None:
    """Valida shapes, dtypes y rangos de A, profiles e y (DEC-10/rules.md)."""
    if A.shape != (10, n_items) or A.dtype != np.float32:
        raise ValueError(f"A inválida: shape={A.shape}, dtype={A.dtype}")
    if profiles.shape != (n_items, 3) or profiles.dtype != np.float32:
        raise ValueError(f"profiles inválido: shape={profiles.shape}, dtype={profiles.dtype}")
    if y.shape != (10,) or y.dtype != np.int32:
        raise ValueError(f"y inválido: shape={y.shape}, dtype={y.dtype}")
    if not np.array_equal(y, np.array([0] * 5 + [1] * 5, dtype=np.int32)):
        raise ValueError(f"y debe ser [0]*5 + [1]*5, recibido {y}")

    T, S, F = profiles[:, 0], profiles[:, 1], profiles[:, 2]
    if not (np.all((T >= 0) & (T <= 1)) and np.all((S >= 0) & (S <= 1))):
        raise ValueError("T y S deben estar en [0, 1]")
    if not np.all(np.isin(F, [0, 1, 2])):
        raise ValueError("F debe contener únicamente valores en {0, 1, 2}")


def sample_candidates(K: int, seed: int = 42) -> np.ndarray:
    """Muestrea K vectores W ~ Dirichlet(1,1,1) sobre el simplex 3D (ec. 5)."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(3), size=K).astype(np.float32)


def score_samples(A: np.ndarray, profiles: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Calcula Score = A @ P con P = W1*T + W2*S + W3*F (ecs. 1-2)."""
    T, S, F = profiles[:, 0], profiles[:, 1], profiles[:, 2]
    P = W[0] * T + W[1] * S + W[2] * F
    return A @ P


def evaluate_candidates(
    A: np.ndarray, profiles: np.ndarray, y: np.ndarray, candidates: np.ndarray
) -> tuple[np.ndarray, float]:
    """Evalúa cada W en `candidates` (ec. 3) y retorna (best_W, best_auc)."""
    best_auc = -1.0
    best_W = candidates[0]
    for W in candidates:
        score = score_samples(A, profiles, W)
        auc = roc_auc_score(y, score)
        if auc > best_auc:
            best_auc = auc
            best_W = W
    return best_W, best_auc


def scoring_consistency(y: np.ndarray, scores: np.ndarray, theta: float | None = None) -> float:
    """Consistencia de scoring (ec. 4): TP/5 + TN/5 para el umbral `theta`."""
    if theta is None:
        theta = float(np.median(scores))
    true_positive = np.sum((scores > theta) & (y == 1))
    true_negative = np.sum((scores <= theta) & (y == 0))
    return float(true_positive / 5 + true_negative / 5)


def compute_metrics(K: int, time_s: float, workers: int, t_seq: float | None = None) -> dict:
    """Calcula candidates_per_second y, si `t_seq` está disponible, speedup/efficiency."""
    metrics: dict = {"candidates_per_second": K / time_s}
    if t_seq is not None:
        speedup = t_seq / time_s
        metrics["speedup"] = speedup
        metrics["efficiency"] = speedup / workers
    else:
        metrics["speedup"] = None
        metrics["efficiency"] = None
    return metrics


def append_benchmark(row: dict, csv_path: Path) -> None:
    """Agrega `row` a `csv_path` (esquema BENCHMARK_COLUMNS); crea cabecera si falta."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BENCHMARK_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_sequential_time(csv_path: Path, n_items: int, k_candidates: int) -> float | None:
    """Retorna `time_seconds` de la última fila 'Python secuencial' que coincida en
    `n_items`/`k_candidates`, o None si no existe."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    last_time = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["implementation"] == "Python secuencial"
                and int(row["n_items"]) == n_items
                and int(row["k_candidates"]) == k_candidates
            ):
                last_time = float(row["time_seconds"])
    return last_time
