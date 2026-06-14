"""Nivel 1 — Python Secuencial: Random Search sobre el simplex (ecs. 1-5).

Ejecutar desde `code/`:
    uv run python/sequential.py --n-items 50 --k-candidates 100000 --seed 42
"""

import argparse
import time
from pathlib import Path

import numpy as np

from common import (
    append_benchmark,
    compute_metrics,
    evaluate_candidates,
    load_dataset,
    sample_candidates,
    score_samples,
    scoring_consistency,
    validate_dataset,
)

BENCHMARK_CSV = Path("results") / "benchmark.csv"


def random_search(
    A: np.ndarray, profiles: np.ndarray, y: np.ndarray, K: int = 100_000, seed: int = 42
) -> tuple[np.ndarray, float]:
    """Random Search secuencial: muestrea K candidatos W y retorna (best_W, best_auc)."""
    candidates = sample_candidates(K, seed)
    return evaluate_candidates(A, profiles, y, candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nivel 1 - Python secuencial (Random Search)")
    parser.add_argument("--n-items", type=int, default=50)
    parser.add_argument("--k-candidates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    A, profiles, y = load_dataset(args.n_items, args.seed)
    validate_dataset(A, profiles, y, args.n_items)

    start = time.perf_counter()
    best_W, best_auc = random_search(A, profiles, y, K=args.k_candidates, seed=args.seed)
    time_seconds = time.perf_counter() - start

    consistency = scoring_consistency(y, score_samples(A, profiles, best_W))
    metrics = compute_metrics(args.k_candidates, time_seconds, workers=1, t_seq=time_seconds)

    append_benchmark(
        {
            "implementation": "Python secuencial",
            "n_items": args.n_items,
            "k_candidates": args.k_candidates,
            "workers": 1,
            "best_auc": best_auc,
            "time_seconds": time_seconds,
            "candidates_per_second": metrics["candidates_per_second"],
            "speedup": metrics["speedup"],
            "efficiency": metrics["efficiency"],
        },
        BENCHMARK_CSV,
    )

    print(f"Best W: {best_W}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Consistencia: {consistency:.4f}")
    print(f"Tiempo: {time_seconds:.4f} s")


if __name__ == "__main__":
    main()
