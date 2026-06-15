"""Nivel 1 — Python Multicore: Random Search sobre el simplex con `multiprocessing.Pool`.

Ejecutar desde `code/`:
    uv run python/multicore.py --n-items 50 --k-candidates 100000 --workers 4 --seed 42
"""

import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from common import (
    append_benchmark,
    compute_metrics,
    evaluate_candidates,
    load_dataset,
    read_sequential_time,
    sample_candidates,
    score_samples,
    scoring_consistency,
    validate_dataset,
)

BENCHMARK_CSV = Path("results") / "benchmark.csv"


def _evaluate_chunk(
    args: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, float]:
    """Función picklable top-level usada por `Pool.map` para evaluar un chunk."""
    A, profiles, y, chunk = args
    return evaluate_candidates(A, profiles, y, chunk)


def random_search_multicore(
    A: np.ndarray,
    profiles: np.ndarray,
    y: np.ndarray,
    K: int = 100_000,
    seed: int = 42,
    workers: int = cpu_count(),
) -> tuple[np.ndarray, float]:
    """Random Search multicore: genera K candidatos una vez y los reparte en `workers`."""
    candidates = sample_candidates(K, seed)
    chunks = np.array_split(candidates, workers)
    with Pool(workers) as pool:
        results = pool.map(_evaluate_chunk, [(A, profiles, y, chunk) for chunk in chunks])
    return max(results, key=lambda r: r[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Nivel 1 - Python multicore (Random Search)")
    parser.add_argument("--n-items", type=int, default=50)
    parser.add_argument("--k-candidates", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    A, profiles, y = load_dataset(args.n_items, args.seed)
    validate_dataset(A, profiles, y, args.n_items)

    start = time.perf_counter()
    best_W, best_auc = random_search_multicore(
        A, profiles, y, K=args.k_candidates, seed=args.seed, workers=args.workers
    )
    time_seconds = time.perf_counter() - start

    consistency = scoring_consistency(y, score_samples(A, profiles, best_W))

    t_seq = read_sequential_time(BENCHMARK_CSV, args.n_items, args.k_candidates)
    if t_seq is None:
        print(
            "[WARN] No se encontró una fila 'Python secuencial' en benchmark.csv "
            "para este n_items/k_candidates; speedup/efficiency quedarán vacíos."
        )
    # "Python secuencial" es el baseline P=1 de la familia Python (DEC-13): t_self_base
    # y t_python_seq coinciden para esta implementación.
    metrics = compute_metrics(
        args.k_candidates,
        time_seconds,
        workers=args.workers,
        t_self_base=t_seq,
        t_python_seq=t_seq,
    )

    append_benchmark(
        {
            "implementation": "Python multicore",
            "n_items": args.n_items,
            "k_candidates": args.k_candidates,
            "workers": args.workers,
            "best_auc": best_auc,
            "time_seconds": time_seconds,
            "candidates_per_second": metrics["candidates_per_second"],
            "speedup": metrics["speedup"],
            "efficiency": metrics["efficiency"],
            "speedup_vs_python": metrics["speedup_vs_python"],
        },
        BENCHMARK_CSV,
    )

    print(f"Best W: {best_W}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Consistencia: {consistency:.4f}")
    print(f"Tiempo: {time_seconds:.4f} s")
    if metrics["speedup"] is not None:
        print(f"Speedup: {metrics['speedup']:.4f} | Efficiency: {metrics['efficiency']:.4f}")


if __name__ == "__main__":
    main()
