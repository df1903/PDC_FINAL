"""
Nivel 1 — Python Multicore: Random Search con multiprocessing.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool, cpu_count


def evaluate_batch(args):
    A, y, candidates = args
    best_auc = 0.0
    best_W = None
    for W in candidates:
        P = A.dot(W) if A.shape[1] == len(W) else A.mean(axis=1)
        auc = roc_auc_score(y, P)
        if auc > best_auc:
            best_auc = auc
            best_W = W
    return best_W, best_auc


def random_search_multicore(A: np.ndarray, y: np.ndarray, K: int = 100_000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    candidates = rng.dirichlet(np.ones(3), size=K).astype(np.float32)
    n_workers = cpu_count()
    chunks = np.array_split(candidates, n_workers)

    with Pool(n_workers) as pool:
        results = pool.map(evaluate_batch, [(A, y, c) for c in chunks])

    best_W, best_auc = max(results, key=lambda r: r[1])
    return best_W, best_auc


if __name__ == "__main__":
    A = np.load("../data/matrix_A.npy")
    y = np.load("../data/labels.npy")
    W, auc = random_search_multicore(A, y)
    print(f"Best W: {W}  |  AUC: {auc:.4f}")
