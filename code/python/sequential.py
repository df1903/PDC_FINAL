"""
Nivel 1 — Python Secuencial: Random Search sobre el simplex.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def random_search(A: np.ndarray, y: np.ndarray, K: int = 100_000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    best_auc = 0.0
    best_W = None

    for _ in range(K):
        W = rng.dirichlet(np.ones(3)).astype(np.float32)
        # Placeholder: T, S, F profiles assumed equal to A columns partitioned
        P = A @ W[:len(A[0])] if len(A[0]) == 3 else A.mean(axis=1)
        auc = roc_auc_score(y, P)
        if auc > best_auc:
            best_auc = auc
            best_W = W

    return best_W, best_auc


if __name__ == "__main__":
    A = np.load("../data/matrix_A.npy")
    y = np.load("../data/labels.npy")
    W, auc = random_search(A, y)
    print(f"Best W: {W}  |  AUC: {auc:.4f}")
