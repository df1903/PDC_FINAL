"""Pruebas de correctitud del baseline Python (Fase 1).

Ejecutar desde `code/`:
    pytest python/tests/ -q
"""

import numpy as np

from common import load_dataset, score_samples, scoring_consistency
from multicore import random_search_multicore
from sequential import random_search

N_ITEMS = 50
K_SMALL = 200


def test_equivalencia_secuencial_multicore_caso_chico():
    """N=3, K=100: mismos candidatos -> mismo best_W y AUC exacto (RIESGO-03)."""
    rng = np.random.default_rng(7)
    n_items = 3
    A = rng.dirichlet(np.ones(n_items), size=10).astype(np.float32)
    profiles = np.column_stack(
        [rng.random(n_items), rng.random(n_items), rng.integers(0, 3, size=n_items)]
    ).astype(np.float32)
    y = np.array([0] * 5 + [1] * 5, dtype=np.int32)

    seq_W, seq_auc = random_search(A, profiles, y, K=100, seed=42)
    mc_W, mc_auc = random_search_multicore(A, profiles, y, K=100, seed=42, workers=2)

    assert np.array_equal(seq_W, mc_W)
    assert seq_auc == mc_auc


def test_equivalencia_secuencial_multicore_n50():
    """N=50: AUC secuencial y multicore deben coincidir dentro de 1e-4 (RF-04)."""
    A, profiles, y = load_dataset(N_ITEMS, seed=42)

    _, seq_auc = random_search(A, profiles, y, K=K_SMALL, seed=42)
    _, mc_auc = random_search_multicore(A, profiles, y, K=K_SMALL, seed=42, workers=2)

    assert abs(seq_auc - mc_auc) < 1e-4


def test_best_auc_en_rango_valido():
    A, profiles, y = load_dataset(N_ITEMS, seed=42)
    _, best_auc = random_search(A, profiles, y, K=K_SMALL, seed=42)
    assert 0.5 <= best_auc <= 1.0


def test_consistencia_minima():
    A, profiles, y = load_dataset(N_ITEMS, seed=42)
    best_W, _ = random_search(A, profiles, y, K=K_SMALL, seed=42)
    consistency = scoring_consistency(y, score_samples(A, profiles, best_W))
    assert consistency >= 0.8


def test_best_w_en_simplex():
    A, profiles, y = load_dataset(N_ITEMS, seed=42)
    best_W, _ = random_search(A, profiles, y, K=K_SMALL, seed=42)
    assert np.all(best_W >= 0)
    assert abs(float(best_W.sum()) - 1.0) < 1e-5
