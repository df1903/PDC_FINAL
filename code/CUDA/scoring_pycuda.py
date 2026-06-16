"""Nivel 3 — PyCUDA: orquestacion reusable del kernel CUDA de scoring (Fase 4, DEC-15).

Fuente derivada (DEC-09): equivale a las celdas 4-7 de `CUDA/scoring_cuda.ipynb`.
Lee `data/n_{n_items}/{matrix_A,profiles,labels}.npy` (DEC-10, solo lectura), compila
`scoring_kernel.cu` con `SourceModule(-O2)`, lanza `grid=ceil(K/BLOCK_SIZE)` con
`BLOCK_SIZE=256` (DEC-05), reduce con `np.argmax` en host y reconstruye `best_W` con
el MISMO SplitMix64 del kernel (`seed+k`, DEC-12/DEC-15). El tiempo registrado excluye
la carga de datos y la H2D unica (rules.md, Benchmark).

No se ejecuta en local sin GPU (RIESGO-01): pensado para un runtime GPU de Google Colab.

Uso:
    from scoring_pycuda import random_search_cuda
    res = random_search_cuda(n_items=50, K=100_000, seed=42)
    print(res["best_W"], res["best_auc"], res["time_seconds"])
"""
from __future__ import annotations

import time
from math import log
from pathlib import Path

import numpy as np

try:  # PyCUDA solo esta disponible en el runtime GPU (Colab)
    import pycuda.autoinit  # noqa: F401  (inicializa contexto CUDA)
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    PYCUDA_AVAILABLE = True
except ImportError:  # entorno local sin GPU (RIESGO-01)
    PYCUDA_AVAILABLE = False

N_SAMPLES = 10
BLOCK_SIZE = 256  # DEC-05
MASK64 = (1 << 64) - 1
_HERE = Path(__file__).resolve().parent
_KERNEL_PATH = _HERE / "scoring_kernel.cu"


def data_dir(n_items: int) -> Path:
    """Directorio de datos para `n_items` (DEC-10): `code/data/n_{n_items}/`."""
    return _HERE.parents[1] / "data" / f"n_{n_items}"


def validate_dataset(A: np.ndarray, profiles: np.ndarray, y: np.ndarray, n_items: int) -> None:
    """Espejo de `common.validate_dataset`: shapes, dtypes y rangos (DEC-10/rules.md)."""
    if A.shape != (N_SAMPLES, n_items) or A.dtype != np.float32:
        raise ValueError(f"A invalida: shape={A.shape}, dtype={A.dtype}")
    if profiles.shape != (n_items, 3) or profiles.dtype != np.float32:
        raise ValueError(f"profiles invalido: shape={profiles.shape}, dtype={profiles.dtype}")
    if y.shape != (N_SAMPLES,) or y.dtype != np.int32:
        raise ValueError(f"y invalido: shape={y.shape}, dtype={y.dtype}")
    if not np.array_equal(y, np.array([0] * 5 + [1] * 5, dtype=np.int32)):
        raise ValueError(f"y debe ser [0]*5 + [1]*5, recibido {y}")
    T, S, F = profiles[:, 0], profiles[:, 1], profiles[:, 2]
    if not (np.all((T >= 0) & (T <= 1)) and np.all((S >= 0) & (S <= 1))):
        raise ValueError("T y S deben estar en [0, 1]")
    if not np.all(np.isin(F, [0, 1, 2])):
        raise ValueError("F debe contener unicamente valores en {0, 1, 2}")


def load_dataset(n_items: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga y valida los `.npy` de `n_{n_items}/` (solo lectura, fuera del cronometro)."""
    d = data_dir(n_items)
    A = np.ascontiguousarray(np.load(d / "matrix_A.npy"), dtype=np.float32)
    profiles = np.ascontiguousarray(np.load(d / "profiles.npy"), dtype=np.float32)
    y = np.ascontiguousarray(np.load(d / "labels.npy"), dtype=np.int32)
    validate_dataset(A, profiles, y, n_items)
    return A, profiles, y


def sample_dirichlet_host(seed_k: int) -> np.ndarray:
    """Mirror Python EXACTO del SplitMix64->Dirichlet `__device__` (kernel/scoring_openmp.c).

    Reconstruye `W_k` para el indice `seed_k = seed + k` con la MISMA secuencia de bits,
    de modo que `best_W` reconstruido en host coincide con el evaluado en GPU.
    """
    state = seed_k & MASK64
    e: list[float] = []
    for _ in range(3):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z = z ^ (z >> 31)
        u = (z >> 11) * (1.0 / 9007199254740992.0) + (1.0 / 18014398509481984.0)
        e.append(-log(u))
    s = sum(e)
    return np.array([v / s for v in e], dtype=np.float64)


def compile_kernel() -> "cuda.Function":
    """Compila `scoring_kernel.cu` con `SourceModule(-O2)` y devuelve la funcion."""
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA no disponible (entorno sin GPU); ejecutar en Google Colab")
    module = SourceModule(_KERNEL_PATH.read_text(), options=["-O2"])
    return module.get_function("scoring_kernel")


def random_search_cuda(
    n_items: int = 50, K: int = 100_000, seed: int = 42, block_size: int = BLOCK_SIZE
) -> dict:
    """Random Search en GPU equivalente a Fases 1-3. Devuelve best_W/best_auc/tiempos.

    El cronometro (`time_seconds`) cubre [lanzamiento -> sync -> D2H -> argmax] y EXCLUYE
    la carga de datos y la H2D unica de A/profiles/labels (DEC-15). `kernel_ms` mide el
    kernel puro con `cudaEvent_t` (analisis Fase 5).
    """
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA no disponible (entorno sin GPU); ejecutar en Google Colab")

    A, profiles, y = load_dataset(n_items)
    kernel = compile_kernel()

    # H2D una sola vez (coding-standards); fuera del cronometro de busqueda.
    dA = cuda.mem_alloc(A.nbytes)
    dProf = cuda.mem_alloc(profiles.nbytes)
    dY = cuda.mem_alloc(y.nbytes)
    cuda.memcpy_htod(dA, A)
    cuda.memcpy_htod(dProf, profiles)
    cuda.memcpy_htod(dY, y)
    auc_out = np.empty(K, dtype=np.float32)
    dAuc = cuda.mem_alloc(auc_out.nbytes)

    grid = ((K + block_size - 1) // block_size, 1)
    block = (block_size, 1, 1)
    start, end = cuda.Event(), cuda.Event()

    cuda.Context.synchronize()
    t0 = time.perf_counter()
    start.record()
    kernel(dA, dProf, dY, dAuc, np.int32(n_items), np.int32(K), np.uint64(seed),
           block=block, grid=grid)
    end.record()
    end.synchronize()
    cuda.memcpy_dtoh(auc_out, dAuc)
    k_star = int(np.argmax(auc_out))  # primer maximo => menor k (igual a Fase 1/MAXLOC Fase 3)
    cuda.Context.synchronize()
    time_seconds = time.perf_counter() - t0
    kernel_ms = start.time_till(end)

    best_W = sample_dirichlet_host((seed + k_star) & MASK64)
    gpu_name = cuda.Context.get_device().name()
    return {
        "best_W": best_W,
        "best_auc": float(auc_out[k_star]),
        "k_star": k_star,
        "time_seconds": time_seconds,
        "kernel_ms": kernel_ms,
        "auc_out": auc_out,
        "gpu_name": gpu_name,
    }


if __name__ == "__main__":
    res = random_search_cuda()
    print(f"GPU: {res['gpu_name']}")
    print(f"Best W: {res['best_W']}  |  AUC: {res['best_auc']:.4f}")
    print(f"Tiempo (busqueda de W*): {res['time_seconds']:.6f} s  |  kernel: {res['kernel_ms']:.3f} ms")
