"""
Nivel 3 — PyCUDA: interfaz Python para el kernel CUDA de scoring.
"""
import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("PyCUDA no disponible — instalar con: pip install pycuda")


def random_search_cuda(A: np.ndarray, y: np.ndarray, K: int = 100_000, seed: int = 42):
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA no instalado")

    rng = np.random.default_rng(seed)
    W_pool = rng.dirichlet(np.ones(3), size=K).astype(np.float32)

    # TODO: compilar kernel, transferir A y W_pool a GPU,
    #       lanzar grid de ceil(K/BLOCK_SIZE) bloques,
    #       recuperar auc_out y retornar argmax

    raise NotImplementedError("Implementar lanzamiento del kernel CUDA")


if __name__ == "__main__":
    A = np.load("../data/matrix_A.npy")
    y = np.load("../data/labels.npy")
    W, auc = random_search_cuda(A, y)
    print(f"Best W: {W}  |  AUC: {auc:.4f}")
