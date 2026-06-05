import numpy as np

def generate_data(n_items: int = 50, seed: int = 42):
    """
    Genera A in R^{10 x n_items} y etiquetas y.
    Filas 0-4: sanas (y=0). Filas 5-9: enfermas (y=1).
    Cada fila de A es Dirichlet(1,...,1) -> suma 1.
    """
    rng = np.random.default_rng(seed)
    A = rng.dirichlet(np.ones(n_items), size=10).astype(np.float32)
    y = np.array([0]*5 + [1]*5, dtype=np.int32)
    return A, y

if __name__ == "__main__":
    A, y = generate_data(n_items=50)
    np.save("data/matrix_A.npy", A)
    np.save("data/labels.npy",   y)
    print(f"A: {A.shape}  |  y: {y}")
