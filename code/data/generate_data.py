"""Generador de datos para el proyecto de Scoring Metagenómico.

Para cada tamaño de problema `n_items` se genera/carga, dentro de
`data/n_{n_items}/`:

- `matrix_A.npy`  -> A en R^{10 x n_items}, float32 (matriz de contribución).
- `labels.npy`    -> y en {0,1}^10, int32 (filas 0-4 sanas, 5-9 enfermas).
- `profiles.npy`  -> perfiles por ítem, float32, shape (n_items, 3):
                      columna 0 = T (taxonómico,      float en [0,1])
                      columna 1 = S (socioeconómico,  float en [0,1])
                      columna 2 = F (funcional,       entero en {0,1,2})

Estos perfiles se usarán posteriormente para calcular, por ítem:

    P_i = W1*T_i + W2*S_i + W3*F_i

con T = profiles[:, 0], S = profiles[:, 1], F = profiles[:, 2], y luego
Score = A @ P para obtener el score por muestra.

Organizar los datos por `n_items` (`data/n_{n_items}/`) permite reutilizar
o regenerar únicamente lo necesario para cada tamaño de problema, facilitando
las pruebas de escalabilidad.
"""

from pathlib import Path
from typing import Callable

import numpy as np

DEFAULT_SEED = 42


def get_data_directory(n_items: int) -> Path:
    """Retorna la ruta del directorio de datos para un tamaño de problema dado.

    Args:
        n_items: número de ítems (N) del problema.

    Returns:
        Ruta `data/n_{n_items}/`.
    """
    return Path("data") / f"n_{n_items}"


def generate_data(n_items: int = 50, seed: int = DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Genera la matriz de contribución A y las etiquetas y.

    Filas 0-4: sanas (y=0). Filas 5-9: enfermas (y=1).
    Cada fila de A es Dirichlet(1,...,1) -> suma 1.

    Args:
        n_items: número de ítems (N), tamaño de las columnas de A.
        seed: semilla para reproducibilidad.

    Returns:
        Tupla (A, y) con A de shape (10, n_items) float32 y y de shape (10,) int32.
    """
    rng = np.random.default_rng(seed)
    A = rng.dirichlet(np.ones(n_items), size=10).astype(np.float32)
    y = np.array([0] * 5 + [1] * 5, dtype=np.int32)
    return A, y


def generate_profiles(n_items: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Genera los perfiles por ítem T, S y F.

    Args:
        n_items: número de ítems (N).
        seed: semilla para reproducibilidad.

    Returns:
        Array de shape (n_items, 3) y dtype float32, donde:
            columna 0 -> T: perfil taxonómico, float en [0, 1].
            columna 1 -> S: perfil socioeconómico/ecológico, float en [0, 1].
            columna 2 -> F: perfil funcional, entero en {0, 1, 2}.
    """
    rng = np.random.default_rng(seed)
    T = rng.random(n_items)
    S = rng.random(n_items)
    F = rng.integers(0, 3, size=n_items)
    return np.column_stack([T, S, F]).astype(np.float32)


def load_or_generate(path: Path, generator: Callable[[], np.ndarray]) -> np.ndarray:
    """Carga un array desde `path` si existe; si no, lo genera y guarda.

    Args:
        path: ruta del archivo `.npy`.
        generator: función sin argumentos que produce el array a guardar
            en caso de que `path` no exista.

    Returns:
        El array cargado desde disco o recién generado.
    """
    if path.exists():
        array = np.load(path)
        print(f"[INFO] Loaded {path.name}")
    else:
        array = generator()
        np.save(path, array)
        print(f"[INFO] Generated {path.name}")
    return array


def load_or_generate_dataset(
    n_items: int, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Carga o genera la matriz de contribución A y las etiquetas y.

    Crea `data/n_{n_items}/` si no existe y reutiliza `matrix_A.npy` /
    `labels.npy` cuando ya están presentes, generando únicamente lo faltante.

    Args:
        n_items: número de ítems (N) del problema.
        seed: semilla para reproducibilidad.

    Returns:
        Tupla (A, y).
    """
    data_dir = get_data_directory(n_items)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using directory: {data_dir}")

    A = load_or_generate(data_dir / "matrix_A.npy", lambda: generate_data(n_items, seed)[0])
    y = load_or_generate(data_dir / "labels.npy", lambda: generate_data(n_items, seed)[1])
    return A, y


def load_or_generate_profiles(n_items: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Carga o genera los perfiles por ítem T, S y F.

    Crea `data/n_{n_items}/` si no existe y reutiliza `profiles.npy` cuando
    ya está presente.

    Args:
        n_items: número de ítems (N) del problema.
        seed: semilla para reproducibilidad.

    Returns:
        Array de shape (n_items, 3) con T, S y F en sus columnas (ver
        `generate_profiles`).
    """
    data_dir = get_data_directory(n_items)
    data_dir.mkdir(parents=True, exist_ok=True)
    return load_or_generate(data_dir / "profiles.npy", lambda: generate_profiles(n_items, seed))


if __name__ == "__main__":
    n_items = 50

    A, y = load_or_generate_dataset(n_items=n_items)
    profiles = load_or_generate_profiles(n_items=n_items)

    print(f"A: {A.shape} | y: {y} | profiles: {profiles.shape}")
