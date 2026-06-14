# Decisions

## DEC-01 — Random Search como estrategia de optimización

**Decisión**: usar Random Search (muestreo uniforme del simplex) en lugar de métodos de gradiente o búsqueda exhaustiva.

**Razón**: permite paralelización trivial (cada candidato W es independiente), es adecuado para demostrar speedup en HPC, y el simplex 3D es suficientemente pequeño para K=100k candidatos.

## DEC-02 — Función objetivo: AUC

**Decisión**: maximizar AUC (área bajo curva ROC) como función objetivo.

**Razón**: el AUC es la métrica estándar para clasificación binaria con clases balanceadas (5+5). Mide separabilidad global entre grupos sano/enfermo sin depender de un umbral fijo.

## DEC-03 — Semilla fija seed=42

**Decisión**: usar `seed=42` en todos los generadores de números aleatorios de todas las implementaciones.

**Razón**: garantizar reproducibilidad y comparabilidad entre implementaciones y entre ejecuciones.

## DEC-04 — K = 100 000 candidatos

**Decisión**: K=100_000 como valor por defecto.

**Razón**: suficiente para explorar el simplex 3D con buena cobertura, y lo bastante grande para que el costo de paralelización sea amortizable (overhead de MPI/CUDA < ganancia).

## DEC-05 — BLOCK_SIZE = 256 para CUDA

**Decisión**: BLOCK_SIZE=256 hilos por bloque en el kernel CUDA.

**Razón**: múltiplo de 32 (warp size), equilibrio entre ocupancia y uso de registros para GPU con capacidad de cómputo ≥ 6.x.

## DEC-06 — Datos en formato .npy

**Decisión**: almacenar A y y como archivos `.npy` de NumPy.

**Razón**: carga O(1) en Python con `np.load`. Para C se leen vía exportación CSV o carga binaria directa.

## DEC-07 — Modelo de perfiles por ítem `[CONFIRMADO]`

**Decisión**: `T`, `S` y `F` son **tres perfiles independientes por ítem** (vectores de longitud N), y `A ∈ ℝ^{10×N}` es la **matriz de contribución**, una entidad distinta. El score se calcula como:

```
P     = W₁·T + W₂·S + W₃·F      (vector ℝ^N)
Score = A · P                     (vector ℝ^{10})
```

- `T` (taxonómico): abundancias de microorganismos y su diferencia (float).
- `S` (socioeconómico): valor contextual que aporta su peso a la suma (float).
- `F` (funcional): conteo de genes beneficiosos por ítem, valores en {0,1,2} (int).

**Razón**: es el modelo del enunciado (`others/proyecto_final.md` §2, ec. 1–2) y de la aclaración del usuario. `A @ P` es dimensionalmente directo (10×N · N) y el simplex de W permanece 3D.

**Interpretación descartada** (deja sin efecto la versión previa de esta decisión): dividir las columnas de `A` en tres bloques T/S/F y promediar cada bloque. También descartadas: W ∈ ℝ^{50} (cambia el modelo a simplex 50D) y N=3 fijo (pierde escalabilidad).

## DEC-08 — Datos reconstruibles con perfiles en `.npy` separados `[CONFIRMADO]`

**Decisión**: el modelo queda abierto; `generate_data.py` reconstruye los datos en archivos `.npy` separados (`matrix_A.npy`, `T.npy`, `S.npy`, `F.npy`, `labels.npy`) con `n_items` configurable, e inyecta **señal diferencial** en las filas enfermas de `A` (correlación con ítems de alto F/T) para garantizar separabilidad.

**Razón**: el enunciado indica que la matriz de ejemplo puede reconstruirse para que el modelo funcione en todos los casos; archivos separados mantienen la convención `.npy` y permiten variar N para pruebas de escalabilidad.

## DEC-09 — Fase 4 (CUDA) en notebook para Google Colab `[CONFIRMADO]`

**Decisión**: la Fase 4 se desarrolla en `CUDA/scoring_cuda.ipynb`, ejecutado en un runtime GPU de Google Colab. El kernel CUDA C se materializa con `%%writefile` + `nvcc` y se orquesta con PyCUDA; `scoring_kernel.cu` y `scoring_pycuda.py` se conservan como fuentes derivadas.

**Razón**: no se asume GPU local; Colab provee GPU y toolkit CUDA, simplificando la reproducibilidad de la medición de speedup en GPU.
