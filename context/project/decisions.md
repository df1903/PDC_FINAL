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

## DEC-10 — `profiles.npy` combinado y estructura `data/n_{n_items}/` `[CONFIRMADO]`

**Decisión**: los perfiles `T`, `S`, `F` se almacenan en un único archivo `profiles.npy` de shape `(n_items, 3)` float32 (`profiles[:,0]=T`, `profiles[:,1]=S`, `profiles[:,2]=F`), y todos los artefactos de datos (`matrix_A.npy`, `profiles.npy`, `labels.npy`) se organizan por tamaño de problema en `code/data/n_{n_items}/` (ej. `data/n_50/`, `data/n_100/`). `generate_data.py` expone `get_data_directory(n_items)`, `generate_profiles(n_items)`, `load_or_generate_dataset(...)` y `load_or_generate_profiles(...)`, que cargan archivos existentes y generan únicamente los faltantes.

**Razón**: esta convención reemplaza a los archivos independientes `T.npy`/`S.npy`/`F.npy` planos descritos originalmente en DEC-08/`rules.md`. Un archivo combinado simplifica la carga (`profiles[:, i]`) y la organización por subdirectorio `n_{n_items}` permite ejecutar pruebas de escalabilidad con múltiples valores de N sin sobrescribir datasets previos ni regenerar innecesariamente.

**Interpretación descartada**: mantener `T.npy`, `S.npy`, `F.npy` como archivos separados y planos en `code/data/` (versión previa de DEC-08), y regenerar todo el dataset en cada ejecución sin reutilizar archivos existentes.

## DEC-09 — Fase 4 (CUDA) en notebook para Google Colab `[CONFIRMADO]`

**Decisión**: la Fase 4 se desarrolla en `CUDA/scoring_cuda.ipynb`, ejecutado en un runtime GPU de Google Colab. El kernel CUDA C se materializa con `%%writefile` + `nvcc` y se orquesta con PyCUDA; `scoring_kernel.cu` y `scoring_pycuda.py` se conservan como fuentes derivadas.

**Razón**: no se asume GPU local; Colab provee GPU y toolkit CUDA, simplificando la reproducibilidad de la medición de speedup en GPU.

## DEC-11 — Señal diferencial en `generate_data.py` y regeneración de `data/n_{n_items}/` `[CONFIRMADO]`

**Decisión**: como parte de la Fase 1, se autoriza modificar `code/data/generate_data.py` (`generate_data`, y si es necesario `generate_profiles`) para inyectar **señal diferencial** en las filas enfermas de `A` (correlación con ítems de alto `T`/`F`), tal como exigen RF-01, `constraints.md` y RIESGO-04. Como consecuencia, se regeneran los artefactos `.npy` existentes en `data/n_50/` y `data/n_100/` con `seed=42`.

**Razón**: con el generador actual (`A` Dirichlet puro, perfiles `T/S` uniformes y `F` uniforme en {0,1,2} sin correlación con `A`), `P = W₁T+W₂S+W₃F` es común a todas las muestras y la separación entre grupos depende solo de cómo `A` pondera los ítems — pero `A` no tiene relación con la etiqueta, por lo que ningún `W` del simplex produce AUC > 0.5 ni consistencia ≥ 0.8 (ec. 4). La Fase 1 no puede validarse (RNF-02) sin esta señal.

**Alcance**: no cambia la estructura de archivos definida en DEC-10 (`matrix_A.npy`, `profiles.npy`, `labels.npy` en `data/n_{n_items}/`, mismas shapes y dtypes) — solo cambian los **valores** generados. `load_or_generate*` deben regenerar (no reutilizar) los archivos afectados; documentar en el código qué cambió para que una regeneración futura sea reproducible con `seed=42`.

## DEC-12 — Estrategia de la Fase 2 (C + OpenMP) `[CONFIRMADO — 2026-06-15]`

**Decisión**: precisar cómo se implementa la Fase 2 (`scoring_openmp.c`), refinando DEC-06 y respetando DEC-10. Plan completo en `context/state/active-tasks.md` (traceability `traceability_data/2026_06_15_14-24.md`).

- **Carga de datos**: parser `.npy` v1.0 **mínimo y propio** en C (`code/C_OpenMP_MPI/npy_io.{h,c}`), que lee directamente `matrix_A.npy`/`profiles.npy`/`labels.npy` (formato verificado con `xxd`: `<f4`/`<i4`, `fortran_order=False`, longitud de header en campo LE de 2 bytes en offset 8). Se elige carga binaria directa (la opción "carga binaria directa" de DEC-06) frente a exportar CSV adicional. Sin librerías externas (cumple `rules.md`); sin modificar los `.npy` (DEC-10); reutilizable por `scoring_mpi.c` (Fase 3).
- **RNG**: SplitMix64 sembrado **por candidato** con `seed + k` → 3 exponenciales normalizadas = Dirichlet(1,1,1). `W_k` depende solo de `k`, lo que hace `best_auc` determinista y reproducible para P ∈ {1,2,4,8} y permite muestreo sin sección crítica. **No** se replica bit a bit el stream de `np.random.default_rng(42).dirichlet(...)` (PCG64) de la Fase 1: no es viable ni necesario, porque RF-04/`rules.md` exigen equivalencia del **valor de AUC**, no `best_W` idéntico.
- **AUC con empates**: la fórmula de conteo de pares debe acreditar empates con 0.5 — `AUC = (concordantes + 0.5·empates)/(pos·neg)` — para igualar `roc_auc_score`. Corrige el placeholder (ISSUE-008, RIESGO-03).
- **Equivalencia**: el criterio `|ΔAUC| < 1e-4` es sobre el **valor de AUC** (no sobre `best_W`); se valida en dos capas (`--self-test` de la función AUC con un caso de empate + comparación de `best_auc` vs Python secuencial), apoyándose en que el dataset de Fase 1 es perfectamente separable (AUC=1.0).
- **Etiqueta de benchmark**: la columna `implementation` para esta fase es **`C OpenMP`** (estilo "Python multicore"), en `results/benchmark.csv` con el mismo esquema de 9 columnas (append-only). `t_seq` para speedup/efficiency se lee de la fila `Python secuencial` coincidente en `n_items`/`k_candidates`.

**Razón**: mantener equivalencia con la Fase 1 validada sin añadir dependencias ni tocar los datos, dejando explícito qué es reproducible bit a bit (no el stream de candidatos) y qué se valida realmente (el valor de AUC).

**Interpretación descartada**: exportar CSV/binarios adicionales desde Python para C (añade artefactos y toca `data/`); intentar replicar PCG64 + el algoritmo gamma de NumPy en C (inviable y no exigido); contar AUC solo con pares estrictos sin acreditar empates (diverge de sklearn).

## DEC-13 — Redefinición de `speedup`/`efficiency` y nueva columna `speedup_vs_python` `[CONFIRMADO — 2026-06-15]`

**Decisión**: corregir el cálculo de `speedup`/`efficiency` en `results/benchmark.csv` para que
representen **speedup paralelo** (respecto al baseline P=1 de la MISMA implementación), no la
diferencia de rendimiento entre lenguajes/implementaciones:

- `speedup(P) = T_impl(P=1) / T_impl(P)`, `efficiency = speedup / workers`.
- "Python secuencial" es el baseline P=1 de la familia Python: `speedup_python_multicore =
  T_python_secuencial / T_python_multicore` (sin cambios respecto a DEC-11, ya era correcto).
- "C OpenMP" (y "C MPI" en Fase 3) usan como baseline P=1 su propia fila `workers=1` en
  `benchmark.csv` (`speedup_openmp(P) = T_openmp(1) / T_openmp(P)`).
- Se agrega la columna `speedup_vs_python = T_python_secuencial / T_impl(P)` para la
  comparación explícita entre implementaciones (lo que antes ocupaba indebidamente la columna
  `speedup` para "C OpenMP"). Esquema de `benchmark.csv` pasa de 9 a **10 columnas**:
  `..., candidates_per_second, speedup, efficiency, speedup_vs_python`.
- `common.py: compute_metrics(K, time_s, workers, t_self_base=None, t_python_seq=None)`
  reemplaza el parámetro `t_seq` único; `scoring_openmp.c` lee su propio baseline P=1 vía
  `read_last_time(csv, "C OpenMP", n_items, k_candidates, required_workers=1, ...)` (función que
  generaliza/reemplaza `read_sequential_time`).

**Razón**: el cálculo previo (`speedup = T_python_secuencial / T_impl`, ver DEC-11/`rules.md`
anterior) producía valores de miles para "C OpenMP" (p.ej. `speedup(P=8)=14813×`) que mezclan
dos efectos distintos — la diferencia de rendimiento C vs Python/NumPy/sklearn (órdenes de
magnitud) y el speedup por paralelización OpenMP (sublineal, esperado para N=50) — haciendo que
`efficiency = speedup/threads` no fuera interpretable como eficiencia de paralelización clásica
(podía superar 1.0 por miles). Separar ambas métricas permite leer `speedup`/`efficiency` como
en cualquier análisis de Amdahl (Fase 6) y conservar la comparación entre lenguajes en
`speedup_vs_python` sin perderla.

**Interpretación descartada**: mantener `speedup` como comparación contra Python secuencial
para todas las implementaciones (DEC-11/`rules.md` previos a esta decisión) — válido para
Python multicore (su baseline P=1 ya es Python secuencial) pero no generalizable a C/CUDA, cuyo
baseline P=1 es su propia versión de un hilo/proceso.
