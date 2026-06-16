# Active Tasks — Fase 4 (CUDA en Google Colab)

**Estado: PLANIFICADA — 2026-06-16** (plan definido en
`traceability_data/2026_06_16_00-43.md`, iteración 1; estrategia formalizada en **DEC-15**).
Pendiente de implementación. **La ejecución ocurre en Google Colab** (runtime GPU): este plan
se deja planteado para correrse después; no se ejecuta en el entorno local.

Fases 1 (Python Baseline), 2 (C + OpenMP) y 3 (C + MPI) **COMPLETADAS**.

> El plan ya ejecutado de Fase 3 (C + MPI) se conserva como referencia histórica en
> `traceability_data/2026_06_15_19-22.md` (planificación) y `2026_06_15_19-32.md`
> (implementación); sus resultados en `context/state/current-phase.md`.
> Fase 4 **reutiliza el cómputo validado** (RNG SplitMix64, Score, AUC con empates, consistencia)
> copiándolo de `scoring_openmp.c` como código `__device__`; solo cambia el modelo de paralelismo
> (miles de hilos GPU, un hilo por candidato) y el lenguaje de orquestación (PyCUDA en notebook).

Objetivo: implementar `code/CUDA/scoring_cuda.ipynb` (Random Search con aceleración **GPU
masiva**) **equivalente a las Fases 1–3** (mismo valor de AUC, `|ΔAUC| < 1e-4`, esquema de
benchmark de 10 columnas), leyendo los `.npy` existentes sin modificarlos (DEC-10), sin
dependencias más allá de PyCUDA (ya en `stack.md`), con `BLOCK_SIZE=256` (DEC-05) y `seed=42`
(DEC-03). Estrategia formalizada en **DEC-15**. `scoring_kernel.cu` y `scoring_pycuda.py` se
conservan como fuentes derivadas del notebook (DEC-09). Ejecutar las tareas en orden.

## Hallazgos verificados contra el repo (base del plan)

- **`scoring_kernel.cu` es un placeholder** (29 LOC): `__global__ scoring_kernel` con `auc_out[k]
  = 0.5f` y `main()` de stub. Reemplazo completo (kernel real + RNG `__device__`).
- **`scoring_pycuda.py` es un placeholder**: `random_search_cuda` con `raise NotImplementedError`
  y rutas `.npy` antiguas (`../data/matrix_A.npy`, sin `n_{n_items}/`). Reescribir conforme DEC-10.
- **Cómputo ya validado (Fases 1–3)**: `splitmix64_next`/`splitmix64_uniform01`/`sample_dirichlet`,
  `compute_P`/`compute_score`/`compute_auc` (empates a 0.5), `scoring_consistency`, caso de
  self-test (empate → 0.875) en `scoring_openmp.c` — se **portan a CUDA** (host/`__device__`).
- **Datos `code/data/n_50/`**: `matrix_A.npy` (10×N float32, C-order), `profiles.npy` (N×3 float32
  → col0=T, col1=S, col2=F), `labels.npy` (10 int32 = `[0]*5+[1]*5`). Solo lectura (DEC-10).
- **`benchmark.csv` en esquema de 10 columnas** (DEC-13). `Python secuencial` n_50/K=100000 →
  `time_seconds = 96.30376639` (baseline para `speedup_vs_python`). Append-only; no tocar filas
  previas ni el esquema.
- **Equivalencia por valor de AUC** (DEC-12/DEC-15): RF-04 exige `|ΔAUC| < 1e-4` sobre el valor de
  AUC, no `best_W`. Se acredita en n_50 (ambos llegan a AUC=1.0).
- **Entorno (RIESGO-01/06)**: GPU de Colab (típicamente T4); no se asume GPU local. Reproducibilidad
  con `seed=42`, registro del modelo de GPU y promedio de repeticiones.

## 1. Estructura del notebook `CUDA/scoring_cuda.ipynb` (orden de celdas)

| # | Celda | Contenido |
|---|-------|-----------|
| 0 | Markdown | Intro + parámetros `N_ITEMS=50`, `K=100000`, `SEED=42`, `BLOCK_SIZE=256`, `N_SAMPLES=10` |
| 1 | GPU | `!nvidia-smi` → capturar y registrar el modelo de GPU (RIESGO-06) |
| 2 | Setup | `import pycuda.autoinit, pycuda.driver as cuda`, `SourceModule`; instalar solo si falta |
| 3 | Datos | subir/cargar `.npy` de `n_50/` + validación (espejo de `common.validate_dataset`) |
| 4 | `%%writefile scoring_kernel.cu` | RNG `__device__` + `__global__ scoring_kernel` |
| 5 | Compilación | `SourceModule(src, options=["-O2"])` (+ `nvcc -O2` opcional como sanity) |
| 6 | Orquestación | H2D una vez → lanzamiento grid `ceil(K/256)` → D2H `auc_out` |
| 7 | Reducción | `np.argmax(auc_out)` → reconstruir `best_W` |
| 8 | Timing | `cudaEvent_t` (kernel) + `perf_counter`+`synchronize()` (búsqueda de W*) |
| 9 | Self-test | empate=0.875 + caso `n_items=3`/K=100 vs mirror SplitMix64 |
| 10 | Validación | AUC∈[0.5,1], `|ΔAUC|<1e-4`, consistencia≥0.8 |
| 11 | Benchmark | construir fila `CUDA` (10 col) y mostrarla/appendear |
| 12 | Resumen | prints de `best_W`, AUC, T, speedup, modelo de GPU |

`scoring_pycuda.py` = export de las celdas 4–7 como script reusable; `scoring_kernel.cu` = el
archivo materializado por `%%writefile` (ambos fuentes derivadas, DEC-09).

## 2. Setup Colab + verificación GPU

- `!nvidia-smi` para confirmar runtime GPU y **registrar el modelo** (típicamente T4) en variable
  y en el informe — exigido por RNF-03 y RIESGO-06.
- PyCUDA ya está en `stack.md`; Colab suele traerlo. Instalar **solo si el import falla** (no se
  añaden dependencias nuevas al proyecto).

## 3. Carga de `.npy` desde `code/data/n_50/`

- Subir los tres `.npy` al runtime (`files.upload()`, `git clone` o Drive — documentar la elegida).
- `np.load` + validación **idéntica** a `common.validate_dataset`: shapes `(10,N)`/`(N,3)`/`(10,)`,
  dtypes `float32`/`float32`/`int32`, `T,S∈[0,1]`, `F∈{0,1,2}`, `y==[0]*5+[1]*5`.
- Garantizar `A` contigua C-order y `float32` antes de H2D. Carga **fuera del cronómetro**.

## 4. RNG — SplitMix64 `__device__` con `seed+k` (DEC-15)

- Copiar `splitmix64_next`/`splitmix64_uniform01`/`sample_dirichlet` de `scoring_openmp.c` como
  `__device__`. Cada hilo reconstruye `W_k = sample_dirichlet(seed + (uint64_t)k)` desde su índice
  **global** `k`.
- Idéntico a Fases 2/3 (mismo AUC por construcción, DEC-12); **cero transferencia** de `W_pool`;
  determinista. **Descartado** W_pool en host salvo que use el mismo SplitMix64 (no PCG64).

## 5. Kernel `scoring_kernel` (un hilo por candidato, `__shared__` para A)

```
grid = ceil(K/256), block = 256                  // BLOCK_SIZE=256 (DEC-05)
__shared__ float sA[N_SAMPLES*MAX_ITEMS];         // ~2 KB para N=50
__shared__ float sProf[MAX_ITEMS*3];              // carga cooperativa + __syncthreads()
por hilo k (k = blockIdx.x*blockDim.x + threadIdx.x; if (k>=K) return;):
   W_k = sample_dirichlet(seed + k)               // RNG device (tarea 4)
   for i: P[i] = W0*sProf[i,0] + W1*sProf[i,1] + W2*sProf[i,2]   // array local, N≤MAX_ITEMS
   for j(10): score[j] = Σ_i sA[j,i]*P[i]
   auc = conteo de pares 5×5 con empates +0.5      // acumular en double
   auc_out[k] = (float)auc
```

- `__shared__` cachea `A` (10×N) y `profiles` reusados por los 256 hilos del bloque
  (`coding-standards.md`); cabe holgado en 48 KB.
- `P` por hilo en array local dimensionado por `#define MAX_ITEMS` (p. ej. 256) — **sin VLAs**.
- AUC replica la fórmula de empates de `scoring_openmp.c` (RIESGO-03).

## 6. Reducción — `np.argmax` en host (DEC-15)

- D2H de `auc_out` (K=100k float32 ≈ 400 KB, despreciable) → `np.argmax` (primer máximo →
  desempate por menor `k`, igual a Fase 1 y al `MAXLOC` de Fase 3) → `best_W =
  sample_dirichlet(seed + k*)` reconstruido en host.
- **Justificación**: un reduction kernel añade complejidad (`<400 LOC`) sin ganancia a esta escala;
  `argmax` host también es la lógica de `evaluate_candidates` (Fase 1).

## 7. Compilación — `%%writefile` + `SourceModule(-O2)` (DEC-15)

- `%%writefile scoring_kernel.cu` **materializa el fuente derivado** (DEC-09/nomenclatura).
- Se lee y se pasa a `pycuda.compiler.SourceModule(source, options=["-O2"])` (JIT) para orquestar.
- `!nvcc -O2 scoring_kernel.cu` opcional como **chequeo de sintaxis**.
- Para `SourceModule` el `.cu` lleva solo `__device__`+`__global__` (sin `main`); `< 400 LOC`.

## 8. Orquestación PyCUDA (H2D / D2H + lanzamiento)

- **H2D una sola vez** (`coding-standards.md`): `A`, `profiles`, `labels` (`mem_alloc` +
  `memcpy_htod`); `auc_out` device (K float32).
- Lanzamiento: `kernel(dA, dProf, dLabels, dAuc, np.int32(N), np.int32(K), np.uint64(SEED),
  block=(256,1,1), grid=(ceil(K/256),1))`.
- D2H: `auc_out` → host. Macro `CUDA_CHECK` para verificar errores CUDA.

## 9. Medición de tiempo (excluye carga de datos)

- **`cudaEvent_t`** alrededor del kernel puro → tiempo de cómputo GPU (análisis/Amdahl Fase 5).
- **`time.perf_counter`** con `cuda.Context.synchronize()` envolviendo [lanzamiento → sync → D2H →
  argmax] = tiempo de **búsqueda de W\*** → es el `time_seconds` registrado en el CSV.
- Se **excluye** la carga `.npy` y la **H2D única** de A/profiles/labels (análoga a "carga de
  datos", `rules.md` §Benchmark). Decisión explícita (DEC-15).

## 10. Self-test (N=3 / K=100)

1. **Unitario AUC**: caso `score=[1,2,2,3]`, `y=[0,0,1,1]` → **0.875** exacto (mismo de `self_test`
   en C, RIESGO-03).
2. **Extremo-a-extremo** `n_items=3`/K=100: comparar `best_auc` CUDA vs una **referencia Python con
   el MISMO SplitMix64** (mirror en el notebook) → `|ΔAUC|≈0`.

> ⚠️ Comparar el caso chico directamente contra `sequential.py` (PCG64, K=100) **no es
> apples-to-apples** (RNG distinto) y puede diferir > 1e-4 sin que haya bug. Por eso el caso chico
> usa el mirror SplitMix64; la equivalencia formal "vs Python secuencial" se acredita en n_50
> (ambos → AUC=1.0, `|ΔAUC|=0`).

## 11. Registro en `results/benchmark.csv` (10 columnas, append-only, DEC-13)

Una sola fila (no hay barrido P, solo un punto de medición):

| columna | valor en CUDA |
|---|---|
| `implementation` | `CUDA` |
| `n_items` / `k_candidates` | `50` / `100000` |
| `workers` | `1` (1 GPU) |
| `best_auc`, `time_seconds` | del run (tiempo solo de búsqueda, tarea 9) |
| `candidates_per_second` | `K / time_seconds` |
| `speedup` | `1.0` (sin barrido P) |
| `efficiency` | `1.0` |
| `speedup_vs_python` | `T_python_secuencial / T_cuda` (`T_python_secuencial = 96.30376639`) |

- El CSV se genera **en Colab** ⇒ la fila se **transcribe/appendea** a `code/results/benchmark.csv`
  del repo sin tocar filas previas ni el esquema de 10 columnas.

## 12. Validación de correctitud (RF-04 / RNF-02)

- `best_auc ∈ [0.5, 1.0]` (esperado 1.0).
- `|best_auc_CUDA − best_auc_PySeq| < 1e-4` (mismo K=100k, seed=42, N=50; esperado 0).
- Consistencia (ec. 4) ≥ 0.8 (esperado 2.0), computando `scoring_consistency(best_W)` en host.

## 13. Rendimiento (RNF-03) y entorno

- `speedup_vs_python = 96.30 / T_cuda ≥ 5×` (objetivo CUDA). Con T4 se espera holgadamente.
- **Registrar el modelo de GPU**; mitigar RIESGO-06: `seed=42`, repetir y promediar, conservar
  `scoring_kernel.cu`/`scoring_pycuda.py` para reejecución.

## 14. Criterios de salida de Fase 4 (no avanzar a Fase 5 sin esto)

- [ ] Self-test de AUC en verde, incluido el caso con empate → 0.875 (RIESGO-03).
- [ ] Self-test `n_items=3`/K=100 vs mirror SplitMix64 → `|ΔAUC| ≈ 0`.
- [ ] `|best_auc_CUDA − best_auc_PySeq| < 1e-4` en n_50/K=100k (RF-04); `best_auc ∈ [0.5,1.0]`.
- [ ] Consistencia (ec. 4) ≥ 0.8.
- [ ] `speedup_vs_python ≥ 5×` (RNF-03), con modelo de GPU registrado.
- [ ] Fila `CUDA` añadida a `results/benchmark.csv` sin perder filas previas ni cambiar el esquema.
- [ ] `scoring_kernel.cu` < 400 LOC; `scoring_pycuda.py` actualizado (rutas `n_{n_items}/`, DEC-10).

## 15. Entorno y comandos (en Google Colab)

```text
# En Colab (runtime GPU):
#  1) Subir code/data/n_50/{matrix_A,profiles,labels}.npy al runtime.
#  2) Abrir CUDA/scoring_cuda.ipynb y ejecutar todas las celdas en orden.
#  3) Verificar GPU:           !nvidia-smi
#  4) (opcional) sanity nvcc:  !nvcc -O2 scoring_kernel.cu -o /tmp/k   # solo compila
#  5) Tras la corrida: transcribir la fila CUDA a code/results/benchmark.csv (append-only).
```

- **No ejecutable en el entorno local** (sin GPU asumida, RIESGO-01): se deja planteado para Colab.

## Restricciones (recordatorio)

Sin modificar `scoring_openmp.c` · `scoring_mpi.c` · `npy_io.{h,c}` · sin tocar los `.npy`
(solo lectura) · sin tocar filas previas ni el esquema de 10 columnas de `benchmark.csv` ·
`scoring_kernel.cu` **< 400 LOC** · sin dependencias nuevas más allá de PyCUDA (ya en `stack.md`) ·
`BLOCK_SIZE = 256` (DEC-05) · `seed = 42` (DEC-03) · `workers = 1` en el CSV (1 GPU, sin barrido P).
