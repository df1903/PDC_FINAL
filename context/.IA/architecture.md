# Architecture

## Visión general

El sistema implementa **Random Search** sobre el simplex de W = (W₁, W₂, W₃) para maximizar el AUC en clasificación binaria de 10 muestras metagenómicas (5 sanas / 5 enfermas). La misma lógica se replica a tres niveles de paralelismo creciente.

## Modelo matemático

```
P_i  = W₁·T_i + W₂·S_i + W₃·F_i          (score por ítem,  P ∈ ℝ^N)
Score = A · P                               (score por muestra, A ∈ ℝ^{10×N})
W*   = argmax_W AUC(y, Score(W))           (función objetivo)
```

`T`, `S` y `F` son **tres perfiles independientes por ítem** (vectores de longitud N), NO bloques de columnas de `A`:

| Perfil | Símbolo | Naturaleza | Descripción |
|--------|---------|-----------|-------------|
| Taxonómico    | `T ∈ ℝ^N`       | float | Abundancia de microorganismos y la diferencia entre sus abundancias. |
| Socioeconómico| `S ∈ ℝ^N`       | float | Valor contextual que aporta su peso a la sumatoria del score. |
| Funcional     | `F ∈ {0,1,2}^N` | int   | Cuántos genes importantes/beneficiosos hay por ítem (0, 1 o 2). |

`A ∈ ℝ^{10×N}` es la **matriz de contribución** (una entidad distinta de los perfiles): `a_{ji}` pondera el ítem *i* dentro de la muestra *j*. El score por ítem `P = W₁·T + W₂·S + W₃·F` es común a todas las muestras; la separación entre grupos surge de cómo `A` pondera esos ítems por muestra. El producto `Score = A · P` es dimensionalmente directo (10×N · N → 10), sin ninguna partición de columnas de `A`.

> Interpretación descartada: versiones anteriores dividían las columnas de `A` en tres bloques T/S/F y promediaban cada bloque. Esa interpretación queda **sin efecto**.

## Niveles de implementación

```
Nivel 1 — Python
├── sequential.py    — bucle for sobre K candidatos, numpy vectorizado
└── multicore.py     — Pool(cpu_count()) sobre chunks de candidatos

Nivel 2 — C
├── scoring_openmp.c — #pragma omp parallel for sobre K candidatos
└── scoring_mpi.c    — MPI_Scatter / MPI_Reduce sobre K/P candidatos

Nivel 3 — CUDA (notebook Google Colab)
└── scoring_cuda.ipynb — entregable principal; compila el kernel (%%writefile + nvcc)
                          y lo orquesta con PyCUDA. scoring_kernel.cu / scoring_pycuda.py
                          se conservan como fuentes derivadas/embebidas del notebook.
```

## Flujo de datos

```
generate_data.py  (guarda/carga en data/n_{n_items}/, ver DEC-10)
  → matrix_A.npy  (10×N, float32)        matriz de contribución
  → profiles.npy  (N, 3, float32)        perfiles por ítem:
                                            columna 0 = T (taxonómico, float)
                                            columna 1 = S (socioeconómico, float)
                                            columna 2 = F (funcional, int ∈ {0,1,2})
  → labels.npy    (10,  int32)

random_search(A, T, S, F, y, K=100_000, seed=42)
  → muestrear K vectores W sobre el simplex (Dirichlet(1,1,1))
  → para cada W_k:
       P = W_k[0]·T + W_k[1]·S + W_k[2]·F   (vector dim N)
       Score = A · P                         (producto matriz-vector, dim 10)
       auc_k = AUC(y, Score)                 (conteo de pares concordantes)
  → retornar W* = argmax auc_k
```

## Paralelización por nivel

### Nivel 1 — Python multicore

```python
candidatos = rng.dirichlet(ones(3), K)          # generados en el proceso principal
chunks = array_split(candidatos, cpu_count())    # N chunks iguales
Pool.map(evaluate_batch, chunks)                # cada proceso evalúa su chunk
best = max(results, key=lambda r: r[1])
```

### Nivel 2a — C + OpenMP

```c
#pragma omp parallel for reduction(max:best_auc) schedule(dynamic, 64)
for (int k = 0; k < K_CANDIDATES; k++) {
    sample_dirichlet(W_k, seed + k);
    double auc = compute_auc(A, W_k, y, N_SAMPLES);
    if (auc > best_auc) { ... }  /* #pragma omp critical */
}
```

### Nivel 2b — C + MPI

```
root:   genera K candidatos → MPI_Scatter → P procesos reciben K/P candidatos
cada:   evalúa su subconjunto de forma independiente
root:   MPI_Reduce(MPI_MAX) sobre best_auc → selecciona W*
```

### Nivel 3 — CUDA (notebook en Google Colab)

```
Grid  = ceil(K / BLOCK_SIZE),  BLOCK_SIZE = 256
cada hilo k:
    reconstruye W_k desde RNG en device (o lee de arreglo pre-cargado)
    P     = W_k[0]·T + W_k[1]·S + W_k[2]·F   (vector dim N)
    Score = A · P  (A cacheada en __shared__ memory por bloque)
    calcula auc_k via conteo de pares
reduction kernel: selecciona best_auc y best_W
```

El kernel se escribe con `%%writefile scoring_kernel.cu`, se compila con `nvcc` y se invoca
vía PyCUDA dentro de `CUDA/scoring_cuda.ipynb`, ejecutado en un runtime con GPU de Google Colab.

## Métricas de evaluación

| Métrica | Fórmula |
|---------|---------|
| Tiempo T | t_fin − t_inicio (excluye carga de datos) |
| Speedup S | T_Python_seq / T_impl |
| Eficiencia E | S / P  (P = núcleos/hilos) |
| Speedup máx. Amdahl | 1 / ((1−f) + f/P) |
