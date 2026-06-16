# Current Phase

**Fase activa**: Fase 3 — C + MPI (`context/project/phases.md`). Fase 2 (C + OpenMP)
`COMPLETADA` (2026-06-15, ver `context/state/active-tasks.md` y
`traceability_data/2026_06_15_14-36.md`).

**Fase 1 — Python Baseline**: `COMPLETADA` (2026-06-14). Implementada y validada según
`context/state/active-tasks.md`; detalle en `traceability_data/2026_06_14_17-18.md`.

## Resultados Fase 1 (seed=42, n_items=50, K=100000, desde `code/`)

> Regenerado 2026-06-15 junto con la corrección DEC-13 (ver "Resultados Fase 2" abajo). Valores
> de tiempo varían levemente entre corridas; las métricas (speedup/efficiency) son estables.

| Implementación        | best_auc | time_seconds | candidates_per_second | speedup | efficiency | speedup_vs_python |
|------------------------|----------|--------------|------------------------|---------|------------|-------------------|
| Python secuencial       | 1.0000   | 93.1382      | 1073.67                | 1.00    | 1.00       | 1.00              |
| Python multicore (w=4)  | 1.0000   | 31.1389      | 3211.41                | 2.9911  | 0.7478     | 2.9911            |

- `best_W = [0.33742523, 0.32787895, 0.33469582]` — idéntico en ambas implementaciones.
- Consistencia (ec. 4) = 2.0000 en ambas (≥ 0.8 ✓).
- `|auc_multicore − auc_secuencial| = 0 < 1e-4` ✓ (RF-04).
- `speedup_multicore = 2.9911× ≥ 1.5×` ✓ (RNF-03). Para Python, el baseline P=1 de DEC-13 ES
  "Python secuencial", por lo que `speedup`/`speedup_vs_python` coinciden (sin cambio de fórmula
  respecto a DEC-11, ya era correcta).

## Resultados Fase 2 (seed=42, n_items=50, K=100000, desde `code/`, gcc 13.3, WSL2, nproc=20)

**Corrección DEC-13 (2026-06-15)**: `speedup`/`efficiency` de "C OpenMP" se calculan ahora
respecto al **baseline propio P=1** (`speedup_openmp(P) = T_openmp(1)/T_openmp(P)`), no contra
Python secuencial. La comparación contra Python secuencial vive en la nueva columna
`speedup_vs_python = T_python_secuencial / T_openmp(P)`. `t_openmp(1) = 0.029690768` (fila
`workers=1` de esta misma corrida); `t_seq_python = 93.13818872600132` (fila "Python
secuencial").

| Implementación | workers (P) | best_auc | time_seconds | candidates_per_second | speedup | efficiency | speedup_vs_python |
|-----------------|:-----------:|----------|--------------|------------------------|---------|------------|-------------------|
| C OpenMP        | 1           | 1.0000   | 0.029691     | 3,368,050.30            | 1.0000  | 1.0000     | 3136.94           |
| C OpenMP        | 2           | 1.0000   | 0.014534     | 6,880,369.57            | 2.0428  | 1.0214     | 6408.25           |
| C OpenMP        | 4           | 1.0000   | 0.007729     | 12,938,255.92           | 3.8415  | 0.9604     | 12050.46          |
| C OpenMP        | 8           | 1.0000   | 0.005501     | 18,177,290.38           | 5.3970  | 0.6746     | 16929.99          |

- `best_W` por P: P=1 → `[0.08767833, 0.53756110, 0.37476056]`, P=2 → igual (empate de AUC),
  P=4 → `[0.35579103, 0.16471726, 0.47949170]`, P=8 → `[0.15604402, 0.14587374, 0.69808223]`.
  `best_auc=1.0000` en los 4 casos (determinista, DEC-12); `best_W` difiere entre P por empates
  de AUC en el óptimo (esperado, no afecta el criterio).
- Consistencia (ec. 4) = 2.0000 en los 4 casos (≥ 0.8 ✓).
- `|best_auc_C − best_auc_PythonSecuencial| = |1.0000 − 1.0000| = 0 < 1e-4` ✓ (RF-04).
- `best_auc ∈ [0.5, 1.0]` ✓ (=1.0).
- **Speedup paralelo (DEC-13)**: `speedup(P=2)=2.04×`, `speedup(P=4)=3.84× ≥ 3×` ✓ (RNF-03),
  `speedup(P=8)=5.40×` — sublineal en P=8 (esperado: overhead de hilos en un cómputo por
  candidato muy pequeño, N=50). `efficiency` ronda 1.0 en P=1/2/4 y cae a 0.67 en P=8.
- `speedup_vs_python` (C vs Python/NumPy/sklearn, columna separada): de 3136.94× (P=1) a
  16929.99× (P=8) — refleja la diferencia de rendimiento entre lenguajes, no el paralelismo;
  no se usa como criterio de RNF-03 (eso es `speedup`).
- `--self-test` de `compute_auc`: 3/3 OK, incluido el caso con empate
  (`scores=[1,2,2,3]`, `y=[0,0,1,1]` → AUC=0.875, igual a `sklearn.roc_auc_score`).
- `valgrind --leak-check=full --show-leak-kinds=all`: `definitely lost: 0 bytes`,
  `indirectly lost: 0 bytes`. Los bloques "possibly lost"/"still reachable" restantes
  pertenecen al runtime de `libgomp` (pool de hilos OpenMP, TLS), no al código propio.
- `gcc -O2 -Wall -Wextra -fopenmp scoring_openmp.c npy_io.c -o scoring_openmp -lm`: sin
  warnings (400 LOC, dentro del límite).

## Cambios relevantes (Fase 2)

- Nuevos `code/C_OpenMP_MPI/npy_io.h` (22 LOC) y `npy_io.c` (126 LOC): parser `.npy` v1.0
  propio (magic+versión+header_len+shape vía `'shape': (...)`, valida `descr`/
  `fortran_order: False`), solo lectura, sin librerías externas (DEC-06/DEC-10).
- `code/C_OpenMP_MPI/scoring_openmp.c` reescrito (394 LOC, < 400): RNG SplitMix64
  sembrado por candidato (`seed+k`) → Dirichlet(1,1,1); `compute_P`/`compute_score` (ecs.
  1-2); `compute_auc` con empates a 0.5 (ec. 3, corrige ISSUE-008); `scoring_consistency`
  (ec. 4); `random_search_openmp` (best local por hilo + `#pragma omp critical`,
  `omp_get_wtime()` solo sobre la búsqueda); `--self-test`; CLI
  `--n-items/--k-candidates/--seed/--threads/--self-test`; `read_sequential_time`/
  `append_benchmark` espejo de `common.py` (esquema de 9 columnas, append-only).
- `code/C_OpenMP_MPI/Makefile`: agregado `npy_io.c` al link de `scoring_openmp`, `-Wextra`
  en `CFLAGS`, y alias `.PHONY openmp`/`mpi` (compatibilidad con `stack.md`/
  `active-tasks.md`: `make openmp`, `make mpi`); target `mpi`/`scoring_mpi.c` sin cambios.
- `code/results/benchmark.csv`: 4 filas nuevas `C OpenMP` (P∈{1,2,4,8}, n_items=50,
  k_candidates=100000), append-only; filas previas de Fase 1 intactas.
- Instalado `valgrind` (no estaba disponible en el entorno) vía `apt-get` para cumplir el
  criterio de "sin fugas".

## Próxima acción

Fase 3 — C + MPI (`scoring_mpi.c`): **plan técnico definido** (2026-06-15,
`traceability_data/2026_06_15_19-22.md`, formalizado en `context/project/decisions.md#DEC-14` y
`context/state/active-tasks.md`). **Pendiente de implementación**.

Estrategia (DEC-14): reutilizar `npy_io.{h,c}` sin cambios y **copiar literalmente** el cómputo
de `scoring_openmp.c`; carga solo en root + `MPI_Bcast`; candidatos por **regeneración local**
`sample_dirichlet(seed+k)` (sin `MPI_Scatter`, DEC-12); reparto contiguo por bloques; reducción
`MPI_Reduce` con `MPI_MAXLOC`/`MPI_DOUBLE_INT` (transporta `k*`); timing `MPI_Wtime` con
`MPI_Barrier` previo; CLI sin `--threads`; registro `C MPI` en `benchmark.csv` (10 cols, solo
rank 0). Criterios de salida y barrido P∈{1,2,4,8} (+ K∈{500k,1M} si RIESGO-05) en
`active-tasks.md` §11–§13.
