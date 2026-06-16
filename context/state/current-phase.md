# Current Phase

**Fase activa**: Fase 4 — CUDA (notebook en Google Colab) — `PLANIFICADA` (2026-06-16, plan en
`context/state/active-tasks.md` y `traceability_data/2026_06_16_00-43.md`; estrategia en
**DEC-15**). Pendiente de implementación; **se ejecuta en Google Colab** (runtime GPU), no en el
entorno local (RIESGO-01). Fase 3 (C + MPI) `COMPLETADA` (2026-06-15, ver
`traceability_data/2026_06_15_19-32.md`).

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

## Resultados Fase 3 (seed=42, n_items=50, K=100000, desde `code/`, OpenMPI 4.1.6, WSL2)

| P | best_auc | time_seconds | candidates_per_second | speedup | efficiency | speedup_vs_python |
|:-:|----------|--------------|------------------------|---------|------------|-------------------|
| 1 | 1.0000   | 0.039927     | 2,504,562              | 1.0000  | 1.0000     | 2411.99×          |
| 2 | 1.0000   | 0.022423     | 4,459,637              | 1.7806  | 0.8903     | 4294.80×          |
| 4 | 1.0000   | 0.010940     | 9,140,351              | 3.6495  | 0.9124     | 8802.50×          |
| 8 | 1.0000   | 0.007415     | 13,485,991             | 5.3846  | 0.6731     | 12987.52×         |

- `best_W = [0.08767833, 0.53756110, 0.37476056]` — idéntico para todo P (MAXLOC desempate menor k).
- Consistencia (ec. 4) = 2.0000 en los 4 casos (≥ 0.8 ✓).
- `|best_auc_MPI − best_auc_PySeq| = |1.0000 − 1.0000| = 0 < 1e-4` ✓ (RF-04).
- `--self-test` de `compute_auc`: 3/3 OK, incluido caso con empate (AUC=0.875).
- `speedup(P=4) = 3.65× ≥ 3×` ✓ (RNF-03). Superlinealidad aparente en P=2/4 (eficiencia >0.89).
- RIESGO-05 no materializado: MPI overhead amortizado con K=100k.
- `mpicc -O2 -Wall -Wextra`: sin warnings; 389 LOC < 400 ✓.

## Cambios relevantes (Fase 3)

- `code/C_OpenMP_MPI/scoring_mpi.c` reescrito (389 LOC): copia literal del cómputo de Fase 2
  (`splitmix64_*`, `sample_dirichlet`, `compute_P`/`compute_score`/`compute_auc`/
  `scoring_consistency`, `self_test`, `load_dataset`, `read_last_time`/`append_benchmark`);
  `<omp.h>` reemplazado por `<mpi.h>`; sin OpenMP ni CUDA. Carga solo en rank 0 + `MPI_Bcast`;
  regeneración local de candidatos (sin `MPI_Scatter`); `MPI_MAXLOC`/`MPI_DOUBLE_INT` para
  reducción y reconstrucción de `best_W`; timing `MPI_Wtime`+`MPI_Barrier`; CLI sin `--threads`.
- `code/C_OpenMP_MPI/Makefile`: target `scoring_mpi` ahora linkea `npy_io.c`.
- `code/results/benchmark.csv`: 4 filas nuevas `C MPI` (P∈{1,2,4,8}), append-only.
- OpenMPI 4.1.6 instalado en el entorno (RIESGO-02 resuelto).

## Plan Fase 4 (PLANIFICADA — 2026-06-16)

Estrategia formalizada en **DEC-15**; plan técnico completo (15 tareas + criterios de salida) en
`context/state/active-tasks.md` (`traceability_data/2026_06_16_00-43.md`, iteración 1). Puntos
clave: notebook `CUDA/scoring_cuda.ipynb` con RNG SplitMix64 `__device__` (`seed+k`, sin transferir
`W_pool`), kernel `scoring_kernel` (un hilo/candidato, `__shared__` para `A`/`profiles`,
`BLOCK_SIZE=256`, AUC con empates +0.5), reducción con `np.argmax` en host, compilación
`%%writefile`+`SourceModule(-O2)`, timing que excluye carga (`cudaEvent_t` + `perf_counter`/
`Context.synchronize()`), self-test (empate=0.875 + `n_3`/K=100 vs mirror SplitMix64), y una sola
fila `CUDA` en `benchmark.csv` (`workers=1`, `speedup=1.0`, `efficiency=1.0`,
`speedup_vs_python = 96.30376639 / T_cuda`). Objetivos: `|ΔAUC|<1e-4` vs Python secuencial,
consistencia ≥ 0.8, `speedup_vs_python ≥ 5×` (RNF-03), con modelo de GPU registrado (RIESGO-06).

## Próxima acción

Implementar la Fase 4 en Google Colab según el plan (DEC-15 / `active-tasks.md`). **No ejecutable
en el entorno local** (sin GPU asumida, RIESGO-01): el prompt de ejecución queda planteado en
`instruction.md` para correrse después en Colab. Ver `context/project/phases.md#Fase 4`.
