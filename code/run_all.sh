#!/usr/bin/env bash
#
# run_all.sh — Benchmark automatizado para Scoring Metagenómico HPC.
#
# Ejecuta todas las fases (excepto CUDA, que corre en Google Colab)
# y registra resultados en results/benchmark.csv (append-only).
#
# Uso: desde code/
#   bash run_all.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"

SEED=42
WORKERS_PY=4
C_THREADS=(1 2 4 8)
MPI_PROCS=(1 2 4 8)

read -r -p "Numero de items (N) [50]: " input
N_ITEMS="${input:-50}"
read -r -p "Numero de candidatos (K) [100000]: " input
K="${input:-100000}"
echo ""
echo "  N_ITEMS=${N_ITEMS}, K=${K}, SEED=${SEED}"
echo ""

PARSE="python python/sequential.py"
MC="python python/multicore.py"
OPENMP="./C_OpenMP_MPI/scoring_openmp"
MPI="mpirun"
TEST="python -m pytest python/tests/ -q"
LINT="python -m ruff check python/"
PLOTS="python python/plot_benchmark.py"

step()  { echo ""; echo "================================================"; echo "  $1"; echo "================================================"; }
ok()    { echo "  ✓ $1"; }
fail()  { echo "  ✗ $1"; exit 1; }

# ---- Prerrequisitos ----
step "[00] Verificando prerequisitos"

command -v python3 &>/dev/null || fail "python3 no encontrado"
command -v uv       &>/dev/null || fail "uv no encontrado"

OPENMP_BIN="C_OpenMP_MPI/scoring_openmp"
MPI_BIN="C_OpenMP_MPI/scoring_mpi"
if [ ! -x "$OPENMP_BIN" ] || [ ! -x "$MPI_BIN" ]; then
    echo "  [INFO] Compilando C..."
    make -C C_OpenMP_MPI openmp mpi
fi
ok "Compilacion C OK"

# ---- Generar datos ----
step "[01] Generando dataset n=${N_ITEMS}"
uv run python data/generate_data.py
ok "Dataset listo en data/n_${N_ITEMS}/"

# ---- Fase 1: Python secuencial ----
step "[02] Python secuencial (n=${N_ITEMS}, K=${K}, seed=${SEED})"
uv run $PARSE --n-items "$N_ITEMS" --k-candidates "$K" --seed "$SEED"
ok "Python secuencial completado"

# ---- Fase 1: Python multicore ----
step "[03] Python multicore (n=${N_ITEMS}, K=${K}, workers=${WORKERS_PY}, seed=${SEED})"
uv run $MC --n-items "$N_ITEMS" --k-candidates "$K" --workers "$WORKERS_PY" --seed "$SEED"
ok "Python multicore completado"

# ---- Fase 2: C + OpenMP ----
for T in "${C_THREADS[@]}"; do
    step "[04] C OpenMP (threads=${T}, n=${N_ITEMS}, K=${K}, seed=${SEED})"
    $OPENMP --n-items "$N_ITEMS" --k-candidates "$K" --seed "$SEED" --threads "$T"
    ok "C OpenMP threads=${T} completado"
done

# ---- Fase 3: C + MPI ----
for P in "${MPI_PROCS[@]}"; do
    step "[05] C MPI (procesos=${P}, n=${N_ITEMS}, K=${K}, seed=${SEED})"
    $MPI -np "$P" --oversubscribe "$MPI_BIN" --n-items "$N_ITEMS" --k-candidates "$K" --seed "$SEED"
    ok "C MPI procesos=${P} completado"
done

# ---- Tests y lint ----
step "[06] Tests"
uv run $TEST || fail "Tests fallaron"
ok "Tests OK"

step "[07] Lint"
uv run $LINT || fail "Lint fallo"
ok "Lint OK"

# ---- Resultados ----
step "[08] Resultados"
echo ""
cat results/benchmark.csv
echo ""

# ---- Graficas ----
step "[09] Generando graficas"
uv run $PLOTS
ok "Graficas en results/plots/"

step "[10] Copiando graficas a Windows"
WINDOWS_PLOTS="/mnt/c/Users/julia/Desktop/PDC_FINAL/PDC_FINAL/code/results/plots/"
if [ -d "$WINDOWS_PLOTS" ]; then
    cp results/plots/*.png "$WINDOWS_PLOTS"
    ok "Graficas copiadas a Windows"
else
    echo "  [INFO] Ruta Windows no disponible; graficas solo en WSL"
fi

# ---- Resumen ----
echo ""
echo "================================================"
echo "  BENCHMARK COMPLETADO"
echo "================================================"
echo "  results/benchmark.csv  — metricas (append-only)"
echo "  results/plots/         — graficas comparativas"
echo "================================================"
