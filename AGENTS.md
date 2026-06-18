# AGENTS.md — Scoring Metagenómico HPC

## Project overview

Multi-level HPC Random Search over a 3D simplex (W₁,W₂,W₃) to maximize AUC for binary classification of 10 metagenomic samples (5 healthy, 5 diseased). Phases must execute **in order**: Python → C+OpenMP → C+MPI → CUDA → Benchmarking → Documentation.

## Key commands (all from `code/`)

```bash
uv sync                          # create .venv + install deps
uv run python data/generate_data.py
uv run python/python/sequential.py --n-items 50 --k-candidates 100000 --seed 42
uv run python/python/multicore.py --n-items 50 --k-candidates 100000 --workers 4 --seed 42
uv run pytest python/tests/ -q
uv run ruff check python/
```

C targets (from `code/C_OpenMP_MPI/`):
```bash
make openmp && ./scoring_openmp
make mpi && mpirun -np 4 ./scoring_mpi
```

## Architecture gotchas

- **T/S/F are independent per-item profiles** (profiles.npy columns), **NOT** blocks of A columns. Common mistake: do not partition A into T/S/F blocks. Score = A @ P with P = W₁·T + W₂·S + W₃·F.
- `seed=42` **everywhere** (Python, C, CUDA).
- C and CUDA source files are **placeholders** (scoring_openmp.c, scoring_mpi.c, scoring_kernel.cu all have `// TODO`). They need full implementation.
- CUDA phase delivers as `CUDA/scoring_cuda.ipynb` run in **Google Colab** (no local GPU assumed). Upload `.npy` files to Colab runtime.
- `benchmark.csv` is **append-only** (9-column schema); never overwrite rows.

## Validation constraints

- |ΔAUC| < 1e-4 vs Python sequential (same K, same seed).
- AUC ∈ [0.5, 1.0]; values outside indicate bugs.
- scoring_consistency ≥ 0.8 (ec. 4).
- W must satisfy simplex: W₁+W₂+W₃ = 1, Wᵢ ≥ 0.

## CI / toolchain quirks

- **uv** manages Python env — always `uv run <cmd>`, never activate `.venv` manually.
- Ruff line-length: 100 (`pyproject.toml`).
- `[tool.pytest.ini_options] pythonpath = ["python"]` in pyproject.toml enables `from common import ...` without sys.path hacks.
- Python max 200 LOC/file, C max 400 LOC/file.
- `results/benchmark.csv` appends rows; `read_sequential_time()` looks up last matching sequential row for speedup calculation.

## What's next

Fase 1 (Python baseline) is **complete**. Fase 2 (C+OpenMP) is next — implement `scoring_openmp.c` (load data via cnpy or similar, Random Search with `#pragma omp parallel for`), validate AUC equivalence.

## Traceability protocol

`instruction.md` at repo root has a traceability protocol for AI sessions (creates per-session files in `traceability_data/`). Honor it when starting a new conversation.
