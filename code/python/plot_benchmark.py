"""Genera gráficas comparativas desde results/benchmark.csv.

Usa TODAS las filas del CSV. Cada configuración (n_items, k_candidates)
aparece como una serie/línea separada con su etiqueta.

Uso: desde code/
    uv run python python/plot_benchmark.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

BENCHMARK_CSV = Path("results") / "benchmark.csv"
PLOTS_DIR = Path("results") / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.figsize": (10, 6),
})


def parse_row(row: dict) -> dict | None:
    try:
        r = dict(row)
        r["n_items"] = int(r["n_items"])
        r["k_candidates"] = int(r["k_candidates"])
        r["workers"] = int(r["workers"])
        r["time_seconds"] = float(r["time_seconds"])
        r["candidates_per_second"] = float(r["candidates_per_second"])
        r["best_auc"] = float(r["best_auc"])
        r["speedup"] = float(r["speedup"]) if r["speedup"] else None
        r["efficiency"] = float(r["efficiency"]) if r["efficiency"] else None
        r["speedup_vs_python"] = float(r["speedup_vs_python"]) if r["speedup_vs_python"] else None
        r["cfg"] = f"n={r['n_items']}, K={r['k_candidates']}"
        return r
    except (ValueError, KeyError):
        return None


def load_all(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = parse_row(row)
            if r:
                rows.append(r)
    seen = {}
    for r in rows:
        key = (r["implementation"], r["n_items"], r["k_candidates"], r["workers"])
        seen[key] = r
    return list(seen.values())


def label_for(r: dict) -> str:
    return f"{r['implementation']} ({r['cfg']})"


IMPL_COLORS = {
    "Python secuencial": "#2E86AB",
    "Python multicore": "#A23B72",
    "C OpenMP": "#F18F01",
    "C MPI": "#C73E1D",
}
IMPL_MARKERS = {"Python secuencial": "o", "Python multicore": "s", "C OpenMP": "^", "C MPI": "D"}
IMPL_LS = {"C OpenMP": "-", "C MPI": "--"}


def _config_groups(rows: list[dict], impls: list[str]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        if r["implementation"] in impls:
            groups[(r["implementation"], r["n_items"], r["k_candidates"])].append(r)
    return groups


def plot_speedup_scalability(rows: list[dict]):
    fig, ax = plt.subplots()
    groups = _config_groups(rows, ["C OpenMP", "C MPI"])

    for (impl, n, k), pts in sorted(groups.items()):
        pts = sorted([p for p in pts if p["speedup"]], key=lambda x: x["workers"])
        if len(pts) < 2:
            continue
        ws = [p["workers"] for p in pts]
        sp = [p["speedup"] for p in pts]
        lbl = f"{impl} (n={n}, K={k})"
        ls = IMPL_LS.get(impl, "-")
        ax.plot(ws, sp, marker=IMPL_MARKERS.get(impl, "o"), linestyle=ls,
                color=IMPL_COLORS.get(impl, "#333"), label=lbl, linewidth=1.8, markersize=7)

    all_w = sorted(set(p["workers"] for p in rows if p["implementation"] in ("C OpenMP", "C MPI")))
    if all_w:
        ax.plot(all_w, all_w, "k--", alpha=0.4, label="Speedup ideal", linewidth=1.5)
        ax.set_xticks(all_w)
    ax.set_xlabel("Workers / procesos (P)")
    ax.set_ylabel("Speedup (vs propio P=1)")
    ax.set_title("Escalabilidad: Speedup vs Workers/Procesos")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "speedup_scalability.png")
    plt.close(fig)


def plot_efficiency(rows: list[dict]):
    fig, ax = plt.subplots()
    groups = _config_groups(rows, ["C OpenMP", "C MPI"])

    for (impl, n, k), pts in sorted(groups.items()):
        pts = sorted([p for p in pts if p["efficiency"]], key=lambda x: x["workers"])
        if len(pts) < 2:
            continue
        ws = [p["workers"] for p in pts]
        ef = [p["efficiency"] for p in pts]
        lbl = f"{impl} (n={n}, K={k})"
        ls = IMPL_LS.get(impl, "-")
        ax.plot(ws, ef, marker=IMPL_MARKERS.get(impl, "o"), linestyle=ls,
                color=IMPL_COLORS.get(impl, "#333"), label=lbl, linewidth=1.8, markersize=7)

    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.4, label="Eficiencia ideal")
    all_w = sorted(set(p["workers"] for p in rows if p["implementation"] in ("C OpenMP", "C MPI")))
    if all_w:
        ax.set_xticks(all_w)
    ax.set_xlabel("Workers / procesos (P)")
    ax.set_ylabel("Eficiencia")
    ax.set_title("Eficiencia vs Workers/Procesos")
    ax.set_ylim(0, 1.35)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "efficiency.png")
    plt.close(fig)


def plot_time_comparison(rows: list[dict]):
    fig, ax = plt.subplots(figsize=(12, 6))
    all_cfgs = sorted(set((r["implementation"], r["n_items"], r["k_candidates"]) for r in rows))
    cfgs_done = set()
    labels_display = []
    times = []
    colors = []
    for impl, n, k in all_cfgs:
        cfg_key = (impl, n, k)
        if cfg_key in cfgs_done:
            continue
        cfgs_done.add(cfg_key)
        cfgs_workers = {r["workers"] for r in rows if r["implementation"] == impl and (r["n_items"], r["k_candidates"]) == (n, k)}
        for w in sorted(cfgs_workers):
            match = [r for r in rows if r["implementation"] == impl and r["workers"] == w and r["n_items"] == n and r["k_candidates"] == k]
            if match:
                labels_display.append(f"{impl}\nn={n} K={k}\nP={w}")
                times.append(match[0]["time_seconds"])
                colors.append(IMPL_COLORS.get(impl, "#666"))

    bars = ax.bar(range(len(times)), times, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(labels_display, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_title("Tiempo de ejecución (escala log) — todas las configuraciones")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, t in zip(bars, times):
        va = "top" if t < 0.1 else "bottom"
        label = f"{t:.4f}s" if t < 0.1 else f"{t:.2f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                ha="center", va=va, fontsize=6, rotation=45)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "time_comparison.png")
    plt.close(fig)


def plot_speedup_vs_python(rows: list[dict]):
    fig, ax = plt.subplots(figsize=(12, 6))
    impls_order = ["Python multicore", "C OpenMP", "C MPI"]
    labels_display = []
    sp_vs_py = []
    colors = []
    for impl in impls_order:
        cfgs = sorted(set((r["n_items"], r["k_candidates"]) for r in rows if r["implementation"] == impl))
        for n, k in cfgs:
            for w in sorted(set(r["workers"] for r in rows if r["implementation"] == impl and r["n_items"] == n and r["k_candidates"] == k)):
                match = [r for r in rows if r["implementation"] == impl and r["workers"] == w and r["n_items"] == n and r["k_candidates"] == k and r["speedup_vs_python"]]
                if match:
                    labels_display.append(f"{impl} P={w}\nn={n} K={k}")
                    sp_vs_py.append(match[0]["speedup_vs_python"])
                    colors.append(IMPL_COLORS.get(impl, "#666"))

    bars = ax.bar(range(len(sp_vs_py)), sp_vs_py, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(sp_vs_py)))
    ax.set_xticklabels(labels_display, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Speedup vs Python secuencial (×)")
    ax.set_title("Aceleración vs Python secuencial — todas las configuraciones")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, t in zip(bars, sp_vs_py):
        label = f"{t:.0f}×" if t > 100 else (f"{t:.2f}×" if t >= 1 else f"{t:.3f}×")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                ha="center", va="bottom", fontsize=6, rotation=45)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "speedup_vs_python.png")
    plt.close(fig)


def plot_throughput(rows: list[dict]):
    fig, ax = plt.subplots()
    groups = _config_groups(rows, ["C OpenMP", "C MPI"])

    for (impl, n, k), pts in sorted(groups.items()):
        pts = sorted(pts, key=lambda x: x["workers"])
        ws = [p["workers"] for p in pts]
        cps = [p["candidates_per_second"] / 1e6 for p in pts]
        lbl = f"{impl} (n={n}, K={k})"
        ls = IMPL_LS.get(impl, "-")
        ax.plot(ws, cps, marker=IMPL_MARKERS.get(impl, "o"), linestyle=ls,
                color=IMPL_COLORS.get(impl, "#333"), label=lbl, linewidth=1.8, markersize=7)

    for impl_name, color_key in [("Python secuencial", "Python secuencial"), ("Python multicore", "Python multicore")]:
        vals = [r["candidates_per_second"] for r in rows if r["implementation"] == impl_name]
        if vals:
            val = max(vals)
            ax.axhline(y=val / 1e6, color=IMPL_COLORS.get(color_key, "#666"),
                       linestyle=":", alpha=0.7,
                       label=f"{impl_name} ({val/1e6:.1f}M/s máx)")

    all_w = sorted(set(p["workers"] for p in rows if p["implementation"] in ("C OpenMP", "C MPI")))
    if all_w:
        ax.set_xticks(all_w)
    ax.set_xlabel("Workers / procesos (P)")
    ax.set_ylabel("Candidatos por segundo (millones)")
    ax.set_title("Rendimiento: candidatos evaluados por segundo")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "throughput.png")
    plt.close(fig)


def plot_summary_table(rows: list[dict]):
    configs = sorted(set((r["implementation"], r["n_items"], r["k_candidates"]) for r in rows))
    table_data = []
    col_labels = ["Implementación", "N", "K", "Workers", "AUC", "T (s)", "Speedup", "Efic.", "×Py"]
    for impl, n, k in configs:
        fs = sorted(set(r["workers"] for r in rows if r["implementation"] == impl and r["n_items"] == n and r["k_candidates"] == k))
        for w in fs:
            match = [r for r in rows if r["implementation"] == impl and r["workers"] == w and r["n_items"] == n and r["k_candidates"] == k]
            if not match:
                continue
            r = match[0]
            sp = f"{r['speedup']:.2f}" if r.get("speedup") else "—"
            ef = f"{r['efficiency']:.2f}" if r.get("efficiency") else "—"
            vp = f"{r['speedup_vs_python']:.0f}" if r.get("speedup_vs_python") else "—"
            table_data.append([
                impl, str(n), str(k), str(w),
                f"{r['best_auc']:.4f}", f"{r['time_seconds']:.6f}",
                sp, ef, vp,
            ])

    if not table_data:
        return

    nrows = len(table_data) + 1
    fig, ax = plt.subplots(figsize=(12, 0.4 * nrows + 1))
    ax.axis("off")

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center", rowLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2E86AB")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")

    ax.set_title("Resumen completo de benchmarks", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "summary_table.png")
    plt.close(fig)


def main():
    rows = load_all(BENCHMARK_CSV)
    if not rows:
        print("[ERROR] No se encontraron filas válidas en benchmark.csv")
        return

    configs = sorted(set((r["implementation"], r["n_items"], r["k_candidates"]) for r in rows))
    print(f"Total filas únicas: {len(rows)}")
    print(f"Configuraciones encontradas: {len(configs)}")
    for impl, n, k in configs:
        ws = sorted(set(r["workers"] for r in rows if r["implementation"] == impl and r["n_items"] == n and r["k_candidates"] == k))
        print(f"  {impl}: n={n}, K={k}, workers={ws}")

    plot_speedup_scalability(rows)
    print("  ✓ speedup_scalability.png")

    plot_efficiency(rows)
    print("  ✓ efficiency.png")

    plot_time_comparison(rows)
    print("  ✓ time_comparison.png")

    plot_speedup_vs_python(rows)
    print("  ✓ speedup_vs_python.png")

    plot_throughput(rows)
    print("  ✓ throughput.png")

    plot_summary_table(rows)
    print("  ✓ summary_table.png")

    print(f"\nTodas las gráficas guardadas en {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
