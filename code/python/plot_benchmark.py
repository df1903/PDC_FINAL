"""Genera gráficas comparativas completas desde results/benchmark.csv."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore

BENCHMARK_CSV = Path("results") / "benchmark.csv"
PLOTS_DIR = Path("results") / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
})


def load_benchmark(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["workers"] = int(row["workers"])
            row["time_seconds"] = float(row["time_seconds"])
            row["candidates_per_second"] = float(row["candidates_per_second"])
            row["best_auc"] = float(row["best_auc"])
            if row["speedup"]:
                row["speedup"] = float(row["speedup"])
            else:
                row["speedup"] = None
            if row["efficiency"]:
                row["efficiency"] = float(row["efficiency"])
            else:
                row["efficiency"] = None
            if row["speedup_vs_python"]:
                row["speedup_vs_python"] = float(row["speedup_vs_python"])
            else:
                row["speedup_vs_python"] = None
            rows.append(row)
    return rows


def deduplicate(rows: list[dict]) -> list[dict]:
    seen = {}
    for r in rows:
        key = (r["implementation"], r["workers"])
        seen[key] = r
    return list(seen.values())


COLORS = {
    "Python secuencial": "#2E86AB",
    "Python multicore": "#A23B72",
    "C OpenMP": "#F18F01",
    "C MPI": "#C73E1D",
}
MARKERS = {
    "Python secuencial": "o",
    "Python multicore": "s",
    "C OpenMP": "^",
    "C MPI": "D",
}
LINESTYLES = {
    "C OpenMP": "-",
    "C MPI": "--",
}
LABELS = {
    "Python secuencial": "Python secuencial",
    "Python multicore": "Python multicore",
    "C OpenMP": "C + OpenMP",
    "C MPI": "C + MPI",
}


def plot_speedup_scalability(rows: list[dict]):
    fig, ax = plt.subplots()
    impls = ["C OpenMP", "C MPI"]
    for impl in impls:
        pts = sorted([r for r in rows if r["implementation"] ==
                     impl and r["speedup"]], key=lambda x: x["workers"])
        ws = [p["workers"] for p in pts]
        sp = [p["speedup"] for p in pts]
        ax.plot(ws, sp, marker=MARKERS[impl], linestyle=LINESTYLES[impl], color=COLORS[impl],
                label=LABELS[impl], linewidth=2, markersize=8)
    ideal = sorted(set(r["workers"]
                   for r in rows if r["implementation"] in impls))
    ax.plot(ideal, ideal, "k--", alpha=0.4,
            label="Speedup ideal", linewidth=1.5)
    ax.set_xlabel("Número de workers / procesos (P)")
    ax.set_ylabel("Speedup (vs propio P=1)")
    ax.set_title("Escalabilidad: Speedup vs Workers/Procesos")
    ax.set_xticks(ideal)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "speedup_scalability.png")
    plt.close(fig)


def plot_efficiency(rows: list[dict]):
    fig, ax = plt.subplots()
    impls = ["C OpenMP", "C MPI"]
    for impl in impls:
        pts = sorted([r for r in rows if r["implementation"] ==
                     impl and r["efficiency"]], key=lambda x: x["workers"])
        ws = [p["workers"] for p in pts]
        ef = [p["efficiency"] for p in pts]
        ax.plot(ws, ef, marker=MARKERS[impl], linestyle=LINESTYLES[impl], color=COLORS[impl],
                label=LABELS[impl], linewidth=2, markersize=8)
    ax.axhline(y=1.0, color="k", linestyle="--",
               alpha=0.4, label="Eficiencia ideal")
    ax.set_xlabel("Número de workers / procesos (P)")
    ax.set_ylabel("Eficiencia")
    ax.set_title("Eficiencia vs Workers/Procesos")
    ax.set_xticks(sorted(set(r["workers"]
                  for r in rows if r["implementation"] in impls)))
    ax.set_ylim(0, 1.3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "efficiency.png")
    plt.close(fig)


def plot_time_comparison(rows: list[dict]):
    fig, ax = plt.subplots()
    impls = ["Python secuencial", "Python multicore", "C OpenMP", "C MPI"]
    labels_display = []
    times = []
    colors_bar = []
    for impl in impls:
        if impl == "Python secuencial":
            pts = [r for r in rows if r["implementation"] == impl]
            t = max(r["time_seconds"] for r in pts) if len(
                pts) > 1 else pts[0]["time_seconds"]
            labels_display.append(LABELS[impl])
            times.append(t)
            colors_bar.append(COLORS[impl])
        elif impl == "Python multicore":
            pts = [r for r in rows if r["implementation"] == impl]
            t = max(r["time_seconds"] for r in pts) if len(
                pts) > 1 else pts[0]["time_seconds"]
            labels_display.append(LABELS[impl] + " (4 workers)")
            times.append(t)
            colors_bar.append(COLORS[impl])
        else:
            for p in [1, 2, 4, 8]:
                match = [r for r in rows if r["implementation"]
                         == impl and r["workers"] == p]
                if match:
                    labels_display.append(f"{LABELS[impl]} P={p}")
                    times.append(match[0]["time_seconds"])
                    colors_bar.append(COLORS[impl])

    bars = ax.bar(range(len(times)), times, color=colors_bar,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(labels_display, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Tiempo (segundos)")
    ax.set_title("Tiempo de ejecución por implementación (escala log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, t in zip(bars, times):
        if t < 0.1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{t:.4f}s",
                    ha="center", va="bottom", fontsize=7, rotation=45)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{t:.2f}s",
                    ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "time_comparison.png")
    plt.close(fig)


def plot_speedup_vs_python(rows: list[dict]):
    fig, ax = plt.subplots()
    impls = ["Python multicore", "C OpenMP", "C MPI"]
    labels_display = []
    sp_vs_py = []
    colors_bar = []
    for impl in impls:
        if impl == "Python multicore":
            pts = [r for r in rows if r["implementation"]
                   == impl and r["speedup_vs_python"]]
            t = max(p["speedup_vs_python"] for p in pts) if len(
                pts) > 1 else pts[0]["speedup_vs_python"]
            labels_display.append(LABELS[impl] + " (4 workers)")
            sp_vs_py.append(t)
            colors_bar.append(COLORS[impl])
        else:
            for p in [1, 2, 4, 8]:
                match = [r for r in rows if r["implementation"] ==
                         impl and r["workers"] == p and r["speedup_vs_python"]]
                if match:
                    labels_display.append(f"{LABELS[impl]} P={p}")
                    sp_vs_py.append(match[0]["speedup_vs_python"])
                    colors_bar.append(COLORS[impl])

    bars = ax.bar(range(len(sp_vs_py)), sp_vs_py,
                  color=colors_bar, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(sp_vs_py)))
    ax.set_xticklabels(labels_display, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Speedup vs Python secuencial (×)")
    ax.set_title("Aceleración vs Python secuencial (escala log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, t in zip(bars, sp_vs_py):
        label = f"{t:.0f}×" if t > 100 else (
            f"{t:.2f}×" if t >= 1 else f"{t:.3f}×")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                ha="center", va="bottom", fontsize=7, rotation=45)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "speedup_vs_python.png")
    plt.close(fig)


def plot_throughput(rows: list[dict]):
    fig, ax = plt.subplots()
    impls = ["C OpenMP", "C MPI"]
    for impl in impls:
        pts = sorted([r for r in rows if r["implementation"]
                     == impl], key=lambda x: x["workers"])
        ws = [p["workers"] for p in pts]
        cps = [p["candidates_per_second"] / 1e6 for p in pts]
        ax.plot(ws, cps, marker=MARKERS[impl], linestyle=LINESTYLES[impl], color=COLORS[impl],
                label=LABELS[impl], linewidth=2, markersize=8)

    py_pts = [r for r in rows if r["implementation"] == "Python secuencial"]
    if py_pts:
        ax.axhline(y=py_pts[0]["candidates_per_second"] / 1e6, color=COLORS["Python secuencial"],
                   linestyle=":", alpha=0.7, label=f"Python secuencial ({py_pts[0]['candidates_per_second']/1e6:.0f}M)")
    py_mc = [r for r in rows if r["implementation"] == "Python multicore"]
    if py_mc:
        ax.axhline(y=max(p["candidates_per_second"] for p in py_mc) / 1e6, color=COLORS["Python multicore"],
                   linestyle=":", alpha=0.7, label=f"Python multicore ({max(p['candidates_per_second'] for p in py_mc)/1e6:.0f}M)")

    ax.set_xlabel("Número de workers / procesos (P)")
    ax.set_ylabel("Candidatos por segundo (millones)")
    ax.set_title("Rendimiento: candidatos evaluados por segundo")
    ax.set_xticks(sorted(set(r["workers"]
                  for r in rows if r["implementation"] in impls)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "throughput.png")
    plt.close(fig)


def plot_summary_table(rows: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    impls = [
        ("Python secuencial", 1),
        ("Python multicore", 4),
    ]
    for impl_name in ["C OpenMP", "C MPI"]:
        for p in [1, 2, 4, 8]:
            impls.append((impl_name, p))

    table_data = []
    col_labels = ["Implementación", "Workers", "AUC",
                  "Tiempo (s)", "Speedup", "Eficiencia", "× vs Python"]
    for impl, w in impls:
        match = [r for r in rows if r["implementation"]
                 == impl and r["workers"] == w]
        if not match:
            continue
        r = match[0]
        sp = f"{r['speedup']:.2f}×" if r.get("speedup") else "—"
        ef = f"{r['efficiency']:.2f}" if r.get("efficiency") else "—"
        vp = f"{r['speedup_vs_python']:.1f}×" if r.get(
            "speedup_vs_python") else "—"
        label = LABELS.get(impl, impl)
        table_data.append(
            [label, str(w), f"{r['best_auc']:.4f}", f"{r['time_seconds']:.6f}", sp, ef, vp])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2E86AB")
            cell.set_text_props(color="white", fontweight="bold")
        elif row < 3:
            cell.set_facecolor("#E8F4F8")
        elif row < 7:
            cell.set_facecolor("#FFF3E0")
        else:
            cell.set_facecolor("#FBE9E7")

    ax.set_title("Resumen comparativo de todas las implementaciones",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "summary_table.png")
    plt.close(fig)


def main():
    rows = load_benchmark(BENCHMARK_CSV)
    rows = deduplicate(rows)
    print(
        f"Generando gráficas desde {len(rows)} filas únicas de benchmark.csv")

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
