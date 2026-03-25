# step5_plot.py — Шаг 5: Гистограмма + кривая показательного распределения

import numpy as np
import matplotlib.pyplot as plt
from config import ALPHA


def _theoretical_pdf(x_vals, params):
    alpha = params['alpha']
    return alpha * np.exp(-alpha * x_vals)


def plot_histogram(counts, edges, midpoints, widths, params,
                   filename="lab2_histogram.png"):
    n       = counts.sum()
    omega   = counts / n
    rho_emp = omega / widths

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(edges[:-1], rho_emp, width=widths, align='edge',
           color='steelblue', alpha=0.6, edgecolor='white',
           label='Эмпирическая плотность')

    x_line = np.linspace(0.0, edges[-1], 400)
    y_line = _theoretical_pdf(x_line, params)
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f"Показательное  λ*={params['alpha']:.4f}")

    rho_th = _theoretical_pdf(midpoints, params)
    ax.plot(midpoints, rho_th,  'rs', markersize=6)
    ax.plot(midpoints, rho_emp, 'b^', markersize=6)

    ax.set_xlabel("x")
    ax.set_ylabel("Плотность частоты ρ")
    ax.set_title(f"Гистограмма и теоретическая плотность  (α = {ALPHA})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n  График сохранён: {filename}")