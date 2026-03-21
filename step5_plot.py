# step5_plot.py — Шаг 5: Гистограмма + кривая теоретического распределения
#
# По примеру задания: строим гистограмму (плотность частоты ρi)
# и поверх — кривую плотности выбранного распределения.
# Параметры берём из метода моментов (step4).

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from config import DISTRIBUTION, ALPHA


def _theoretical_pdf(x_vals, params):
    dist = DISTRIBUTION
    if dist == 'expon':
        alpha = params['alpha']
        return alpha * np.exp(-alpha * x_vals)
    elif dist == 'normal':
        return stats.norm.pdf(x_vals, loc=params['a'], scale=params['sigma'])
    elif dist == 'laplace':
        return stats.laplace.pdf(x_vals, loc=params['a'],
                                 scale=params['sigma'] / np.sqrt(2))
    elif dist == 'rayleigh':
        return stats.rayleigh.pdf(x_vals, scale=params['sigma'])
    elif dist == 'uniform':
        return stats.uniform.pdf(x_vals, loc=params['a'],
                                 scale=params['b'] - params['a'])
    elif dist == 'chi2':
        return stats.chi2.pdf(x_vals, df=params['k'])
    elif dist == 'student':
        return stats.t.pdf(x_vals, df=params['k'])


def _dist_label(params):
    dist = DISTRIBUTION
    if dist == 'expon':
        return f"Показательное  λ*={params['alpha']:.4f}"
    elif dist == 'normal':
        return f"Нормальное  a*={params['a']:.2f}, σ*={params['sigma']:.2f}"
    elif dist == 'laplace':
        return f"Лапласа  a*={params['a']:.2f}, σ*={params['sigma']:.2f}"
    elif dist == 'rayleigh':
        return f"Рэлея  σ*={params['sigma']:.4f}"
    elif dist == 'uniform':
        return f"Равномерное  a*={params['a']:.2f}, b*={params['b']:.2f}"
    elif dist == 'chi2':
        return f"Хи-квадрат χ²  k*={params['k']:.2f}"
    elif dist == 'student':
        return f"Стьюдента t_k  k*={params['k']:.2f}"


def plot_histogram(counts, edges, midpoints, widths, params, filename="lab2_histogram.png"):
    n      = counts.sum()
    omega  = counts / n
    rho_emp = omega / widths

    fig, ax = plt.subplots(figsize=(9, 5))

    # Гистограмма (bar — по плотности частоты, как в примере задания)
    ax.bar(edges[:-1], rho_emp, width=widths, align='edge',
           color='steelblue', alpha=0.6, edgecolor='white', label='Эмпирическая плотность')

    # Теоретическая кривая
    x_lo = max(edges[0], -50)
    x_hi = edges[-1]
    if DISTRIBUTION in ('expon', 'rayleigh'):
        x_lo = 0.0
    x_line = np.linspace(x_lo, x_hi, 400)
    y_line = _theoretical_pdf(x_line, params)
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=_dist_label(params))

    # Точки серединам интервалов (как в примере — на графике сравнения)
    rho_th = _theoretical_pdf(midpoints, params)
    ax.plot(midpoints, rho_th,    'rs', markersize=6)
    ax.plot(midpoints, rho_emp,   'b^', markersize=6)

    ax.set_xlabel("x")
    ax.set_ylabel("Плотность частоты ρ")
    ax.set_title(f"Гистограмма и теоретическая плотность  (α = {ALPHA})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n  График сохранён: {filename}")