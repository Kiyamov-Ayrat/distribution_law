# step7_mnk.py — Шаг 7-8: Уточнение параметров МНК + итоговый результат
#
# Минимизируем статистику Пирсона ρ по параметру α.
# α₀ = P(ρ > ρ_набл) при истинности H₀.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from config import ALPHA
from step6_pearson import _interval_prob


def _rho_func(param_vec, counts, edges):
    """Статистика ρ как функция параметра α (для минимизации)."""
    alpha = param_vec[0]
    if alpha <= 0:
        return 1e9
    n      = counts.sum()
    probs  = _interval_prob(edges, {'alpha': alpha})
    probs  = np.clip(probs, 1e-12, None)
    np_    = n * probs
    return float(np.sum((counts - np_) ** 2 / np_))


def mnk_refine(params_mm, counts, edges, rho_before):
    result = optimize.minimize(
        _rho_func, x0=[params_mm['alpha']], args=(counts, edges),
        method='L-BFGS-B', bounds=[(1e-6, None)],
        options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 1000}
    )

    alpha_opt  = result.x[0]
    params_mnk = {'alpha': alpha_opt}
    rho_after  = result.fun

    m        = len(counts)
    df       = m - 1 - 1       # l = 1
    rho_cr   = stats.chi2.ppf(1 - ALPHA, df)
    p_val    = 1.0 - stats.chi2.cdf(rho_after, df)
    accepted = rho_after < rho_cr

    print("\n" + "=" * 55)
    print("ШАГ 7: УТОЧНЕНИЕ ПАРАМЕТРОВ (МНК)")
    print("=" * 55)
    print(f"  ρ до МНК  (метод моментов): {rho_before:.4f}")
    print(f"  ρ после МНК:                {rho_after:.4f}")
    print(f"  Параметры МНК: α* = {alpha_opt:.6f}")

    print("\n" + "=" * 55)
    print("ШАГ 8: ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 55)
    print(f"  Распределение: Показательное")
    print(f"  Параметры МНК: α* = {alpha_opt:.6f}")
    print(f"  ρ_набл = {rho_after:.4f}")
    print(f"  ρ_кр   = {rho_cr:.4f}  (α = {ALPHA},  df = {df})")
    print(f"  α₀     = {p_val:.4f}   ← реально достигнутый уровень значимости")
    print()
    if accepted:
        print(f"  Гипотеза H₀: ПРИНЯТА ✓  (α₀ = {p_val:.4f} > α = {ALPHA})")
    else:
        print(f"  Гипотеза H₀: ОТВЕРГНУТА ✗  (α₀ = {p_val:.4f} ≤ α = {ALPHA})")

    return params_mnk, rho_after, p_val


def plot_final(counts, edges, params_mnk, rho_after, p_val,
               filename="lab2_final.png"):
    from step5_plot import _theoretical_pdf

    n       = counts.sum()
    widths  = np.diff(edges)
    omega   = counts / n
    rho_emp = omega / widths
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    rho_th  = _theoretical_pdf(midpoints, params_mnk)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(edges[:-1], rho_emp, width=widths, align='edge',
           color='steelblue', alpha=0.55, edgecolor='white',
           label='Эмпирическое')

    x_line = np.linspace(0.0, edges[-1], 500)
    y_line = _theoretical_pdf(x_line, params_mnk)
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f"Показательное (МНК)  λ*={params_mnk['alpha']:.4f}")

    ax.plot(midpoints, rho_th,  'rs', markersize=7)
    ax.plot(midpoints, rho_emp, 'b-^', markersize=7, label='Эмпирическое (середины)')

    ax.set_xlabel("x")
    ax.set_ylabel("Плотность частоты")
    ax.set_title(f"Сравнение распределений — Показательное\n"
                 f"ρ = {rho_after:.3f},  α₀ = {p_val:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  График сохранён: {filename}")