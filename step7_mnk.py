# step7_mnk.py — Шаг 7: Уточнение параметров методом МНК + α₀
#
# По заданию: уточняем оценки путём минимизации статистики ρ(X).
# Это и есть МНК по смыслу задания (минимум критерия Пирсона).
# α₀ = P(ρ > ρ_набл) при истинности H₀.

import numpy as np
from scipy import stats, optimize
from config import DISTRIBUTION, ALPHA
from step6_pearson import _interval_prob, _N_PARAMS, _DIST_NAMES


def _rho(param_vec, counts, edges):
    """Статистика ρ как функция параметров (для минимизации)."""
    n = counts.sum()
    dist = DISTRIBUTION

    if dist == 'expon':
        if param_vec[0] <= 0:
            return 1e9
        params = {'alpha': param_vec[0]}

    elif dist == 'normal':
        if param_vec[1] <= 0:
            return 1e9
        params = {'a': param_vec[0], 'sigma': param_vec[1]}

    elif dist == 'laplace':
        if param_vec[1] <= 0:
            return 1e9
        params = {'a': param_vec[0], 'sigma': param_vec[1]}

    elif dist == 'rayleigh':
        if param_vec[0] <= 0:
            return 1e9
        params = {'sigma': param_vec[0]}

    elif dist == 'uniform':
        if param_vec[1] <= param_vec[0]:
            return 1e9
        params = {'a': param_vec[0], 'b': param_vec[1]}

    elif dist == 'chi2':
        if param_vec[0] <= 0:
            return 1e9
        params = {'k': param_vec[0]}

    elif dist == 'student':
        if param_vec[0] <= 2:
            return 1e9
        params = {'k': param_vec[0]}

    probs = _interval_prob(edges, params)
    # Защита от нуля
    probs = np.clip(probs, 1e-12, None)
    np_ = n * probs
    return float(np.sum((counts - np_) ** 2 / np_))


def mnk_refine(params_mm, counts, edges, rho_before):
    """Минимизируем ρ, стартуя из оценок метода моментов."""
    dist = DISTRIBUTION

    # Начальная точка
    if dist == 'expon':
        x0 = [params_mm['alpha']]
        bounds = [(1e-6, None)]
    elif dist == 'normal':
        x0 = [params_mm['a'], params_mm['sigma']]
        bounds = [(None, None), (1e-6, None)]
    elif dist == 'laplace':
        x0 = [params_mm['a'], params_mm['sigma']]
        bounds = [(None, None), (1e-6, None)]
    elif dist == 'rayleigh':
        x0 = [params_mm['sigma']]
        bounds = [(1e-6, None)]
    elif dist == 'uniform':
        x0 = [params_mm['a'], params_mm['b']]
        bounds = [(None, None), (None, None)]
    elif dist == 'chi2':
        x0 = [params_mm['k']]
        bounds = [(1e-3, None)]
    elif dist == 'student':
        x0 = [params_mm['k']]
        bounds = [(2.01, None)]

    result = optimize.minimize(
        _rho, x0, args=(counts, edges),
        method='L-BFGS-B', bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 1000}
    )

    rho_after = result.fun
    x_opt = result.x

    # Восстанавливаем словарь параметров МНК
    if dist == 'expon':
        params_mnk = {'alpha': x_opt[0]}
        param_str  = f"α* = {x_opt[0]:.6f}"
    elif dist in ('normal', 'laplace'):
        params_mnk = {'a': x_opt[0], 'sigma': x_opt[1]}
        param_str  = f"a* = {x_opt[0]:.6f},  σ* = {x_opt[1]:.6f}"
    elif dist == 'rayleigh':
        params_mnk = {'sigma': x_opt[0]}
        param_str  = f"σ* = {x_opt[0]:.6f}"
    elif dist == 'uniform':
        params_mnk = {'a': x_opt[0], 'b': x_opt[1]}
        param_str  = f"a* = {x_opt[0]:.6f},  b* = {x_opt[1]:.6f}"
    elif dist == 'chi2':
        params_mnk = {'k': x_opt[0]}
        param_str  = f"k* = {x_opt[0]:.6f}"
    elif dist == 'student':
        params_mnk = {'k': x_opt[0]}
        param_str  = f"k* = {x_opt[0]:.6f}"

    l      = _N_PARAMS[dist]
    m      = len(counts)
    df     = m - l - 1
    rho_cr = stats.chi2.ppf(1 - ALPHA, df)
    p_val  = 1.0 - stats.chi2.cdf(rho_after, df)
    accepted = rho_after < rho_cr

    print("\n" + "=" * 55)
    print("ШАГ 7: УТОЧНЕНИЕ ПАРАМЕТРОВ (МНК)")
    print("=" * 55)
    print(f"  ρ до МНК  (метод моментов): {rho_before:.4f}")
    print(f"  ρ после МНК:                {rho_after:.4f}")
    print(f"  Параметры МНК: {param_str}")
    print()
    print("=" * 55)
    print("ШАГ 8: ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 55)
    print(f"  Распределение: {_DIST_NAMES[dist]}")
    print(f"  Параметры МНК: {param_str}")
    print(f"  ρ_набл = {rho_after:.4f}")
    print(f"  ρ_кр   = {rho_cr:.4f}  (α = {ALPHA},  df = {df})")
    print(f"  α₀     = {p_val:.4f}   ← реально достигнутый уровень значимости")
    print()
    if accepted:
        print(f"  Гипотеза H₀: ПРИНЯТА ✓")
        print(f"  (α₀ = {p_val:.4f} > α = {ALPHA})")
    else:
        print(f"  Гипотеза H₀: ОТВЕРГНУТА ✗")
        print(f"  (α₀ = {p_val:.4f} ≤ α = {ALPHA})")

    return params_mnk, rho_after, p_val


def plot_final(counts, edges, params_mnk, rho_after, p_val,
               filename="lab2_final.png"):
    """График финального результата (как в примере задания — сравнение распределений)."""
    import matplotlib.pyplot as plt
    from step6_pearson import _interval_prob

    dist   = DISTRIBUTION
    n      = counts.sum()
    widths = np.diff(edges)
    omega  = counts / n
    rho_emp = omega / widths
    midpoints = 0.5 * (edges[:-1] + edges[1:])

    # Теоретические плотности в серединах
    from step5_plot import _theoretical_pdf
    rho_th = _theoretical_pdf(midpoints, params_mnk)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(edges[:-1], rho_emp, width=widths, align='edge',
           color='steelblue', alpha=0.55, edgecolor='white',
           label='Эмпирическое')

    # Кривая плотности
    x_lo = 0.0 if dist in ('expon', 'rayleigh') else edges[0]
    x_line = np.linspace(x_lo, edges[-1], 500)
    y_line = _theoretical_pdf(x_line, params_mnk)
    ax.plot(x_line, y_line, 'r-', linewidth=2)

    # Точки сравнения (как на графике в примере задания)
    ax.plot(midpoints, rho_th,  'r-s', markersize=7, label='Теоретическое (МНК)')
    ax.plot(midpoints, rho_emp, 'b-^', markersize=7, label='Эмпирическое')

    ax.set_xlabel("x")
    ax.set_ylabel("Плотность частоты")
    dist_name = _DIST_NAMES[dist]
    ax.set_title(f"Сравнение распределений — {dist_name}\n"
                 f"ρ = {rho_after:.3f},  α₀ = {p_val:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  График сохранён: {filename}")