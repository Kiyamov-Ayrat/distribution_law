# step6_pearson.py — Шаг 6: Критерий Пирсона χ²
#
# По заданию:
#   ρ(X) = Σ (ni - n·pi*)² / (n·pi*)
#   df = m - l - 1,  где l — число оцениваемых параметров
#   τ(1-α) — квантиль χ²(df)
#   Принять H0 если ρ_набл < τ(1-α)

import numpy as np
from scipy import stats
from config import DISTRIBUTION, ALPHA


# Число параметров распределения (l)
_N_PARAMS = {
    'expon':    1,   # α
    'normal':   2,   # a, σ
    'laplace':  2,   # a, σ
    'rayleigh': 1,   # σ
    'uniform':  2,   # a, b
}

_DIST_NAMES = {
    'expon':    'Показательное',
    'normal':   'Нормальное',
    'laplace':  'Лапласа',
    'rayleigh': 'Рэлея',
    'uniform':  'Равномерное',
}


def _interval_prob(edges, params):
    """Теоретические вероятности попадания в каждый интервал."""
    dist = DISTRIBUTION
    m    = len(edges) - 1
    probs = np.zeros(m)

    def cdf(x):
        if dist == 'expon':
            alpha = params['alpha']
            return 1.0 - np.exp(-alpha * x) if x >= 0 else 0.0
        elif dist == 'normal':
            return stats.norm.cdf(x, loc=params['a'], scale=params['sigma'])
        elif dist == 'laplace':
            return stats.laplace.cdf(x, loc=params['a'],
                                     scale=params['sigma'] / np.sqrt(2))
        elif dist == 'rayleigh':
            return stats.rayleigh.cdf(x, scale=params['sigma'])
        elif dist == 'uniform':
            return stats.uniform.cdf(x, loc=params['a'],
                                     scale=params['b'] - params['a'])
        raise ValueError(f"Неизвестное: {dist}")

    for i in range(m):
        lo = edges[i]
        hi = edges[i + 1]
        # Для последнего интервала правая граница = +∞ (как в примере задания)
        if i == m - 1:
            probs[i] = 1.0 - cdf(lo)
        else:
            probs[i] = cdf(hi) - cdf(lo)

    return probs


def pearson_test(counts, edges, params, silent=False):
    n      = counts.sum()
    m      = len(counts)
    l      = _N_PARAMS[DISTRIBUTION]
    df     = m - l - 1
    probs  = _interval_prob(edges, params)
    np_    = n * probs
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(np_ > 0, (counts - np_)**2 / np_, np.where(counts > 0, 1e15, 0.0))
    rho    = terms.sum()
    rho_cr = stats.chi2.ppf(1 - ALPHA, df)
    p_val  = 1.0 - stats.chi2.cdf(rho, df)
    accepted = rho < rho_cr

    print("\n" + "=" * 55)
    print("ШАГ 6: КРИТЕРИЙ ПИРСОНА")
    print("=" * 55)
    print(f"Гипотеза H₀: {_DIST_NAMES[DISTRIBUTION]} распределение")
    print()
    print(f"  {'Интервал':<22} {'nᵢ':>6} {'n·pᵢ*':>9} {'(nᵢ-npᵢ*)²/npᵢ*':>18}")
    print(f"  {'-' * 60}")
    for i in range(m):
        lo = edges[i]; hi = edges[i + 1]
        t_str = f"{terms[i]:18.5f}" if terms[i] < 1e13 else f"{'∞':>18}"
        print(f"  [{lo:6.2f}, {hi:6.2f})     "
              f"{counts[i]:6d} {np_[i]:9.4f} {t_str}")
    print(f"  {'-' * 60}")
    rho_str = f"{rho:18.5f}" if rho < 1e13 else f"{'∞':>18}"
    print(f"  {'ИТОГО':>28} {n:6d} {np_.sum():9.4f} {rho_str}")
    print()
    rho_disp = f"{rho:.4f}" if rho < 1e13 else "∞"
    print(f"  ρ_набл = {rho_disp}")
    print(f"  df     = m - l - 1 = {m} - {l} - 1 = {df}")
    print(f"  ρ_кр   = {rho_cr:.4f}  (α = {ALPHA})")
    print()
    if accepted:
        print(f"  Гипотеза H₀ ПРИНЯТА ✓  (ρ_набл = {rho:.4f} < ρ_кр = {rho_cr:.4f})")
    else:
        print(f"  Гипотеза H₀ ОТВЕРГНУТА ✗  (ρ_набл = {rho:.4f} ≥ ρ_кр = {rho_cr:.4f})")

    return rho, rho_cr, df, p_val, accepted, probs