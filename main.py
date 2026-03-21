# main.py — Лабораторная работа 2
# Запуск: python main.py  (проверяет все распределения автоматически)

import numpy as np
from scipy import stats, optimize

from step1_load    import load_data
from step2_stats   import compute_stats
from step3_grouped import build_grouped, print_table_for_report
from step4_params  import estimate_params
from step5_plot    import plot_histogram
from step6_pearson import pearson_test, _interval_prob, _N_PARAMS
from step7_mnk     import mnk_refine, plot_final
from config        import ALPHA

_NAMES = {
    'expon':    'Показательное',
    'normal':   'Нормальное',
    'laplace':  'Лапласа',
    'rayleigh': 'Рэлея',
    'uniform':  'Равномерное',
    'chi2':     'Хи-квадрат χ²',
    'student':  'Стьюдента t_k',
}


def _params_mm(dist, xbar, s):
    s2 = s ** 2
    if dist == 'expon':    return {'alpha': 1.0 / xbar}
    if dist == 'normal':   return {'a': xbar, 'sigma': s}
    if dist == 'laplace':  return {'a': xbar, 'sigma': s}
    if dist == 'rayleigh': return {'sigma': xbar * np.sqrt(2.0 / np.pi)}
    if dist == 'uniform':  return {'a': xbar - s*np.sqrt(3), 'b': xbar + s*np.sqrt(3)}
    if dist == 'chi2':     return {'k': max(1.0, xbar)}
    if dist == 'student':
        k = 2.0 * s2 / (s2 - 1.0) if s2 > 1.0 else 30.0
        return {'k': max(3.0, k)}


def _rho_val(dist, edges, counts, params):
    import config as cfg; cfg.DISTRIBUTION = dist
    import importlib, step6_pearson; importlib.reload(step6_pearson)
    n  = counts.sum()
    p  = np.clip(step6_pearson._interval_prob(edges, params), 1e-12, None)
    np_ = n * p
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(np_ > 0, (counts - np_)**2 / np_,
                         np.where(counts > 0, 1e15, 0.0))
    return float(terms.sum())


def _mnk_fit(dist, edges, counts, params_mm):
    if dist == 'expon':
        x0, bounds = [params_mm['alpha']], [(1e-9, None)]
        def p2d(v): return {'alpha': v[0]}
    elif dist in ('normal', 'laplace'):
        x0 = [params_mm['a'], params_mm['sigma']]
        bounds = [(None, None), (1e-9, None)]
        def p2d(v): return {'a': v[0], 'sigma': v[1]}
    elif dist == 'rayleigh':
        x0, bounds = [params_mm['sigma']], [(1e-9, None)]
        def p2d(v): return {'sigma': v[0]}
    elif dist == 'uniform':
        x0 = [params_mm['a'], params_mm['b']]
        bounds = [(None, None), (None, None)]
        def p2d(v): return {'a': v[0], 'b': v[1]}

    res = optimize.minimize(
        lambda v: _rho_val(dist, edges, counts, p2d(v)),
        x0, method='L-BFGS-B', bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 2000}
    )
    return p2d(res.x), res.fun


def main():
    # 1–3. Данные, характеристики, группировка
    data = load_data()
    xbar, s2, s, A, E = compute_stats(data)
    counts, edges, midpoints, widths, n = build_grouped(data)
    print_table_for_report(counts, edges, midpoints, widths, n)

    m = len(counts)

    # ── Проверка всех распределений ──────────────────────────────────
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ВСЕХ РАСПРЕДЕЛЕНИЙ")
    print("=" * 70)
    print(f"  {'Распределение':<16} {'ρ (ММ)':>12}  {'ρ (МНК)':>12}  "
          f"{'df':>4}  {'ρ_кр':>8}  {'α₀':>8}  Итог")
    print(f"  {'-'*74}")

    results = {}
    _L = {'expon': 1, 'normal': 2, 'laplace': 2, 'rayleigh': 1, 'uniform': 2}

    for dist in _NAMES:
        try:
            pm     = _params_mm(dist, xbar, s)
            rho_mm = _rho_val(dist, edges, counts, pm)
            pm_mnk, rho_mnk = _mnk_fit(dist, edges, counts, pm)

            l      = _L[dist]
            df     = m - l - 1
            if df < 1:
                print(f"  {_NAMES[dist]:<16}  df<1, пропущено")
                continue

            rho_cr = stats.chi2.ppf(1 - ALPHA, df)
            pval   = 1 - stats.chi2.cdf(rho_mnk, df)
            ok     = rho_mnk < rho_cr
            status = "✓ ПРИНЯТА" if ok else "✗ отверг."

            rho_mm_str  = f"{rho_mm:.2f}"  if rho_mm  < 1e13 else "∞"
            rho_mnk_str = f"{rho_mnk:.2f}" if rho_mnk < 1e13 else "∞"

            results[dist] = dict(rho_mm=rho_mm, rho_mnk=rho_mnk,
                                 df=df, rho_cr=rho_cr, pval=pval,
                                 ok=ok, pm=pm, pm_mnk=pm_mnk)

            print(f"  {_NAMES[dist]:<16} {rho_mm_str:>12}  {rho_mnk_str:>12}  "
                  f"{df:>4}  {rho_cr:>8.4f}  {pval:>8.4f}  {status}")
        except Exception as ex:
            print(f"  {_NAMES[dist]:<16}  ошибка: {ex}")

    # ── Лучшее распределение ─────────────────────────────────────────
    valid   = {k: v for k, v in results.items() if v is not None}
    best    = min(valid, key=lambda k: valid[k]['rho_mnk'])
    r       = valid[best]
    accepted_any = any(v['ok'] for v in valid.values())

    print(f"\n  → Лучшее: {_NAMES[best]}  (ρ_МНК = {r['rho_mnk']:.4f})")
    if accepted_any:
        acc_list = [_NAMES[d] for d, v in valid.items() if v['ok']]
        print(f"  → ПРИНЯТЫЕ гипотезы: {', '.join(acc_list)}")
    else:
        print(f"  → Все гипотезы отвергнуты.")
        print(f"     В отчёт идёт лучшее: {_NAMES[best]}")

    # ── Детальный вывод для лучшего ──────────────────────────────────
    import config as cfg; cfg.DISTRIBUTION = best
    import importlib
    import step4_params; importlib.reload(step4_params)
    import step5_plot;   importlib.reload(step5_plot)
    import step6_pearson; importlib.reload(step6_pearson)
    import step7_mnk;    importlib.reload(step7_mnk)

    pm_mnk = r['pm_mnk']
    p      = np.clip(step6_pearson._interval_prob(edges, pm_mnk), 1e-12, None)
    np_    = n * p
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(np_ > 0, (counts - np_)**2 / np_,
                         np.where(counts > 0, 1e15, 0.0))

    print("\n" + "=" * 70)
    print(f"ДЕТАЛЬНЫЙ РАСЧЁТ: {_NAMES[best]} (МНК)")
    print("=" * 70)
    print(f"\n  {'Интервал':<22} {'nᵢ':>6} {'n·pᵢ*':>10} {'(nᵢ-npᵢ*)²/npᵢ*':>18}")
    print(f"  {'-'*62}")
    for i in range(m):
        lo = edges[i]; hi = edges[i+1]
        t_str = f"{terms[i]:.5f}" if terms[i] < 1e13 else "∞"
        print(f"  [{lo:7.2f}, {hi:7.2f})  "
              f"{counts[i]:6d} {np_[i]:10.4f} {t_str:>18}")
    print(f"  {'-'*62}")
    rho_final = terms.sum()
    rho_str = f"{rho_final:.5f}" if rho_final < 1e13 else "∞"
    print(f"  {'ИТОГО':>29} {n:6d} {np_.sum():10.4f} {rho_str:>18}")

    print(f"\n  ρ_набл = {r['rho_mnk']:.4f}")
    print(f"  df     = m - l - 1 = {m} - {_L[best]} - 1 = {r['df']}")
    print(f"  ρ_кр   = {r['rho_cr']:.4f}  (α = {ALPHA})")
    print(f"  α₀     = {r['pval']:.4f}")

    # ── Вывод для отчёта ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ВЫВОД ДЛЯ ОТЧЁТА")
    print("=" * 70)
    print(f"  n  = {n}")
    print(f"  X̄  = {xbar:.4f}")
    print(f"  s² = {s2:.4f}")
    print(f"  A  = {A:.4f}")
    print(f"  E  = {E:.4f}")
    print()
    print(f"  Гипотеза H₀: {_NAMES[best]} распределение")
    print(f"  ρ_набл (МНК) = {r['rho_mnk']:.4f}")
    print(f"  ρ_кр         = {r['rho_cr']:.4f}  (α={ALPHA}, df={r['df']})")
    print(f"  α₀           = {r['pval']:.4f}")
    fin = r['rho_mnk'] < r['rho_cr']
    print(f"  Гипотеза {'ПРИНЯТА ✓' if fin else 'ОТВЕРГНУТА ✗'}")

    # ── Графики ───────────────────────────────────────────────────────
    step5_plot.plot_histogram(counts, edges, midpoints, widths, r['pm'])
    step7_mnk.plot_final(counts, edges, pm_mnk, r['rho_mnk'], r['pval'])


if __name__ == "__main__":
    main()