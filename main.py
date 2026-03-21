# main.py — Лабораторная работа 2
# Запуск: python main.py

from step1_load   import load_data
from step2_stats  import compute_stats
from step3_grouped import build_grouped, print_table_for_report
from step4_params import estimate_params
from step5_plot   import plot_histogram
from step6_pearson import pearson_test
from step7_mnk    import mnk_refine, plot_final
from config       import ALPHA, DISTRIBUTION


def main():
    # 1. Данные
    data = load_data()

    # 2. Числовые характеристики
    xbar, s2, s, A, E = compute_stats(data)

    # 3. Группированный ряд
    counts, edges, midpoints, widths, n = build_grouped(data)
    print_table_for_report(counts, edges, midpoints, widths, n)

    # 4. Параметры методом моментов
    params_mm = estimate_params(xbar, s)

    # 5. Гистограмма с кривой плотности
    plot_histogram(counts, edges, midpoints, widths, params_mm)

    # 6. Критерий Пирсона (с параметрами метода моментов)
    rho_mm, rho_cr, df, p_val_mm, accepted_mm, probs = pearson_test(
        counts, edges, params_mm
    )

    # 7. Уточнение параметров МНК + итог + α₀
    params_mnk, rho_mnk, p_val = mnk_refine(
        params_mm, counts, edges, rho_mm
    )

    # Финальный график
    plot_final(counts, edges, params_mnk, rho_mnk, p_val)

    # Вывод для отчёта
    _NAMES = {
        'expon': 'Показательное', 'normal': 'Нормальное',
        'laplace': 'Лапласа', 'rayleigh': 'Рэлея', 'uniform': 'Равномерное',
    }
    print("\n" + "=" * 55)
    print("ВЫВОД ДЛЯ ОТЧЁТА")
    print("=" * 55)
    print(f"  n  = {n}")
    print(f"  X̄  = {xbar:.4f}")
    print(f"  s² = {s2:.4f}")
    print(f"  A  = {A:.4f}")
    print(f"  E  = {E:.4f}")
    print()
    print(f"  Гипотеза H₀: {_NAMES[DISTRIBUTION]} распределение")
    print(f"  ρ_набл (МНК) = {rho_mnk:.4f}")
    print(f"  ρ_кр         = {rho_cr:.4f}  (α = {ALPHA}, df = {df})")
    print(f"  α₀           = {p_val:.4f}")
    accepted = rho_mnk < rho_cr
    print(f"  Гипотеза {'ПРИНЯТА ✓' if accepted else 'ОТВЕРГНУТА ✗'}")


if __name__ == "__main__":
    main()