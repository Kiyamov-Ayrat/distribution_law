# ЗАПУСК 2: python step2_test_hypothesis.py
# Проверяет гипотезу H₀: показательное распределение.

from step1_load import load_data
from step2_stats import compute_stats
from step3_grouped import build_grouped, print_table_for_report
from step4_params import estimate_params
import step5_plot
from step6_pearson import pearson_test
from step7_mnk import mnk_refine, plot_final
from config import ALPHA

data = load_data()
xbar, s2, s, A, E = compute_stats(data)
counts, edges, midpoints, widths, n = build_grouped(data)
print_table_for_report(counts, edges, midpoints, widths, n)

print(f"\n  Гипотеза H₀: Показательное распределение")

# Шаг 4: параметры методом моментов
params_mm = estimate_params(xbar, s)

# Шаг 5: гистограмма + теоретическая кривая
step5_plot.plot_histogram(counts, edges, midpoints, widths, params_mm)

# Шаг 6: критерий Пирсона
rho_mm, rho_cr, df, p_val_mm, accepted_mm, probs = pearson_test(
    counts, edges, params_mm
)

# Шаги 7-8: МНК + итог + финальный график
params_mnk, rho_mnk, p_val = mnk_refine(params_mm, counts, edges, rho_mm)
plot_final(counts, edges, params_mnk, rho_mnk, p_val)

# Итоговая сводка для отчёта
print("\n" + "=" * 55)
print("  СВОДКА ДЛЯ ОТЧЁТА")
print("=" * 55)
print(f"  n            = {n}")
print(f"  X̄            = {xbar:.4f}")
print(f"  s²           = {s2:.4f}")
print(f"  A            = {A:.4f}")
print(f"  E            = {E:.4f}")
print(f"  α* (МНК)     = {params_mnk['alpha']:.6f}")
print(f"  ρ_набл (МНК) = {rho_mnk:.4f}")
print(f"  ρ_кр         = {rho_cr:.4f}  (α = {ALPHA}, df = {df})")
print(f"  α₀           = {p_val:.4f}")
fin = rho_mnk < rho_cr
print(f"  Гипотеза H₀: {'ПРИНЯТА ✓' if fin else 'ОТВЕРГНУТА ✗'}")