# ЗАПУСК 1: python step1_explore.py


import matplotlib.pyplot as plt
from step1_load import load_data
from step2_stats import compute_stats
from step3_grouped import build_grouped, print_table_for_report

data  = load_data()
xbar, s2, s, A, E = compute_stats(data)
counts, edges, midpoints, widths, n = build_grouped(data)
print_table_for_report(counts, edges, midpoints, widths, n)

# Гистограмма без теории
omega = counts / n
rho_emp = omega / widths

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(edges[:-1], rho_emp, width=widths, align='edge',
       color='steelblue', alpha=0.7, edgecolor='white')
ax.set_xlabel("x")
ax.set_ylabel("Плотность частоты ρ")
ax.set_title(f"Гистограмма   A={A:.3f}   E={E:.3f}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lab2_explore.png", dpi=150)
plt.close()

print(f"\n  {'Распределение':<18} {'A':>6}  {'E':>6}")
print(f"  {'-'*32}")
print(f"  {'Нормальное':<18} {'0':>6}  {'0':>6}")
print(f"  {'Лапласа':<18} {'0':>6}  {'3':>6}")
print(f"  {'Показательное':<18} {'2':>6}  {'6':>6}")
print(f"  {'Рэлея':<18} {'0.63':>6}  {'0.25':>6}")
print(f"  {'Равномерное':<18} {'0':>6}  {'-1.2':>6}")
print(f"  {'Хи-квадрат chi2':<18} {'>0':>6}  {'>0':>6}")
print(f"  {'Стьюдента student':<18} {'0':>6}  {'>0':>6}")
print(f"\n  Твои значения:  A = {A:.4f},  E = {E:.4f}")
print(f"\n  Открой config.py, поставь DISTRIBUTION = 'нужное'")
print(f"  и запусти: python step2_test.py")