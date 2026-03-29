# ЗАПУСК 1: python step1_explore.py
# Строит гистограмму и выводит A, E для выдвижения гипотезы.
# Никакой проверки — только смотрим на форму данных.

import matplotlib.pyplot as plt
from step1_load import load_data
from step2_stats import compute_stats
from step3_grouped import build_grouped

data = load_data()
xbar, s2, s, A, E = compute_stats(data)
counts, edges, midpoints, widths, n = build_grouped(data)

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
