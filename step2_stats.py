# step2_stats.py — Шаг 2: Числовые характеристики
# Формулы строго по заданию:
#   X̄  = (1/n) Σ Xi
#   s²  = 1/(n-1) · Σ(Xi - X̄)²
#   A   = 1/((n-1)·s³) · Σ(Xi - X̄)³
#   E   = 1/((n-1)·s⁴) · Σ(Xi - X̄)⁴  - 3

import numpy as np


def compute_stats(data):
    n   = len(data)
    xbar = data.mean()
    s2   = np.sum((data - xbar) ** 2) / (n - 1)
    s    = np.sqrt(s2)
    A    = np.sum((data - xbar) ** 3) / ((n - 1) * s ** 3)
    E    = np.sum((data - xbar) ** 4) / ((n - 1) * s ** 4) - 3

    print("\n" + "=" * 55)
    print("ШАГ 2: ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ")
    print("=" * 55)
    print(f"n  = {n}")
    print(f"X̄  = {xbar:.4f}")
    print(f"s² = {s2:.4f}")
    print(f"s  = {s:.4f}")
    print(f"A  = {A:.4f}   (асимметрия)")
    print(f"E  = {E:.4f}   (эксцесс)")

    print("\n  Теоретические A и E для возможных распределений:")
    print(f"  {'Распределение':<16} {'A':>6}  {'E':>6}")
    print(f"  {'-'*32}")
    print(f"  {'Нормальное':<16} {'0':>6}  {'0':>6}")
    print(f"  {'Лапласа':<16} {'0':>6}  {'3':>6}")
    print(f"  {'Показательное':<16} {'2':>6}  {'6':>6}")
    print(f"  {'Рэлея':<16} {'0.63':>6}  {'0.25':>6}")
    print(f"  {'Равномерное':<16} {'0':>6}  {'-1.2':>6}")

    return xbar, s2, s, A, E