# step2_stats.py — Шаг 2: Числовые характеристики

import numpy as np


def compute_stats(data):
    n = len(data)
    xbar = data.mean()
    s2 = np.sum((data - xbar) ** 2) / (n - 1)
    s = np.sqrt(s2)
    A = np.sum((data - xbar) ** 3) / ((n - 1) * s ** 3)
    E = np.sum((data - xbar) ** 4) / ((n - 1) * s ** 4) - 3

    print("ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ")
    print(f"n = {n}")
    print(f"X- = {xbar:.4f}")
    print(f"s2 = {s2:.4f}")
    print(f"s = {s:.4f}")
    print(f"A = {A:.4f} (асимметрия)")
    print(f"E = {E:.4f} (эксцесс)")

    return xbar, s2, s, A, E