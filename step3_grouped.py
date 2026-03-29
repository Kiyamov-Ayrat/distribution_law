# step3_grouped.py — Шаг 3: Группированный статистический ряд

import numpy as np


def build_grouped(data):
    n = len(data)

    m = 1 + int(np.log2(n))

    x_low = 0.0
    step = np.ceil(data.max() / m)

    edges = [x_low + i * step for i in range(m + 1)]
    counts, edges = np.histogram(data, bins=edges)

    # Объединяем интервалы с ni < 5
    counts = list(counts)
    edges = list(edges)

    changed = True
    while changed:
        changed = False
        for i in range(len(counts) - 1, -1, -1):
            if counts[i] < 5 and len(counts) > 1:
                if i == len(counts) - 1:
                    counts[i - 1] += counts[i]
                    counts.pop(i)
                    edges.pop(i)
                else:
                    counts[i + 1] += counts[i]
                    counts.pop(i)
                    edges.pop(i + 1)
                changed = True
                break

    counts = np.array(counts)
    edges = np.array(edges)
    m_new = len(counts)
    widths = np.diff(edges)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    omega = counts / n
    rho = omega / widths

    print("ГРУППИРОВАННЫЙ РЯД")
    print(f"Интервалов по Стёрджессу: {m}")
    print(f"После объединения (ni≥5): {m_new}")
    print(f"  {'Интервал':<22} {'Середина':>9} {'nᵢ':>6} {'ωᵢ':>8} {'ρᵢ':>12}")
    for i in range(m_new):
        print(f"  [{edges[i]:6.2f}, {edges[i+1]:6.2f})     "
              f"{midpoints[i]:9.2f} {counts[i]:6d} {omega[i]:8.4f} {rho[i]:12.6f}")

    return counts, edges, midpoints, widths, n


def print_table_for_report(counts, edges, midpoints, widths, n):
    omega = counts / n
    rho   = omega / widths
    print(f"  {'Интервал':<14} {'Середина xᵢ':>12} {'nᵢ':>5} {'ωᵢ':>8} {'ρᵢ':>12}")
    for i in range(len(counts)):
        lo = edges[i]; hi = edges[i + 1]
        print(f"  {lo:.0f} - {hi:.0f}:<14 {midpoints[i]:>12.1f} {counts[i]:>5d} "
              f"{omega[i]:>8.3f} {rho[i]:>12.5f}")