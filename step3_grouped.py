# step3_grouped.py — Шаг 3: Группированный статистический ряд
#
# По заданию:
#   - число интервалов m по формуле Стёрджесса: m = 1 + [log2(n)]
#   - нижняя граница первого интервала — удобное округлённое значение
#   - верхняя граница последнего — удобное округлённое значение
#   - объединяем соседние интервалы пока ni < 5

import numpy as np


def build_grouped(data):
    n = len(data)

    # Число интервалов по Стёрджессу
    m = 1 + int(np.log2(n))

    # Нижняя граница — 0 (или удобное округлённое число, как в примере задания)
    # Верхняя граница — округлённое «удобное» значение выше максимума
    x_low = 0.0   # нижняя граница = 0 (как в примере задания)

    # Шаг: делим диапазон [0, xmax] на m равных частей, округляем вверх до целого
    step = np.ceil(data.max() / m)
    x_high = x_low + m * step

    # Границы
    edges = [x_low + i * step for i in range(m + 1)]

    # Подсчёт частот
    counts, edges = np.histogram(data, bins=edges)

    # Объединяем интервалы с ni < 5.
    # По примеру задания: объединяем ТОЛЬКО с соседним (предпочтительно — хвостовые
    # объединяются с предыдущим). Продолжаем пока есть ni < 5.
    counts  = list(counts)
    edges   = list(edges)

    changed = True
    while changed:
        changed = False
        for i in range(len(counts) - 1, -1, -1):
            if counts[i] < 5 and len(counts) > 1:
                if i == len(counts) - 1:
                    # Последний — объединяем с предыдущим
                    counts[i - 1] += counts[i]
                    counts.pop(i)
                    edges.pop(i)        # убираем левую границу последнего
                else:
                    # Объединяем со следующим
                    counts[i + 1] += counts[i]
                    counts.pop(i)
                    edges.pop(i + 1)    # убираем правую границу текущего
                changed = True
                break

    counts = np.array(counts)
    edges  = np.array(edges)
    m_new  = len(counts)

    # Ширина каждого интервала (может быть разная после объединений)
    widths = np.diff(edges)

    # Середины интервалов
    midpoints = 0.5 * (edges[:-1] + edges[1:])

    # Относительные частоты и плотности
    omega = counts / n
    rho   = omega / widths

    print("\n" + "=" * 55)
    print("ШАГ 3: ГРУППИРОВАННЫЙ РЯД")
    print("=" * 55)
    print(f"Интервалов по Стёрджессу: {m}")
    print(f"После объединения (ni≥5): {m_new}")
    print(f"Все ni ≥ 5: {'ДА ✓' if counts.min() >= 5 else 'НЕТ ✗'}")
    print()
    print(f"  {'Интервал':<22} {'Середина':>9} {'nᵢ':>6} {'ωᵢ':>8} {'ρᵢ':>12}")
    print(f"  {'-' * 62}")
    for i in range(m_new):
        lo = edges[i]
        hi = edges[i + 1]
        print(f"  [{lo:6.2f}, {hi:6.2f})     "
              f"{midpoints[i]:9.2f} {counts[i]:6d} {omega[i]:8.4f} {rho[i]:12.6f}")

    return counts, edges, midpoints, widths, n


def print_table_for_report(counts, edges, midpoints, widths, n):
    """Таблица в формате как в примере задания."""
    omega = counts / n
    rho   = omega / widths
    print("\n  Таблица для отчёта:")
    header = f"  {'Интервал':<14} {'Середина xᵢ':>12} {'nᵢ':>5} {'ωᵢ':>8} {'ρᵢ':>12}"
    print(header)
    print("  " + "-" * 55)
    for i in range(len(counts)):
        lo = edges[i]; hi = edges[i + 1]
        interval_str = f"{lo:.0f} - {hi:.0f}"
        print(f"  {interval_str:<14} {midpoints[i]:>12.1f} {counts[i]:>5d} "
              f"{omega[i]:>8.3f} {rho[i]:>12.5f}")