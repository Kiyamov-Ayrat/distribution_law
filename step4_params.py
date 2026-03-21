# step4_params.py — Шаг 4: Оценка параметров распределения методом моментов
#
# По примеру задания: параметры оцениваются через выборочные моменты.
# Формулы для каждого распределения:
#
#   Показательное:  α* = 1 / X̄
#   Нормальное:     a* = X̄,  σ* = s
#   Лапласа:        a* = X̄,  σ* = s
#   Рэлея:          σ* = X̄ / sqrt(π/2)  =>  σ* = X̄ * sqrt(2/π)
#   Равномерное:    a* = X̄ - s*sqrt(3),  b* = X̄ + s*sqrt(3)

import numpy as np
from config import DISTRIBUTION


def estimate_params(xbar, s):
    """Возвращает словарь параметров и строку описания."""

    dist = DISTRIBUTION

    if dist == 'expon':
        alpha = 1.0 / xbar
        params = {'alpha': alpha}
        desc   = f"α* = 1/X̄ = 1/{xbar:.4f} = {alpha:.4f}"

    elif dist == 'normal':
        params = {'a': xbar, 'sigma': s}
        desc   = f"a* = X̄ = {xbar:.4f},  σ* = s = {s:.4f}"

    elif dist == 'laplace':
        params = {'a': xbar, 'sigma': s}
        desc   = f"a* = X̄ = {xbar:.4f},  σ* = s = {s:.4f}"

    elif dist == 'rayleigh':
        sigma = xbar * np.sqrt(2.0 / np.pi)
        params = {'sigma': sigma}
        desc   = f"σ* = X̄·√(2/π) = {xbar:.4f}·{np.sqrt(2/np.pi):.4f} = {sigma:.4f}"

    elif dist == 'uniform':
        a = xbar - s * np.sqrt(3)
        b = xbar + s * np.sqrt(3)
        params = {'a': a, 'b': b}
        desc   = f"a* = X̄ - s√3 = {a:.4f},  b* = X̄ + s√3 = {b:.4f}"

    else:
        raise ValueError(f"Неизвестное распределение: {dist}")

    print("\n" + "=" * 55)
    print("ШАГ 4: ПАРАМЕТРЫ (МЕТОД МОМЕНТОВ)")
    print("=" * 55)
    print(f"Распределение: {dist}")
    print(f"Оценки: {desc}")

    return params