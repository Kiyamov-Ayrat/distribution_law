# step4_params.py — Шаг 4: Оценка параметров распределения методом моментов
#
# Формулы оценок по методу моментов:
#   Показательное:   α* = 1 / X̄
#   Нормальное:      a* = X̄,  σ* = s
#   Лапласа:         a* = X̄,  σ* = s
#   Рэлея:           σ* = X̄ · √(2/π)
#   Равномерное:     a* = X̄ - s√3,  b* = X̄ + s√3
#   Хи-квадрат χ²:  k* = X̄  (E[X]=k для χ²(k))
#   Стьюдента t_k:  k* = 2·s²/(s²-1)  (D[X]=k/(k-2) для t_k при k>2)

import numpy as np
from config import DISTRIBUTION


def estimate_params(xbar, s):
    dist = DISTRIBUTION
    s2 = s ** 2

    if dist == 'expon':
        alpha  = 1.0 / xbar
        params = {'alpha': alpha}
        desc   = f"α* = 1/X̄ = 1/{xbar:.4f} = {alpha:.4f}"

    elif dist == 'normal':
        params = {'a': xbar, 'sigma': s}
        desc   = f"a* = X̄ = {xbar:.4f},  σ* = s = {s:.4f}"

    elif dist == 'laplace':
        params = {'a': xbar, 'sigma': s}
        desc   = f"a* = X̄ = {xbar:.4f},  σ* = s = {s:.4f}"

    elif dist == 'rayleigh':
        sigma  = xbar * np.sqrt(2.0 / np.pi)
        params = {'sigma': sigma}
        desc   = f"σ* = X̄·√(2/π) = {xbar:.4f}·{np.sqrt(2/np.pi):.4f} = {sigma:.4f}"

    elif dist == 'uniform':
        a = xbar - s * np.sqrt(3)
        b = xbar + s * np.sqrt(3)
        params = {'a': a, 'b': b}
        desc   = f"a* = X̄ - s√3 = {a:.4f},  b* = X̄ + s√3 = {b:.4f}"

    elif dist == 'chi2':
        # E[X] = k  =>  k* = X̄
        k = max(1.0, xbar)
        params = {'k': k}
        desc   = f"k* = X̄ = {k:.4f}"

    elif dist == 'student':
        # D[X] = k/(k-2)  =>  k* = 2s²/(s²-1), требует s²>1
        if s2 > 1.0:
            k = 2.0 * s2 / (s2 - 1.0)
        else:
            k = 30.0   # при малой дисперсии — большое k (≈нормальное)
        k = max(3.0, k)   # k>2 обязательно для конечной дисперсии
        params = {'k': k}
        desc   = f"k* = 2s²/(s²-1) = {k:.4f}"

    else:
        raise ValueError(f"Неизвестное распределение: '{dist}'.\n"
                         f"Допустимые: expon, normal, laplace, rayleigh, "
                         f"uniform, chi2, student")

    print("\n" + "=" * 55)
    print("ШАГ 4: ПАРАМЕТРЫ (МЕТОД МОМЕНТОВ)")
    print("=" * 55)
    print(f"Распределение: {dist}")
    print(f"Оценки: {desc}")

    return params