# step4_params.py — Шаг 4: Оценка параметров показательного распределения
#
# Метод моментов: α* = 1 / X̄


def estimate_params(xbar, s):
    alpha  = 1.0 / xbar
    params = {'alpha': alpha}

    print("ПАРАМЕТРЫ (МЕТОД МОМЕНТОВ)")
    print(f"Распределение: Показательное")
    print(f"Оценка: a* = 1/X- = 1/{xbar:.4f} = {alpha:.4f}")

    return params