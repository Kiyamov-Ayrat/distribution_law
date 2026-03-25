# step4_params.py — Шаг 4: Оценка параметров показательного распределения
#
# Метод моментов: α* = 1 / X̄


def estimate_params(xbar, s):
    alpha  = 1.0 / xbar
    params = {'alpha': alpha}

    print("\n" + "=" * 55)
    print("ШАГ 4: ПАРАМЕТРЫ (МЕТОД МОМЕНТОВ)")
    print("=" * 55)
    print(f"Распределение: Показательное")
    print(f"Оценка: α* = 1/X̄ = 1/{xbar:.4f} = {alpha:.4f}")

    return params