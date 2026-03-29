# step1_load.py — Шаг 1: Загрузка данных из CSV

import numpy as np
import pandas as pd
from config import CSV_FILE, CSV_COLUMN, VALUE_MIN, VALUE_MAX


def load_data():
    df = pd.read_csv(CSV_FILE)
    df.replace(-999.25, np.nan, inplace=True)

    if CSV_COLUMN not in df.columns:
        raise ValueError(
            f"Столбец '{CSV_COLUMN}' не найден.\n"
            f"Доступные: {list(df.columns)}"
        )

    total_rows = len(df)
    mask = pd.Series(True, index=df.index)

    if "On Bottom (unitless)" in df.columns:
        mask &= df["On Bottom (unitless)"] == 1

    if "Total Pump Output (gal_per_min)" in df.columns:
        mask &= df["Total Pump Output (gal_per_min)"] > 50

    raw = df.loc[mask, CSV_COLUMN].dropna()

    n_before = len(raw)
    if VALUE_MIN is not None:
        raw = raw[raw >= VALUE_MIN]
    if VALUE_MAX is not None:
        raw = raw[raw <= VALUE_MAX]

    data = raw.values

    print("ДАННЫЕ")
    print(f"Файл: {CSV_FILE}")
    print(f"Столбец: {CSV_COLUMN}")
    print(f"Строк в файле: {total_rows}")
    lo = VALUE_MIN if VALUE_MIN is not None else '-∞'
    hi = VALUE_MAX if VALUE_MAX is not None else '+∞'
    print(f"После фильтра [{lo}, {hi}]: {len(data)}")
    print(f"n (итого): {len(data)}")
    print(f"Минимум: {data.min():.4f}")
    print(f"Максимум: {data.max():.4f}")
    print(f"Среднее: {data.mean():.4f}")

    return data