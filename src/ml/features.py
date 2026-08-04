"""Ingeniería de features para el modelo de forecasting de demanda."""
import pandas as pd

FEATURES = [
    "Store", "Dept", "Type_encoded", "Size",
    "month", "week_of_year", "year", "IsHoliday",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "sales_lag_1", "sales_lag_52", "rolling_mean_4",
]
TARGET = "Weekly_Sales"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["Store", "Dept", "Date"])

    df["IsHoliday"] = df["IsHoliday"].astype(int)
    df["Type_encoded"] = df["Type"].map({"A": 0, "B": 1, "C": 2})

    # Las promociones (MarkDown) vienen NaN cuando no hubo promoción activa -> tratamos como 0
    for col in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]:
        df[col] = df[col].fillna(0)

    grouped = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    # Lag 1: venta de la semana inmediatamente anterior (misma tienda+depto)
    df["sales_lag_1"] = grouped.shift(1)

    # Rolling mean 4: promedio de las 4 semanas anteriores.
    # shift(1) ANTES del rolling es clave: evita que la fila "vea" su propio valor
    # (data leakage) al calcular su propio promedio de referencia.
    df["rolling_mean_4"] = grouped.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())

    # Lag 52: venta de la misma semana, un año antes — el mejor predictor individual
    # de estacionalidad en retail. Si no hay año anterior disponible (primeros 12 meses
    # del dataset), usamos rolling_mean_4 como respaldo razonable en vez de perder la fila.
    df["sales_lag_52"] = grouped.shift(52)
    df["sales_lag_52"] = df["sales_lag_52"].fillna(df["rolling_mean_4"])

    # La primera semana de cada tienda+depto no tiene lag_1 posible -> se descarta
    df = df.dropna(subset=["sales_lag_1"])

    return df