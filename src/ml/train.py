"""Entrena y evalúa un modelo XGBoost de forecasting de demanda (Weekly_Sales)."""
import os
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.data_loader import weekly_all
from src.ml.features import build_features, FEATURES, TARGET

MODEL_PATH = "models/demand_forecast_xgb.pkl"


def weighted_mae(y_true, y_pred, is_holiday):
    """WMAE: la métrica OFICIAL de la competencia de Kaggle de este dataset — las semanas
    de festivo pesan 5x más, porque equivocarse en esas semanas le cuesta más al negocio."""
    weights = np.where(is_holiday == 1, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def cross_validate(df):
    """Validación cruzada respetando el orden temporal (4 cortes, no solo uno),
    para una estimación de error más confiable antes de entrenar el modelo final."""
    tscv = TimeSeriesSplit(n_splits=4)
    X = df[FEATURES].values
    y = df[TARGET].values
    holiday_flags = df["IsHoliday"].values

    print("🔁 Validación cruzada temporal (4 cortes):")
    fold_wmaes = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        model = XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        wmae = weighted_mae(y[test_idx], preds, holiday_flags[test_idx])
        fold_wmaes.append(wmae)
        print(f"   Fold {fold}: WMAE = ${wmae:,.2f}")

    print(f"   Promedio WMAE (4 folds): ${np.mean(fold_wmaes):,.2f}\n")


def train():
    df = build_features(weekly_all)

    cross_validate(df)

    # Entrenamiento final: 80% más antiguo para entrenar, 20% más reciente para evaluar
    df = df.sort_values("Date")
    cutoff = df["Date"].quantile(0.8)
    train_df = df[df["Date"] <= cutoff]
    test_df = df[df["Date"] > cutoff]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    model = XGBRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        early_stopping_rounds=30, eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    wmae = weighted_mae(y_test.values, preds, test_df["IsHoliday"].values)
    avg_sales = y_test.mean()

    print(f"📊 Evaluación final (datos NO vistos, posteriores a {cutoff.date()}):")
    print(f"   MAE:  ${mae:,.2f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   WMAE (métrica oficial, pondera festivos 5x): ${wmae:,.2f}")
    print(f"   Venta semanal promedio real: ${avg_sales:,.2f}")
    print(f"   Error relativo (MAE/promedio): {mae / avg_sales * 100:.1f}%")
    print(f"   Mejor iteración (early stopping): {model.best_iteration}")

    print("\n🔝 Importancia de variables:")
    for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"   {feat}: {imp:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"\n✅ Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    train()