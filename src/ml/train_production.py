"""
Entrena el modelo FINAL de producción usando TODOS los datos disponibles
(a diferencia de train.py, que reserva 20% para evaluación).

Reutiliza el número óptimo de árboles ya encontrado con early stopping en el
modelo de evaluación, para no sobreajustar al entrenar sin un holdout que lo controle.
"""
import os
import joblib
from xgboost import XGBRegressor

from src.data_loader import weekly_all
from src.ml.features import build_features, FEATURES, TARGET

EVAL_MODEL_PATH = "models/demand_forecast_xgb.pkl"
PRODUCTION_MODEL_PATH = "models/demand_forecast_xgb_production.pkl"


def train_production():
    eval_bundle = joblib.load(EVAL_MODEL_PATH)
    best_n_estimators = eval_bundle["model"].best_iteration

    df = build_features(weekly_all)
    X, y = df[FEATURES], df[TARGET]

    model = XGBRegressor(
        n_estimators=best_n_estimators, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, PRODUCTION_MODEL_PATH)
    print(f"✅ Modelo de producción entrenado con TODOS los datos ({len(df)} filas)")
    print(f"   Árboles usados: {best_n_estimators} (número óptimo encontrado previamente por early stopping)")
    print(f"   Guardado en: {PRODUCTION_MODEL_PATH}")


if __name__ == "__main__":
    train_production()