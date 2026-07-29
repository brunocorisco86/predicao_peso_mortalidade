# src/models/train_predict_weight.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from src.utils.logger import logger

def gompertz_model(t, A, b, k):
    """
    Gompertz non-linear growth curve:
    W(t) = A * exp(-b * exp(-k * t))
    t: age in days
    A: asymptotic weight (g)
    b: initial weight scaling constant
    k: maturation rate constant
    """
    return A * np.exp(-b * np.exp(-k * t))

def train_and_evaluate():
    logger.info("Starting Weight Prediction Model Training (Gompertz + Machine Learning)...")

    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(cleaned_csv, low_memory=False)
    logger.info(f"Loaded cleaned_data.csv with {len(df)} rows.")

    # 1. Fit Gompertz Non-Linear Model
    # Filter valid age and weight in grams
    df_gomp = df[['idade', 'peso_g']].dropna().copy()
    t_data = df_gomp['idade'].values
    w_data = df_gomp['peso_g'].values

    # Initial parameter guess: A=4000g, b=5.0, k=0.06
    p0 = [4000.0, 5.0, 0.06]
    bounds = ([1000.0, 1.0, 0.01], [7000.0, 15.0, 0.3])

    popt, pcov = curve_fit(gompertz_model, t_data, w_data, p0=p0, bounds=bounds)
    A_fit, b_fit, k_fit = popt
    logger.info(f"Gompertz Fitted Parameters: A (Assíntota) = {A_fit:.2f}g, b = {b_fit:.4f}, k (Taxa Maturação) = {k_fit:.4f}")

    w_gomp_pred = gompertz_model(t_data, A_fit, b_fit, k_fit)
    gomp_rmse = np.sqrt(mean_squared_error(w_data, w_gomp_pred))
    gomp_mae = mean_absolute_error(w_data, w_gomp_pred)
    gomp_r2 = r2_score(w_data, w_gomp_pred)
    logger.info(f"Gompertz Model Metrics -> R²: {gomp_r2:.4f}, MAE: {gomp_mae:.2f}g, RMSE: {gomp_rmse:.2f}g")

    # Save Gompertz parameters
    gomp_params = {'A': A_fit, 'b': b_fit, 'k': k_fit, 'r2': gomp_r2, 'mae': gomp_mae, 'rmse': gomp_rmse}
    joblib.dump(gomp_params, os.path.join(models_dir, 'gompertz_params.pkl'))

    # 2. Machine Learning Model (GradientBoosting / RandomForest)
    feature_candidates = [
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'f07', 'f15', 'x02'
    ]
    avail_features = [c for c in feature_candidates if c in df.columns]
    logger.info(f"Available features for ML model: {avail_features}")

    df_ml = df[['peso_g'] + avail_features].dropna().copy()
    X = df_ml[avail_features]
    y = df_ml['peso_g']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)
    logger.info(f"RandomForest Model Metrics -> R²: {rf_r2:.4f}, MAE: {rf_mae:.2f}g, RMSE: {rf_rmse:.2f}g")

    # Save ML Model
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_weight_model.pkl'))

    # 3. Generate Visualizations
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Gompertz Fitted Curve vs Data
    fig, ax = plt.subplots(figsize=(10, 6))
    t_span = np.linspace(1, 55, 100)
    w_span_gomp = gompertz_model(t_span, A_fit, b_fit, k_fit)

    ax.scatter(t_data, w_data, alpha=0.03, s=8, color='#1f77b4', label='Pesagens Observadas (g)')
    ax.plot(t_span, w_span_gomp, color='#e377c2', linewidth=3.0, label=f'Curva de Gompertz (R²={gomp_r2:.3f})\n$W(t) = {A_fit:.0f} \\cdot e^{{-{b_fit:.2f} e^{{-{k_fit:.3f} t}}}}$')
    ax.set_title("Ajuste da Curva de Crescimento Não-Linear de Gompertz", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Idade (dias)", fontsize=12)
    ax.set_ylabel("Peso Corporal (g)", fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'curva_crescimento_gompertz.png'), dpi=300)
    plt.close()

    # Plot 2: Predicted vs Observed (ML Model)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred_rf, alpha=0.15, color='#2ca02c', s=12)
    min_val = min(y_test.min(), y_pred_rf.min())
    max_val = max(y_test.max(), y_pred_rf.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predição Perfeita (1:1)')
    ax.set_title(f"Peso Predito vs Observado (Random Forest)\nR² = {rf_r2:.4f} | MAE = {rf_mae:.1f}g", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Peso Observado (g)", fontsize=12)
    ax.set_ylabel("Peso Predito (g)", fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predito_vs_observado_peso.png'), dpi=300)
    plt.close()

    # Plot 3: Feature Importances
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [avail_features[i] for i in indices]
    sorted_importances = importances[indices]

    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette='Blues_r')
    ax.set_title("Importância das Variáveis no Modelo Preditivo de Peso (ML)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Importância Relativa (Gini Importance)", fontsize=12)
    ax.set_ylabel("Variável", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'importancia_features.png'), dpi=300)
    plt.close()

    # Save summary metrics table
    metrics_df = pd.DataFrame([
        {'Modelo': 'Gompertz (Não-Linear)', 'R2': gomp_r2, 'MAE (g)': gomp_mae, 'RMSE (g)': gomp_rmse},
        {'Modelo': 'Random Forest (ML)', 'R2': rf_r2, 'MAE (g)': rf_mae, 'RMSE (g)': rf_rmse}
    ])
    metrics_path = os.path.join('data', 'processed', 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Model metrics saved to {metrics_path}")

    logger.info("Weight Prediction Training & Evaluation Completed Successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    train_and_evaluate()
