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
    Gompertz non-linear growth curve for slaughter age:
    W(t) = A * exp(-b * exp(-k * t))
    """
    return A * np.exp(-b * np.exp(-k * t))

def train_and_evaluate():
    logger.info("Starting Weight Prediction Model Training for SLAUGHTER AGE (idade >= 42 dias)...")

    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(cleaned_csv, low_memory=False)
    logger.info(f"Loaded cleaned_data.csv with {len(df)} slaughter age records.")

    # 1. Fit Gompertz Non-Linear Model for Slaughter Range
    df_gomp = df[['idade', 'peso_g']].dropna().copy()
    t_data = df_gomp['idade'].values
    w_data = df_gomp['peso_g'].values

    # Initial parameter guess
    p0 = [3800.0, 4.0, 0.05]
    bounds = ([2000.0, 1.0, 0.01], [6000.0, 10.0, 0.2])

    popt, pcov = curve_fit(gompertz_model, t_data, w_data, p0=p0, bounds=bounds)
    A_fit, b_fit, k_fit = popt
    logger.info(f"Gompertz Fitted Parameters (Abate): A = {A_fit:.2f}g, b = {b_fit:.4f}, k = {k_fit:.4f}")

    w_gomp_pred = gompertz_model(t_data, A_fit, b_fit, k_fit)
    gomp_rmse = np.sqrt(mean_squared_error(w_data, w_gomp_pred))
    gomp_mae = mean_absolute_error(w_data, w_gomp_pred)
    gomp_r2 = r2_score(w_data, w_gomp_pred)
    logger.info(f"Gompertz Slaughter Model Metrics -> R²: {gomp_r2:.4f}, MAE: {gomp_mae:.2f}g, RMSE: {gomp_rmse:.2f}g")

    joblib.dump({'A': A_fit, 'b': b_fit, 'k': k_fit, 'r2': gomp_r2, 'mae': gomp_mae, 'rmse': gomp_rmse}, os.path.join(models_dir, 'gompertz_params.pkl'))

    # 2. Machine Learning Model (RandomForest / GradientBoosting for Slaughter Weight)
    feature_candidates = [
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'f07', 'f15', 'x02'
    ]
    avail_features = [c for c in feature_candidates if c in df.columns]

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
    logger.info(f"RandomForest Slaughter Model Metrics -> R²: {rf_r2:.4f}, MAE: {rf_mae:.2f}g, RMSE: {rf_rmse:.2f}g")

    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_weight_model.pkl'))

    # 3. Visualizations for Slaughter Model
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Gompertz Curve vs Slaughter Data
    fig, ax = plt.subplots(figsize=(10, 6))
    t_span = np.linspace(42, 55, 100)
    w_span_gomp = gompertz_model(t_span, A_fit, b_fit, k_fit)

    ax.scatter(t_data, w_data, alpha=0.08, s=12, color='#d62728', label='Pesagens no Abate (g)')
    ax.plot(t_span, w_span_gomp, color='#1f77b4', linewidth=3.0, label=f'Curva de Gompertz de Abate (R²={gomp_r2:.3f})')
    ax.set_title("Curva Preditiva do Peso corporal no Abate (42 a 55 dias)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Idade das Aves no Abate (dias)", fontsize=12)
    ax.set_ylabel("Peso Corporal de Abate (g)", fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'curva_crescimento_gompertz.png'), dpi=300)
    plt.close()

    # Plot 2: Predicted vs Observed Slaughter Weight
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred_rf, alpha=0.2, color='#2ca02c', s=15)
    min_val = min(y_test.min(), y_pred_rf.min())
    max_val = max(y_test.max(), y_pred_rf.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predição Perfeita (1:1)')
    ax.set_title(f"Peso Predito vs Observado no Abate (Random Forest)\nR² = {rf_r2:.4f} | MAE = {rf_mae:.1f}g", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Peso Real no Abate (g)", fontsize=12)
    ax.set_ylabel("Peso Predito no Abate (g)", fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predito_vs_observado_peso.png'), dpi=300)
    plt.close()

    # Plot 3: Feature Importances for Slaughter Weight
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [avail_features[i] for i in indices]
    sorted_importances = importances[indices]

    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette='Oranges_r')
    ax.set_title("Importância das Variáveis no Modelo Preditivo do Peso de Abate", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Importância Relativa (Gini Importance)", fontsize=12)
    ax.set_ylabel("Variável", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'importancia_features.png'), dpi=300)
    plt.close()

    # Save metrics table
    metrics_df = pd.DataFrame([
        {'Modelo': 'Gompertz (Abate)', 'R2': gomp_r2, 'MAE (g)': gomp_mae, 'RMSE (g)': gomp_rmse},
        {'Modelo': 'Random Forest (Abate)', 'R2': rf_r2, 'MAE (g)': rf_mae, 'RMSE (g)': rf_rmse}
    ])
    metrics_path = os.path.join('data', 'processed', 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Slaughter model metrics saved to {metrics_path}")

    logger.info("Slaughter Weight Model Training Completed Successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    train_and_evaluate()
