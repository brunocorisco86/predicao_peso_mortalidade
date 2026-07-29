# src/models/train_predict_weight.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from src.utils.logger import logger

def gompertz_model(t, A, b, k):
    """
    Gompertz non-linear biological growth curve:
    W(t) = A * exp(-b * exp(-k * t))
    """
    return A * np.exp(-b * np.exp(-k * t))

def train_and_evaluate():
    logger.info("Starting Weight Model Training: Gompertz fitted on Full Growth (1-60d) and evaluated at Slaughter Age (>= 42d)...")

    unified_csv = os.path.join('data', 'processed', 'unified_data.csv')
    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df_full = pd.read_csv(unified_csv, low_memory=False)
    df_clean = pd.read_csv(cleaned_csv, low_memory=False)

    # 1. Fit Gompertz on Full Growth Trajectory (Idades 1 a 60 dias)
    df_full['idade'] = pd.to_numeric(df_full['idade'], errors='coerce')
    df_full['peso'] = pd.to_numeric(df_full['peso'], errors='coerce')

    valid_full = df_full[
        (df_full['idade'] >= 1) & (df_full['idade'] <= 60) &
        (df_full['peso'].notnull()) & (df_full['peso'] >= 0.02) & (df_full['peso'] <= 5.0)
    ].copy()
    valid_full['peso_g'] = valid_full['peso'] * 1000.0

    t_all = valid_full['idade'].values
    w_all = valid_full['peso_g'].values

    p0 = [4000.0, 4.5, 0.045]
    bounds = ([1000.0, 1.0, 0.01], [7000.0, 10.0, 0.2])

    popt, pcov = curve_fit(gompertz_model, t_all, w_all, p0=p0, bounds=bounds)
    A_fit, b_fit, k_fit = popt
    logger.info(f"Gompertz Fitted on Full History (1-60d): A = {A_fit:.2f}g, b = {b_fit:.4f}, k = {k_fit:.4f}")

    # Evaluate Gompertz ONLY at Slaughter Age (>= 42d)
    slaughter_gomp = valid_full[valid_full['idade'] >= 42].copy()
    t_slaughter = slaughter_gomp['idade'].values
    w_slaughter = slaughter_gomp['peso_g'].values

    w_gomp_pred = gompertz_model(t_slaughter, A_fit, b_fit, k_fit)
    gomp_rmse = np.sqrt(mean_squared_error(w_slaughter, w_gomp_pred))
    gomp_mae = mean_absolute_error(w_slaughter, w_gomp_pred)
    gomp_r2 = r2_score(w_slaughter, w_gomp_pred)
    logger.info(f"Gompertz Model Evaluated at Slaughter (>=42d) -> R²: {gomp_r2:.4f}, MAE: {gomp_mae:.2f}g, RMSE: {gomp_rmse:.2f}g")

    joblib.dump({'A': A_fit, 'b': b_fit, 'k': k_fit, 'r2': gomp_r2, 'mae': gomp_mae, 'rmse': gomp_rmse}, os.path.join(models_dir, 'gompertz_params.pkl'))

    # 2. Plot Full Gompertz Curve with Highlighted Slaughter Window
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    fig, ax = plt.subplots(figsize=(10, 6))
    t_span = np.linspace(1, 60, 150)
    w_span_gomp = gompertz_model(t_span, A_fit, b_fit, k_fit)

    # Plot all observations in background
    ax.scatter(t_all, w_all, alpha=0.02, s=6, color='#1f77b4', label='Pesagens Históricas (1-60 dias)')
    # Highlight slaughter window
    ax.axvspan(42, 60, color='#ff7f0e', alpha=0.15, label='Janela de Abate (>= 42 dias)')
    # Plot Gompertz curve
    ax.plot(t_span, w_span_gomp, color='#d62728', linewidth=3.0, label=f'Curva de Gompertz Biológica\n$W(t) = {A_fit:.0f} \\cdot e^{{-{b_fit:.2f} e^{{-{k_fit:.3f} t}}}}$')

    ax.set_title("Curva de Crescimento Biológico de Gompertz (Ajuste Global 1-60d)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Idade das Aves (dias)", fontsize=12)
    ax.set_ylabel("Peso Corporal (g)", fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'curva_crescimento_gompertz.png'), dpi=300)
    plt.close()

    logger.info("Gompertz Training & Plotting Completed Successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    train_and_evaluate()
