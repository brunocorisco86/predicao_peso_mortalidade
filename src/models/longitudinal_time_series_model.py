# src/models/longitudinal_time_series_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import logger

def gompertz_func(t, A, b, k):
    return A * np.exp(-b * np.exp(-k * t))

def run_longitudinal_modeling():
    logger.info("Starting Longitudinal Time Series & Dynamic Growth Modeling (Item 2)...")

    unified_csv = os.path.join('data', 'processed', 'unified_data.csv')
    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    data_proc_dir = os.path.join('data', 'processed')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df_full = pd.read_csv(unified_csv, low_memory=False)
    df_clean = pd.read_csv(cleaned_csv, low_memory=False)

    df_full['idade'] = pd.to_numeric(df_full['idade'], errors='coerce')
    df_full['peso'] = pd.to_numeric(df_full['peso'], errors='coerce')

    valid_hist = df_full[
        (df_full['idade'] >= 1) & (df_full['idade'] <= 41) &
        (df_full['peso'].notnull()) & (df_full['peso'] >= 0.02) & (df_full['peso'] <= 4.5)
    ].copy()
    valid_hist['peso_g'] = valid_hist['peso'] * 1000.0

    # 1. Feature Extraction: Last known weights prior to slaughter window (Day 28 and Day 35)
    logger.info("Extracting batch-level intermediate weights (Day 21, 28, 35) and Growth Slopes...")

    # Group by batch and extract weights at age 28 and 35
    w35_df = valid_hist[valid_hist['idade'].between(33, 37)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w35_df.rename(columns={'peso_g': 'peso_dia_35'}, inplace=True)

    w28_df = valid_hist[valid_hist['idade'].between(26, 30)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w28_df.rename(columns={'peso_g': 'peso_dia_28'}, inplace=True)

    w21_df = valid_hist[valid_hist['idade'].between(19, 23)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w21_df.rename(columns={'peso_g': 'peso_dia_21'}, inplace=True)

    # Merge features back to df_clean
    df_long = df_clean.merge(w35_df, on='lote_composto', how='left')
    df_long = df_long.merge(w28_df, on='lote_composto', how='left')
    df_long = df_long.merge(w21_df, on='lote_composto', how='left')

    # Impute missing intermediate weights with age-group medians if needed
    if 'peso_dia_35' in df_long.columns:
        df_long['peso_dia_35'] = df_long['peso_dia_35'].fillna(df_long['peso_dia_35'].median())
    if 'peso_dia_28' in df_long.columns:
        df_long['peso_dia_28'] = df_long['peso_dia_28'].fillna(df_long['peso_dia_28'].median())
    if 'peso_dia_21' in df_long.columns:
        df_long['peso_dia_21'] = df_long['peso_dia_21'].fillna(df_long['peso_dia_21'].median())

    # 2. Compute Growth Slopes & Acceleration
    # GPD (Ganho de Peso Diário) na 5ª Semana (dias 28 a 35)
    df_long['gpd_semana5'] = (df_long['peso_dia_35'] - df_long['peso_dia_28']) / 7.0
    # GPD na 4ª Semana (dias 21 a 28)
    df_long['gpd_semana4'] = (df_long['peso_dia_28'] - df_long['peso_dia_21']) / 7.0
    # Aceleração de crescimento
    df_long['aceleracao_crescimento'] = df_long['gpd_semana5'] / df_long['gpd_semana4'].replace(0, np.nan)
    df_long['aceleracao_crescimento'] = df_long['aceleracao_crescimento'].fillna(1.0).clip(0.5, 2.5)

    # Projected weight trajectory based on day 35 + (idade - 35) * gpd_semana5
    df_long['peso_projetado_gpd'] = df_long['peso_dia_35'] + (df_long['idade'] - 35) * df_long['gpd_semana5']

    # 3. Base & Engineered Features Selection
    df_long['mortalidade_pct'] = (df_long['mortalidade'] / df_long['cab_alojadas'].replace(0, np.nan)) * 100.0
    df_long['descartados_pct'] = (df_long['descartados'] / df_long['cab_alojadas'].replace(0, np.nan)) * 100.0
    df_long['taxa_perda_total'] = df_long['mortalidade_pct'] + df_long['descartados_pct']
    df_long['log_x02_distancia'] = np.log1p(df_long['x02'].clip(lower=0))

    cat_cols = ['c16', 'c17']
    for c in cat_cols:
        if c in df_long.columns:
            df_long[c] = df_long[c].astype('category').cat.codes

    longitudinal_features = [
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'mortalidade_pct', 'descartados_pct', 'taxa_perda_total',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'c16', 'c17', 'f07', 'f15', 'x02', 'log_x02_distancia',
        # Longitudinal features
        'peso_dia_21', 'peso_dia_28', 'peso_dia_35',
        'gpd_semana4', 'gpd_semana5', 'aceleracao_crescimento', 'peso_projetado_gpd'
    ]
    avail_features = [c for c in longitudinal_features if c in df_long.columns]
    logger.info(f"Total features for Longitudinal Model: {len(avail_features)} (including {7} time-series features)")

    df_ml = df_long[['lote_composto', 'peso_g'] + avail_features].dropna().copy().reset_index(drop=True)
    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # 4. Evaluate Longitudinal Models via 5-Fold GroupKFold
    gkf = GroupKFold(n_splits=5)

    models = {
        'HistGradientBoosting_Longitudinal': HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.1, min_samples_leaf=10, random_state=42),
        'LightGBM_Longitudinal': lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, min_child_samples=15, random_state=42, verbose=-1),
        'XGBoost_Longitudinal': xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    }

    eval_results = []
    oof_predictions = {name: np.zeros(len(df_ml)) for name in models.keys()}
    oof_predictions['Stacking_Longitudinal'] = np.zeros(len(df_ml))

    for name, model in models.items():
        rmse_l, mae_l, r2_l = [], [], []

        for train_idx, test_idx in gkf.split(X, y, groups):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            oof_predictions[name][test_idx] = preds

            rmse_l.append(np.sqrt(mean_squared_error(y_te, preds)))
            mae_l.append(mean_absolute_error(y_te, preds))
            r2_l.append(r2_score(y_te, preds))

        m_rmse, m_mae, m_r2 = np.mean(rmse_l), np.mean(mae_l), np.mean(r2_l)
        logger.info(f"Model {name:34s} -> RMSE: {m_rmse:.2f}g | MAE: {m_mae:.2f}g | R²: {m_r2:.4f}")
        eval_results.append({'Modelo': name, 'RMSE (g)': m_rmse, 'MAE (g)': m_mae, 'R2': m_r2})

    # Stacking Ensemble with Longitudinal Features
    logger.info("Training Stacking Ensemble with Longitudinal Features...")
    estimators = [
        ('hgb', HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.1, min_samples_leaf=10, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1)),
        ('xgb', xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1))
    ]

    stack_rmse_l, stack_mae_l, stack_r2_l = [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        stack = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)
        stack.fit(X_tr, y_tr)
        preds = stack.predict(X_te)
        oof_predictions['Stacking_Longitudinal'][test_idx] = preds

        stack_rmse_l.append(np.sqrt(mean_squared_error(y_te, preds)))
        stack_mae_l.append(mean_absolute_error(y_te, preds))
        stack_r2_l.append(r2_score(y_te, preds))

    s_rmse, s_mae, s_r2 = np.mean(stack_rmse_l), np.mean(stack_mae_l), np.mean(stack_r2_l)
    logger.info(f"Model Stacking_Longitudinal                -> RMSE: {s_rmse:.2f}g | MAE: {s_mae:.2f}g | R²: {s_r2:.4f}")
    eval_results.append({'Modelo': 'Stacking_Longitudinal', 'RMSE (g)': s_rmse, 'MAE (g)': s_mae, 'R2': s_r2})

    eval_df = pd.DataFrame(eval_results).sort_values(by='RMSE (g)').reset_index(drop=True)
    eval_df.to_csv(os.path.join(data_proc_dir, 'longitudinal_model_results.csv'), index=False)

    best_row = eval_df.iloc[0]
    logger.info(f"\n🏆 LONGITUDINAL MODEL BEST RESULT: {best_row['Modelo']} -> RMSE: {best_row['RMSE (g)']:.2f}g | MAE: {best_row['MAE (g)']:.2f}g | R²: {best_row['R2']:.4f}")

    # 5. Fit & Save Final Best Longitudinal Model
    best_long_model = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)
    best_long_model.fit(X, y)
    joblib.dump(best_long_model, os.path.join(models_dir, 'longitudinal_slaughter_model.pkl'))

    # 6. Visualizations
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Observed vs Predicted Longitudinal Stacking Model
    fig, ax = plt.subplots(figsize=(8, 8))
    y_oof = oof_predictions['Stacking_Longitudinal']
    ax.scatter(y, y_oof, alpha=0.25, color='#1f77b4', s=16)
    min_v, max_v = min(y.min(), y_oof.min()), max(y.max(), y_oof.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Predição Perfeita (1:1)')
    ax.set_title(f"Modelo Longitudinal com Séries Temporais de Pesagem\nRMSE = {s_rmse:.1f}g | MAE = {s_mae:.1f}g | R² = {s_r2:.4f}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Peso Real no Abate (g)", fontsize=11)
    ax.set_ylabel("Peso Predito pelo Modelo Longitudinal (g)", fontsize=11)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'modelo_longitudinal_predito_vs_real.png'), dpi=300)
    plt.close()

    # Plot 2: Feature Importance of Longitudinal Model (using LightGBM)
    lgb_long = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1)
    lgb_long.fit(X, y)
    importances = lgb_long.feature_importances_
    indices = np.argsort(importances)[::-1][:12]
    sorted_features = [avail_features[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette='viridis')
    ax.set_title("Importância das Variáveis no Modelo Longitudinal de Séries Temporais", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Importância Relativa (Split Count)", fontsize=12)
    ax.set_ylabel("Variável", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'importancia_features_longitudinal.png'), dpi=300)
    plt.close()

    logger.info("Longitudinal Time Series Modeling (Item 2) Completed Successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_longitudinal_modeling()
