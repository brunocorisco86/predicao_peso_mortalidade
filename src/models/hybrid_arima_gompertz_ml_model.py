# src/models/hybrid_arima_gompertz_ml_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import logger

def gompertz_func(t, A=6260.16, b=4.7378, k=0.0449):
    return A * np.exp(-b * np.exp(-k * t))

def run_hybrid_tri_model():
    logger.info("Starting Hybrid Tri-Ensemble Modeling (Gompertz + ARIMA + Machine Learning Stacking)...")

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
        (df_full['idade'] >= 1) & (df_full['idade'] <= 60) &
        (df_full['peso'].notnull()) & (df_full['peso'] >= 0.02) & (df_full['peso'] <= 5.0)
    ].copy()
    valid_hist['peso_g'] = valid_hist['peso'] * 1000.0

    # 1. Component 1: Gompertz Biological Prediction
    df_clean['w_gompertz'] = gompertz_func(df_clean['idade'].values)

    # 2. Component 2: Fit ARIMA on Average Daily Weight Growth Trajectory (Days 1 to 60)
    logger.info("Fitting ARIMA(1,1,1) Time Series Model on Average Growth Trajectory...")
    daily_mean_weights = valid_hist.groupby('idade')['peso_g'].mean().sort_index()
    
    try:
        arima_model = ARIMA(daily_mean_weights, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        arima_preds_dict = arima_fit.fittedvalues.to_dict()
        logger.info(f"ARIMA(1,1,1) fitted successfully across {len(daily_mean_weights)} age points.")
    except Exception as e:
        logger.warning(f"ARIMA fit fallback: {e}")
        arima_preds_dict = daily_mean_weights.to_dict()

    df_clean['w_arima'] = df_clean['idade'].map(arima_preds_dict).fillna(df_clean['w_gompertz'])

    # 3. Extract Intermediate Longitudinal Features (Days 21, 28, 35)
    valid_pre_slaughter = valid_hist[valid_hist['idade'] <= 41].copy()

    w35_df = valid_pre_slaughter[valid_pre_slaughter['idade'].between(33, 37)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w35_df.rename(columns={'peso_g': 'peso_dia_35'}, inplace=True)

    w28_df = valid_pre_slaughter[valid_pre_slaughter['idade'].between(26, 30)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w28_df.rename(columns={'peso_g': 'peso_dia_28'}, inplace=True)

    w21_df = valid_pre_slaughter[valid_pre_slaughter['idade'].between(19, 23)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w21_df.rename(columns={'peso_g': 'peso_dia_21'}, inplace=True)

    df_hybrid = df_clean.merge(w35_df, on='lote_composto', how='left')
    df_hybrid = df_hybrid.merge(w28_df, on='lote_composto', how='left')
    df_hybrid = df_hybrid.merge(w21_df, on='lote_composto', how='left')

    df_hybrid['peso_dia_35'] = df_hybrid['peso_dia_35'].fillna(df_hybrid['peso_dia_35'].median())
    df_hybrid['peso_dia_28'] = df_hybrid['peso_dia_28'].fillna(df_hybrid['peso_dia_28'].median())
    df_hybrid['peso_dia_21'] = df_hybrid['peso_dia_21'].fillna(df_hybrid['peso_dia_21'].median())

    df_hybrid['gpd_semana5'] = (df_hybrid['peso_dia_35'] - df_hybrid['peso_dia_28']) / 7.0
    df_hybrid['gpd_semana4'] = (df_hybrid['peso_dia_28'] - df_hybrid['peso_dia_21']) / 7.0
    df_hybrid['aceleracao_crescimento'] = (df_hybrid['gpd_semana5'] / df_hybrid['gpd_semana4'].replace(0, np.nan)).fillna(1.0).clip(0.5, 2.5)
    df_hybrid['peso_projetado_gpd'] = df_hybrid['peso_dia_35'] + (df_hybrid['idade'] - 35) * df_hybrid['gpd_semana5']

    df_hybrid['mortalidade_pct'] = (df_hybrid['mortalidade'] / df_hybrid['cab_alojadas'].replace(0, np.nan)) * 100.0
    df_hybrid['descartados_pct'] = (df_hybrid['descartados'] / df_hybrid['cab_alojadas'].replace(0, np.nan)) * 100.0
    df_hybrid['taxa_perda_total'] = df_hybrid['mortalidade_pct'] + df_hybrid['descartados_pct']
    df_hybrid['log_x02_distancia'] = np.log1p(df_hybrid['x02'].clip(lower=0))

    cat_cols = ['c16', 'c17']
    for c in cat_cols:
        if c in df_hybrid.columns:
            df_hybrid[c] = df_hybrid[c].astype('category').cat.codes

    # Tri-Hybrid Feature Set: Gompertz + ARIMA + Machine Learning Features
    tri_features = [
        'w_gompertz', 'w_arima',
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'mortalidade_pct', 'descartados_pct', 'taxa_perda_total',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'c16', 'c17', 'f07', 'f15', 'x02', 'log_x02_distancia',
        'peso_dia_21', 'peso_dia_28', 'peso_dia_35',
        'gpd_semana4', 'gpd_semana5', 'aceleracao_crescimento', 'peso_projetado_gpd'
    ]
    avail_features = [c for c in tri_features if c in df_hybrid.columns]
    logger.info(f"Tri-Hybrid feature set size: {len(avail_features)} (including Gompertz & ARIMA components).")

    df_ml = df_hybrid[['lote_composto', 'peso_g'] + avail_features].dropna().copy().reset_index(drop=True)
    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # 4. Evaluate Individual Components & Hybrid Stacking Ensemble via 5-Fold GroupKFold
    gkf = GroupKFold(n_splits=5)

    comp_eval = []

    # A. Gompertz Pure Component Metrics
    mae_gomp = mean_absolute_error(y, df_ml['w_gompertz'])
    rmse_gomp = np.sqrt(mean_squared_error(y, df_ml['w_gompertz']))
    r2_gomp = r2_score(y, df_ml['w_gompertz'])
    comp_eval.append({'Modelo': 'Componente 1: Gompertz Puro', 'RMSE (g)': rmse_gomp, 'MAE (g)': mae_gomp, 'R2': r2_gomp})

    # B. ARIMA Pure Component Metrics
    mae_arima = mean_absolute_error(y, df_ml['w_arima'])
    rmse_arima = np.sqrt(mean_squared_error(y, df_ml['w_arima']))
    r2_arima = r2_score(y, df_ml['w_arima'])
    comp_eval.append({'Modelo': 'Componente 2: ARIMA(1,1,1) Puro', 'RMSE (g)': rmse_arima, 'MAE (g)': mae_arima, 'R2': r2_arima})

    # C. Tri-Hybrid Stacking Ensemble (Gompertz + ARIMA + LightGBM + XGBoost + HistGB)
    tuned_lgb = lgb.LGBMRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
    tuned_xgb = xgb.XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.05, subsample=0.85, random_state=42, n_jobs=-1)
    tuned_hgb = HistGradientBoostingRegressor(max_iter=400, max_depth=7, learning_rate=0.05, random_state=42)

    estimators = [('lgb', tuned_lgb), ('xgb', tuned_xgb), ('hgb', tuned_hgb)]
    tri_stacking = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)

    oof_preds = np.zeros(len(df_ml))
    rmse_folds, mae_folds, r2_folds = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        tri_stacking.fit(X_tr, y_tr)
        preds = tri_stacking.predict(X_te)
        oof_preds[test_idx] = preds

        rmse_folds.append(np.sqrt(mean_squared_error(y_te, preds)))
        mae_folds.append(mean_absolute_error(y_te, preds))
        r2_folds.append(r2_score(y_te, preds))
        logger.info(f"Tri-Hybrid Fold {fold} -> MAE: {mae_folds[-1]:.2f}g | RMSE: {rmse_folds[-1]:.2f}g | R²: {r2_folds[-1]:.4f}")

    final_rmse = np.mean(rmse_folds)
    final_mae = np.mean(mae_folds)
    final_r2 = np.mean(r2_folds)

    comp_eval.append({'Modelo': 'Modelo Tri-Híbrido (Gompertz + ARIMA + Stacking ML)', 'RMSE (g)': final_rmse, 'MAE (g)': final_mae, 'R2': final_r2})

    eval_df = pd.DataFrame(comp_eval).sort_values(by='RMSE (g)').reset_index(drop=True)
    eval_df.to_csv(os.path.join(data_proc_dir, 'tri_hybrid_ensemble_results.csv'), index=False)

    logger.info(f"\n🏆 TRI-HYBRID ENSEMBLE RESULTS -> RMSE: {final_rmse:.2f}g | MAE: {final_mae:.2f}g | R²: {final_r2:.4f}")

    # Save tri-hybrid model
    tri_stacking.fit(X, y)
    joblib.dump(tri_stacking, os.path.join(models_dir, 'tri_hybrid_slaughter_model.pkl'))

    # 5. Visualizations for Tri-Hybrid Model
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Tri-Hybrid Predicted vs Real
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y, oof_preds, alpha=0.25, color='#9467bd', s=16)
    min_v, max_v = min(y.min(), oof_preds.min()), max(y.max(), oof_preds.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Predição Perfeita (1:1)')
    ax.set_title(f"Modelo Tri-Híbrido (Gompertz + ARIMA + Stacking ML)\nRMSE = {final_rmse:.1f}g | MAE = {final_mae:.1f}g | R² = {final_r2:.4f}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Peso Real no Abate (g)", fontsize=11)
    ax.set_ylabel("Peso Predito pelo Modelo Tri-Híbrido (g)", fontsize=11)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'modelo_tri_hibrido_predito_vs_real.png'), dpi=300)
    plt.close()

    # Plot 2: Comparison Barplot between Gompertz, ARIMA, and Tri-Hybrid
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=eval_df, x='MAE (g)', y='Modelo', ax=ax, palette='rocket')
    ax.set_title("Comparativo de Erro Absoluto: Gompertz vs ARIMA vs Modelo Tri-Híbrido", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("MAE (g) - Quanto menor, melhor", fontsize=11)
    for i, row in eval_df.iterrows():
        ax.text(row['MAE (g)'] + 1, i, f"{row['MAE (g)']:.1f}g", va='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comparativo_tri_hibrido_gompertz_arima.png'), dpi=300)
    plt.close()

    logger.info("Hybrid Tri-Ensemble Modeling Completed Successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_hybrid_tri_model()
