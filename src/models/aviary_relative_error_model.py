# src/models/aviary_relative_error_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report, accuracy_score, f1_score
from src.utils.logger import logger, setup_logging
from config.settings import settings

def gompertz_func(t, A=6260.16, b=4.7378, k=0.0449):
    return A * np.exp(-b * np.exp(-k * t))

def run_aviary_relative_error_experiment():
    logger.info("Starting Relative Error per Aviary Feature Engineering & Model Training on SLAUGHTER GROUND TRUTH...")

    unified_csv = settings.UNIFIED_CSV_PATH
    cleaned_csv = os.path.join(settings.PROCESSED_DIR, 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    data_proc_dir = settings.PROCESSED_DIR

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df_full = pd.read_csv(unified_csv, low_memory=False)
    df_clean = pd.read_csv(cleaned_csv, low_memory=False)

    df_full['idade'] = pd.to_numeric(df_full['idade'], errors='coerce')
    df_full['peso'] = pd.to_numeric(df_full['peso'], errors='coerce')

    # Historical field weighings for Gompertz & Aviary Error baseline
    valid_hist = df_full[
        (df_full['idade'] >= 1) & (df_full['idade'] <= 60) &
        (df_full['peso'].notnull()) & (df_full['peso'] >= 0.02) & (df_full['peso'] <= 5.0)
    ].copy()
    valid_hist['peso_g'] = valid_hist['peso'] * 1000.0

    df_clean['aviario_id'] = df_clean['lote_composto'].astype(str).str.split('-').str[0]
    valid_hist['aviario_id'] = valid_hist['lote_composto'].astype(str).str.split('-').str[0]

    # 1. Feature Engineering: Relative Error per Aviary (%) & Absolute Delta (g)
    logger.info("Computing Relative Error (%) and Absolute Delta (g) per Aviary...")
    valid_hist['w_gompertz'] = gompertz_func(valid_hist['idade'].values)
    valid_hist['delta_res_g'] = valid_hist['peso_g'] - valid_hist['w_gompertz']
    valid_hist['erro_relativo_aviario_pct'] = ((valid_hist['peso_g'] - valid_hist['w_gompertz']) / valid_hist['w_gompertz']) * 100.0

    aviary_dim = valid_hist.groupby('aviario_id').agg({
        'delta_res_g': ['mean', 'std'],
        'erro_relativo_aviario_pct': ['mean', 'std'],
        'peso_g': 'count'
    }).reset_index()

    aviary_dim.columns = [
        'aviario_id',
        'delta_aviario_g', 'std_delta_aviario_g',
        'erro_relativo_aviario_pct', 'std_erro_relativo_aviario_pct',
        'count_pesagens_aviario'
    ]

    aviary_csv_path = os.path.join(data_proc_dir, 'dimensao_completa_aviario.csv')
    aviary_dim.to_csv(aviary_csv_path, index=False)
    logger.info(f"Full Aviary Dimension saved to {aviary_csv_path} with {len(aviary_dim)} aviaries.")

    # 2. Merge Aviary Features into Cleaned Dataset
    df_clean = df_clean.merge(
        aviary_dim[['aviario_id', 'delta_aviario_g', 'erro_relativo_aviario_pct', 'std_erro_relativo_aviario_pct']],
        on='aviario_id', how='left'
    )
    df_clean['delta_aviario_g'] = df_clean['delta_aviario_g'].fillna(0.0)
    df_clean['erro_relativo_aviario_pct'] = df_clean['erro_relativo_aviario_pct'].fillna(0.0)
    df_clean['std_erro_relativo_aviario_pct'] = df_clean['std_erro_relativo_aviario_pct'].fillna(df_clean['std_erro_relativo_aviario_pct'].median())

    # 3. Intermediate Longitudinal Features & Time Series
    df_clean['w_gompertz'] = gompertz_func(df_clean['idade'].values)

    daily_mean_weights = valid_hist.groupby('idade')['peso_g'].mean().sort_index()
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            daily_series = daily_mean_weights.copy()
            daily_series.index = pd.date_range(start='2026-01-01', periods=len(daily_series), freq='D')
            arima_model = ARIMA(daily_series, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_preds_dict = dict(zip(daily_mean_weights.index, arima_fit.fittedvalues.values))
    except Exception:
        arima_preds_dict = daily_mean_weights.to_dict()

    df_clean['w_arima'] = df_clean['idade'].map(arima_preds_dict).fillna(df_clean['w_gompertz'])

    valid_pre = valid_hist[valid_hist['idade'] <= 41].copy()

    w35_df = valid_pre[valid_pre['idade'].between(33, 37)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w35_df.rename(columns={'peso_g': 'peso_dia_35'}, inplace=True)

    w28_df = valid_pre[valid_pre['idade'].between(26, 30)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w28_df.rename(columns={'peso_g': 'peso_dia_28'}, inplace=True)

    w21_df = valid_pre[valid_pre['idade'].between(19, 23)].groupby('lote_composto')['peso_g'].mean().reset_index()
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

    # Features set (RN-08: Apenas taxas relativas em %, sem contagens brutas de mortes/descartes)
    full_feature_set = [
        'delta_aviario_g', 'erro_relativo_aviario_pct', 'std_erro_relativo_aviario_pct',
        'w_gompertz', 'w_arima',
        'idade', 'cab_alojadas',
        'mortalidade_pct', 'descartados_pct', 'taxa_perda_total',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'c16', 'c17', 'f07', 'f15', 'x02', 'log_x02_distancia',
        'peso_dia_21', 'peso_dia_28', 'peso_dia_35',
        'gpd_semana4', 'gpd_semana5', 'aceleracao_crescimento', 'peso_projetado_gpd'
    ]

    avail_features = []
    for c in full_feature_set:
        if c in df_hybrid.columns and c not in avail_features:
            avail_features.append(c)

    df_ml = df_hybrid[['lote_composto', 'peso_g', 'c15'] + avail_features].dropna().copy().reset_index(drop=True)
    df_ml = df_ml.loc[:, ~df_ml.columns.duplicated()].copy()

    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    logger.info(f"Dataset for modeling: {len(X)} samples, {len(avail_features)} features.")

    # 4. Train Model with Stacking Ensemble (GroupKFold = 5)
    gkf = GroupKFold(n_splits=5)
    tuned_lgb = lgb.LGBMRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
    tuned_xgb = xgb.XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.05, subsample=0.85, random_state=42, n_jobs=-1)
    tuned_hgb = HistGradientBoostingRegressor(max_iter=400, max_depth=7, learning_rate=0.05, random_state=42)

    estimators = [('lgb', tuned_lgb), ('xgb', tuned_xgb), ('hgb', tuned_hgb)]
    relative_aviary_stacking = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)

    oof_preds = np.zeros(len(df_ml))
    rmse_folds, mae_folds, r2_folds, mape_folds = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        relative_aviary_stacking.fit(X_tr, y_tr)
        preds = relative_aviary_stacking.predict(X_te)
        oof_preds[test_idx] = preds

        rmse = np.sqrt(mean_squared_error(y_te, preds))
        mae = mean_absolute_error(y_te, preds)
        r2 = r2_score(y_te, preds)
        mape = np.mean(np.abs((y_te - preds) / y_te)) * 100.0

        rmse_folds.append(rmse)
        mae_folds.append(mae)
        r2_folds.append(r2)
        mape_folds.append(mape)

        logger.info(f"Fold {fold} -> MAE: {mae:.2f}g | RMSE: {rmse:.2f}g | R²: {r2:.4f} | MAPE: {mape:.2f}%")

    final_rmse = float(np.mean(rmse_folds))
    final_mae = float(np.mean(mae_folds))
    final_r2 = float(np.mean(r2_folds))
    final_mape = float(np.mean(mape_folds))

    logger.info(f"\n🏆 CHAMPION MODEL (SLAUGHTER GROUND TRUTH) -> RMSE: {final_rmse:.2f}g | MAE: {final_mae:.2f}g | R²: {final_r2:.4f} | MAPE: {final_mape:.2f}%")

    # Fit final model on full dataset
    relative_aviary_stacking.fit(X, y)
    model_artifact_path = os.path.join(models_dir, 'relative_aviary_slaughter_model.pkl')
    joblib.dump(relative_aviary_stacking, model_artifact_path)
    logger.info(f"Saved champion model artifact to {model_artifact_path}")

    # Save Cross Validation Results
    cv_df = pd.DataFrame({
        'Fold': list(range(1, 6)) + ['Média'],
        'MAE (g)': mae_folds + [final_mae],
        'RMSE (g)': rmse_folds + [final_rmse],
        'R2': r2_folds + [final_r2],
        'MAPE (%)': mape_folds + [final_mape]
    })
    cv_df.to_csv(os.path.join(data_proc_dir, 'cross_validation_results.csv'), index=False)

    # 5. Generate Evaluation & Diagnostic Plots
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Observed vs Predicted Slaughter Weight
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y, oof_preds, alpha=0.25, color='#1f77b4', s=16)
    min_v, max_v = min(y.min(), oof_preds.min()), max(y.max(), oof_preds.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Predição Perfeita (1:1)')
    ax.set_title(f"Modelo Campeão: Predito vs Observado no Abate\nMAE: {final_mae:.2f}g | RMSE: {final_rmse:.2f}g | R²: {final_r2:.4f}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Peso Real no Abate (g)", fontsize=11)
    ax.set_ylabel("Peso Predito no Abate (g)", fontsize=11)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predito_vs_observado_peso.png'), dpi=300)
    plt.savefig(os.path.join(plots_dir, 'modelo_tri_hibrido_predito_vs_real.png'), dpi=300)
    plt.close()

    # Plot 2: Residual Histogram
    residuals = y - oof_preds
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=40, color='#2ca02c', ax=ax)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Resíduo Zero')
    ax.set_title("Análise de Resíduos (Out-of-Fold) - Erro de Predição no Abate", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Resíduo (Peso Real - Peso Predito) em g", fontsize=11)
    ax.set_ylabel("Frequência de Lotes", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'analise_residuos_histograma.png'), dpi=300)
    plt.close()

    # Plot 3: Residual Scatter Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(oof_preds, residuals, alpha=0.25, color='#ff7f0e', s=16)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title("Homocedasticidade dos Resíduos vs Peso Predito no Abate", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Peso Predito no Abate (g)", fontsize=11)
    ax.set_ylabel("Resíduo (g)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'analise_residuos_scatter.png'), dpi=300)
    plt.close()

    # Plot 4: Confusion Matrix for Commercial Weight Categories (Leve < 2600g, Padrão 2600-3100g, Pesado > 3100g)
    def categorize_weight(w):
        if w < 2600:
            return 'Leve (<2.6kg)'
        elif w <= 3100:
            return 'Padrão (2.6-3.1kg)'
        else:
            return 'Pesado (>3.1kg)'

    y_cat = [categorize_weight(v) for v in y]
    pred_cat = [categorize_weight(v) for v in oof_preds]
    labels_order = ['Leve (<2.6kg)', 'Padrão (2.6-3.1kg)', 'Pesado (>3.1kg)']

    cm = confusion_matrix(y_cat, pred_cat, labels=labels_order)
    acc = accuracy_score(y_cat, pred_cat)
    f1 = f1_score(y_cat, pred_cat, average='macro')

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order, ax=ax)
    ax.set_title(f"Matriz de Confusão por Categoria Comercial de Abate\nAcurácia Global: {acc*100:.1f}% | F1-Macro: {f1:.4f}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Categoria Predita", fontsize=11)
    ax.set_ylabel("Categoria Real", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'matriz_confusao_peso.png'), dpi=300)
    plt.close()

    # Plot 5: Feature Importances
    lgb_imp = lgb.LGBMRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
    lgb_imp.fit(X, y)
    importances = lgb_imp.feature_importances_
    indices = np.argsort(importances)[::-1][:12]
    sorted_features = [avail_features[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette='magma')
    ax.set_title("Importância das Variáveis no Modelo Campeão de Peso de Abate", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Importância Relativa (Split Count)", fontsize=12)
    ax.set_ylabel("Variável", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'importancia_features_erro_relativo_aviario.png'), dpi=300)
    plt.savefig(os.path.join(plots_dir, 'importancia_features.png'), dpi=300)
    plt.close()

    logger.info("Champion model evaluation completed successfully!")

if __name__ == "__main__":
    setup_logging()
    run_aviary_relative_error_experiment()
