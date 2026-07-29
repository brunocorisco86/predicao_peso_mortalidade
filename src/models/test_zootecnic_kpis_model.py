# src/models/test_zootecnic_kpis_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import logger

def gompertz_func(t, A=6260.16, b=4.7378, k=0.0449):
    return A * np.exp(-b * np.exp(-k * t))

def run_zootecnic_kpis_experiment():
    logger.info("Starting Zootecnic KPIs Feature Engineering & Model Testing...")

    unified_csv = os.path.join('data', 'processed', 'unified_data.csv')
    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    data_proc_dir = os.path.join('data', 'processed')

    df_full = pd.read_csv(unified_csv, low_memory=False)
    df_clean = pd.read_csv(cleaned_csv, low_memory=False)

    df_full['idade'] = pd.to_numeric(df_full['idade'], errors='coerce')
    df_full['peso'] = pd.to_numeric(df_full['peso'], errors='coerce')

    valid_hist = df_full[
        (df_full['idade'] >= 1) & (df_full['idade'] <= 60) &
        (df_full['peso'].notnull()) & (df_full['peso'] >= 0.02) & (df_full['peso'] <= 5.0)
    ].copy()
    valid_hist['peso_g'] = valid_hist['peso'] * 1000.0

    df_clean['aviario_id'] = df_clean['lote_composto'].astype(str).str.split('-').str[0]
    valid_hist['aviario_id'] = valid_hist['lote_composto'].astype(str).str.split('-').str[0]

    # 1. Aviary Dimension
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

    df_clean = df_clean.merge(
        aviary_dim[['aviario_id', 'delta_aviario_g', 'erro_relativo_aviario_pct', 'std_erro_relativo_aviario_pct']],
        on='aviario_id', how='left'
    )
    df_clean['delta_aviario_g'] = df_clean['delta_aviario_g'].fillna(0.0)
    df_clean['erro_relativo_aviario_pct'] = df_clean['erro_relativo_aviario_pct'].fillna(0.0)
    df_clean['std_erro_relativo_aviario_pct'] = df_clean['std_erro_relativo_aviario_pct'].fillna(df_clean['std_erro_relativo_aviario_pct'].median())

    # 2. Intermediate Longitudinal Features & Time Series
    df_clean['w_gompertz'] = gompertz_func(df_clean['idade'].values)

    daily_mean_weights = valid_hist.groupby('idade')['peso_g'].mean().sort_index()
    try:
        arima_model = ARIMA(daily_mean_weights, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        arima_preds_dict = arima_fit.fittedvalues.to_dict()
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

    # --- 3. NEW ZOOTECNIC KPIS FEATURE ENGINEERING --- #
    logger.info("Computing 5 New Zootecnic KPIs...")

    # KPI 1: IEP Proxy (Índice de Eficiência Produtiva Europeu Proxy)
    viabilidade_pct = (100.0 - df_hybrid['taxa_perda_total']).clip(lower=50.0, upper=100.0)
    df_hybrid['kpi_iep_proxy'] = (viabilidade_pct * (df_hybrid['peso_dia_35'] / 1000.0)) / (35.0 * 1.60) * 100.0

    # KPI 2: Razão GPD 5ª Semana vs GPD Acumulado da Vida (Aceleração Retida)
    gpd_acumulado_35d = (df_hybrid['peso_dia_35'] - 42.0) / 35.0
    df_hybrid['kpi_razao_gpd_sem5_vs_vida'] = df_hybrid['gpd_semana5'] / gpd_acumulado_35d.replace(0, np.nan)
    df_hybrid['kpi_razao_gpd_sem5_vs_vida'] = df_hybrid['kpi_razao_gpd_sem5_vs_vida'].fillna(1.0).clip(0.5, 2.5)

    # KPI 3: Taxa de Refugagem / Descarte Sanulógico vs Perda Total (%)
    df_hybrid['kpi_taxa_refugagem_relativa'] = (df_hybrid['descartados'] / (df_hybrid['mortalidade'] + df_hybrid['descartados'] + 1.0)) * 100.0

    # KPI 4: Desvio do Peso Inicial do Pintainho (c15) vs Padrão Biológico (42g)
    df_hybrid['kpi_delta_c15_pct'] = ((df_hybrid['c15'] - 42.0) / 42.0) * 100.0

    # KPI 5: Densidade Relativa de Cabeças Alojadas (Escore de Carga)
    median_density = df_hybrid['cab_alojadas'].median()
    df_hybrid['kpi_densidade_relativa'] = df_hybrid['cab_alojadas'] / median_density

    zootecnic_kpi_features = [
        'kpi_iep_proxy',
        'kpi_razao_gpd_sem5_vs_vida',
        'kpi_taxa_refugagem_relativa',
        'kpi_delta_c15_pct',
        'kpi_densidade_relativa'
    ]

    full_feature_set = [
        'delta_aviario_g', 'erro_relativo_aviario_pct', 'std_erro_relativo_aviario_pct',
        'w_gompertz', 'w_arima',
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'mortalidade_pct', 'descartados_pct', 'taxa_perda_total',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'c16', 'c17', 'f07', 'f15', 'x02', 'log_x02_distancia',
        'peso_dia_21', 'peso_dia_28', 'peso_dia_35',
        'gpd_semana4', 'gpd_semana5', 'aceleracao_crescimento', 'peso_projetado_gpd'
    ] + zootecnic_kpi_features

    avail_features = []
    for c in full_feature_set:
        if c in df_hybrid.columns and c not in avail_features:
            avail_features.append(c)

    logger.info(f"Full feature set size with Zootecnic KPIs: {len(avail_features)}")

    df_ml = df_hybrid[['lote_composto', 'peso_g'] + avail_features].dropna().copy().reset_index(drop=True)
    df_ml = df_ml.loc[:, ~df_ml.columns.duplicated()].copy()

    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # 4. Train & Evaluate Stacking Ensemble with Zootecnic KPIs
    gkf = GroupKFold(n_splits=5)
    tuned_lgb = lgb.LGBMRegressor(n_estimators=450, max_depth=7, learning_rate=0.04, num_leaves=63, random_state=42, verbose=-1)
    tuned_xgb = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, random_state=42, n_jobs=-1)
    tuned_hgb = HistGradientBoostingRegressor(max_iter=450, max_depth=7, learning_rate=0.04, random_state=42)

    estimators = [('lgb', tuned_lgb), ('xgb', tuned_xgb), ('hgb', tuned_hgb)]
    kpi_stacking = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)

    oof_preds = np.zeros(len(df_ml))
    rmse_folds, mae_folds, r2_folds = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        kpi_stacking.fit(X_tr, y_tr)
        preds = kpi_stacking.predict(X_te)
        oof_preds[test_idx] = preds

        rmse_folds.append(np.sqrt(mean_squared_error(y_te, preds)))
        mae_folds.append(mean_absolute_error(y_te, preds))
        r2_folds.append(r2_score(y_te, preds))
        logger.info(f"Zootecnic KPI Model Fold {fold} -> MAE: {mae_folds[-1]:.2f}g | RMSE: {rmse_folds[-1]:.2f}g | R²: {r2_folds[-1]:.4f}")

    final_rmse = np.mean(rmse_folds)
    final_mae = np.mean(mae_folds)
    final_r2 = np.mean(r2_folds)

    logger.info(f"\n🏆 MODELO COM KPIS ZOOTÉCNICOS -> RMSE: {final_rmse:.2f}g | MAE: {final_mae:.2f}g | R²: {final_r2:.4f}")

    kpi_stacking.fit(X, y)
    joblib.dump(kpi_stacking, os.path.join(models_dir, 'zootecnic_kpi_slaughter_model.pkl'))

    # 5. Plot Feature Importances with Zootecnic KPIs
    lgb_imp = lgb.LGBMRegressor(n_estimators=450, max_depth=7, learning_rate=0.04, num_leaves=63, random_state=42, verbose=-1)
    lgb_imp.fit(X, y)
    importances = lgb_imp.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    sorted_features = [avail_features[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette='plasma')
    ax.set_title("Importância das Variáveis Incluindo os Novos KPIs Zootécnicos", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Importância Relativa (Split Count)", fontsize=12)
    ax.set_ylabel("Variável / KPI Zootécnico", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'importancia_features_kpis_zootecnicos.png'), dpi=300)
    plt.close()

    # Export KPI summary table
    kpi_summary = pd.DataFrame([{
        'Modelo': 'Stacking + KPIs Zootécnicos',
        'MAE (g)': final_mae,
        'RMSE (g)': final_rmse,
        'R2': final_r2
    }])
    kpi_summary.to_csv(os.path.join(data_proc_dir, 'zootecnic_kpis_model_results.csv'), index=False)

    logger.info("Zootecnic KPIs experiment completed successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_zootecnic_kpis_experiment()
