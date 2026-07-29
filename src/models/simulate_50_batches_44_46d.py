# src/models/simulate_50_batches_44_46d.py
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

def run_50_batches_simulation():
    logger.info("Starting Simulation of 50 Random Batches aged 44 to 46 days...")

    unified_csv = os.path.join('data', 'processed', 'unified_data.csv')
    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
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

    # 2. Time Series & Intermediate Features
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

    # 3. Zootecnic KPIs
    viabilidade_pct = (100.0 - df_hybrid['taxa_perda_total']).clip(lower=50.0, upper=100.0)
    df_hybrid['kpi_iep_proxy'] = (viabilidade_pct * (df_hybrid['peso_dia_35'] / 1000.0)) / (35.0 * 1.60) * 100.0

    gpd_acumulado_35d = (df_hybrid['peso_dia_35'] - 42.0) / 35.0
    df_hybrid['kpi_razao_gpd_sem5_vs_vida'] = df_hybrid['gpd_semana5'] / gpd_acumulado_35d.replace(0, np.nan)
    df_hybrid['kpi_razao_gpd_sem5_vs_vida'] = df_hybrid['kpi_razao_gpd_sem5_vs_vida'].fillna(1.0).clip(0.5, 2.5)

    df_hybrid['kpi_taxa_refugagem_relativa'] = (df_hybrid['descartados'] / (df_hybrid['mortalidade'] + df_hybrid['descartados'] + 1.0)) * 100.0
    df_hybrid['kpi_delta_c15_pct'] = ((df_hybrid['c15'] - 42.0) / 42.0) * 100.0
    df_hybrid['kpi_densidade_relativa'] = df_hybrid['cab_alojadas'] / df_hybrid['cab_alojadas'].median()

    features = [
        'delta_aviario_g', 'erro_relativo_aviario_pct', 'std_erro_relativo_aviario_pct',
        'w_gompertz', 'w_arima',
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'mortalidade_pct', 'descartados_pct', 'taxa_perda_total',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'c16', 'c17', 'f07', 'f15', 'x02', 'log_x02_distancia',
        'peso_dia_21', 'peso_dia_28', 'peso_dia_35',
        'gpd_semana4', 'gpd_semana5', 'aceleracao_crescimento', 'peso_projetado_gpd',
        'kpi_iep_proxy', 'kpi_razao_gpd_sem5_vs_vida', 'kpi_taxa_refugagem_relativa',
        'kpi_delta_c15_pct', 'kpi_densidade_relativa'
    ]

    avail_features = []
    for c in features:
        if c in df_hybrid.columns and c not in avail_features:
            avail_features.append(c)

    df_ml = df_hybrid[['lote_composto', 'peso_g'] + avail_features].dropna().copy().reset_index(drop=True)
    df_ml = df_ml.loc[:, ~df_ml.columns.duplicated()].copy()

    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # 4. Model Training & Out-Of-Fold Predictions
    gkf = GroupKFold(n_splits=5)
    tuned_lgb = lgb.LGBMRegressor(n_estimators=450, max_depth=7, learning_rate=0.04, num_leaves=63, random_state=42, verbose=-1)
    tuned_xgb = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.85, random_state=42, n_jobs=-1)
    tuned_hgb = HistGradientBoostingRegressor(max_iter=450, max_depth=7, learning_rate=0.04, random_state=42)

    estimators = [('lgb', tuned_lgb), ('xgb', tuned_xgb), ('hgb', tuned_hgb)]
    kpi_stacking = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)

    oof_preds = np.zeros(len(df_ml))
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        kpi_stacking.fit(X_tr, y_tr)
        oof_preds[test_idx] = kpi_stacking.predict(X_te)

    df_ml['peso_predito_g'] = oof_preds
    df_ml['erro_absoluto_g'] = np.abs(df_ml['peso_g'] - df_ml['peso_predito_g'])
    df_ml['erro_relativo_pct'] = (df_ml['erro_absoluto_g'] / df_ml['peso_g']) * 100.0

    # 5. Filter for Ages 44 to 46 Days and Sample 50 Random Batches
    df_44_46 = df_ml[df_ml['idade'].between(44, 46)].copy()
    logger.info(f"Available records for ages 44-46: {len(df_44_46)}")

    # Sample 50 random batches
    np.random.seed(42)
    sample_size = min(50, len(df_44_46))
    sim_50 = df_44_46.sample(n=sample_size, random_state=42).copy().reset_index(drop=True)

    sim_50['Simulação'] = [f"#{i+1:02d}" for i in range(len(sim_50))]
    sim_50['lote_id_short'] = sim_50['lote_composto'].astype(str).str[:14]

    # Calculate metrics for the 50 sampled batches
    sample_mae = mean_absolute_error(sim_50['peso_g'], sim_50['peso_predito_g'])
    sample_rmse = np.sqrt(mean_squared_error(sim_50['peso_g'], sim_50['peso_predito_g']))
    sample_r2 = r2_score(sim_50['peso_g'], sim_50['peso_predito_g'])
    mean_err_pct = sim_50['erro_relativo_pct'].mean()

    logger.info(f"50 Batches Sample Metrics -> MAE: {sample_mae:.2f}g | RMSE: {sample_rmse:.2f}g | R²: {sample_r2:.4f} | Erro Média %: {mean_err_pct:.2f}%")

    # Export CSV
    export_cols = [
        'Simulação', 'lote_id_short', 'idade', 'peso_dia_35',
        'peso_g', 'peso_predito_g', 'erro_absoluto_g', 'erro_relativo_pct'
    ]
    sim_50_csv = os.path.join(data_proc_dir, 'simulacoes_50_lotes_44_46d.csv')
    sim_50[export_cols].rename(columns={
        'lote_id_short': 'Lote_ID',
        'idade': 'Idade (dias)',
        'peso_dia_35': 'Peso 35d (g)',
        'peso_g': 'Peso Real (g)',
        'peso_predito_g': 'Peso Predito (g)',
        'erro_absoluto_g': 'Erro (g)',
        'erro_relativo_pct': 'Erro (%)'
    }).to_csv(sim_50_csv, index=False)
    logger.info(f"50 Batches Simulation saved to {sim_50_csv}")

    # 6. Generate High-Resolution Scatter Plot
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot colored by Age (44d, 45d, 46d)
    scatter = sns.scatterplot(
        data=sim_50,
        x='peso_g',
        y='peso_predito_g',
        hue='idade',
        style='idade',
        palette='Set1',
        s=120,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.9,
        ax=ax
    )

    # 1:1 Identity Line (Ideal Prediction Line)
    min_val = min(sim_50['peso_g'].min(), sim_50['peso_predito_g'].min()) - 100
    max_val = max(sim_50['peso_g'].max(), sim_50['peso_predito_g'].max()) + 100
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.0, label='Linha de Perfeição (Y = X)')

    # ±100g Error Band Lines
    ax.plot([min_val, max_val], [min_val + 100, max_val + 100], 'k:', alpha=0.5, label='Faixa de Tolerância (±100g)')
    ax.plot([min_val, max_val], [min_val - 100, max_val - 100], 'k:', alpha=0.5)

    ax.set_title("Simulação de 50 Lotes Aleatórios no Abate (44 a 46 Dias)\nPeso Predito vs Peso Real Observado", fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel("Peso Real Medido no Abatedouro (g)", fontsize=12)
    ax.set_ylabel("Peso Predito pelo Modelo (g)", fontsize=12)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Summary Annotation Box
    stats_text = (
        f"📊 Métricas das 50 Simulações (44-46d):\n"
        f"• MAE Amostral: {sample_mae:.1f} g\n"
        f"• RMSE Amostral: {sample_rmse:.1f} g\n"
        f"• R² Amostral: {sample_r2:.4f}\n"
        f"• Erro Relativo Médio: {mean_err_pct:.2f}%"
    )
    ax.text(
        0.04, 0.94, stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.9)
    )

    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, 'simulacao_scatter_50lotes_44_46d.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info(f"Scatter plot saved to {plot_path}")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_50_batches_simulation()
