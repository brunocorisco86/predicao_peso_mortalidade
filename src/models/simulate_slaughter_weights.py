# src/models/simulate_slaughter_weights.py
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
from src.utils.logger import logger

def gompertz_func(t, A=6260.16, b=4.7378, k=0.0449):
    return A * np.exp(-b * np.exp(-k * t))

def run_slaughter_simulations():
    logger.info("Starting 10 Slaughter Weight Simulations (Predito vs Real)...")

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

    # 1. Prepare Features & Model
    df_clean['w_gompertz'] = gompertz_func(df_clean['idade'].values)

    daily_mean_weights = valid_hist.groupby('idade')['peso_g'].mean().sort_index()
    try:
        arima_model = ARIMA(daily_mean_weights, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        arima_preds_dict = arima_fit.fittedvalues.to_dict()
    except Exception:
        arima_preds_dict = daily_mean_weights.to_dict()

    df_clean['w_arima'] = df_clean['idade'].map(arima_preds_dict).fillna(df_clean['w_gompertz'])

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

    tri_features = [
        'w_gompertz', 'w_arima',
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'mortalidade_pct', 'descartados_pct', 'taxa_perda_total',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'c16', 'c17', 'f07', 'f15', 'x02', 'log_x02_distancia',
        'peso_dia_21', 'peso_dia_28', 'peso_dia_35',
        'gpd_semana4', 'gpd_semana5', 'aceleracao_crescimento', 'peso_projetado_gpd'
    ]

    # Ensure unique features list
    avail_features = []
    for c in tri_features:
        if c in df_hybrid.columns and c not in avail_features:
            avail_features.append(c)

    df_ml = df_hybrid[['lote_composto', 'peso_g'] + avail_features].dropna().copy().reset_index(drop=True)
    # Remove duplicate columns if any
    df_ml = df_ml.loc[:, ~df_ml.columns.duplicated()].copy()

    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # 2. Out-of-fold predictions using Stacking Ensemble
    gkf = GroupKFold(n_splits=5)
    tuned_lgb = lgb.LGBMRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
    tuned_xgb = xgb.XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.05, subsample=0.85, random_state=42, n_jobs=-1)
    tuned_hgb = HistGradientBoostingRegressor(max_iter=400, max_depth=7, learning_rate=0.05, random_state=42)

    estimators = [('lgb', tuned_lgb), ('xgb', tuned_xgb), ('hgb', tuned_hgb)]
    tri_stacking = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)

    oof_preds = np.zeros(len(df_ml))
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        tri_stacking.fit(X_tr, y_tr)
        oof_preds[test_idx] = tri_stacking.predict(X_te)

    df_ml['peso_predito_g'] = oof_preds
    df_ml['erro_absoluto_g'] = np.abs(df_ml['peso_g'] - df_ml['peso_predito_g'])
    df_ml['erro_relativo_pct'] = (df_ml['erro_absoluto_g'] / df_ml['peso_g']) * 100.0

    # 3. Sample 10 Representative Slaughter Simulations Across Ages (42 to 52 days)
    simulations = []
    target_ages = [42, 43, 44, 45, 46, 47, 48, 49, 50, 52]
    
    np.random.seed(42)
    for i, age in enumerate(target_ages, 1):
        age_sub = df_ml[df_ml['idade'] == age]
        if len(age_sub) > 0:
            sample_row = age_sub.sample(1, random_state=42).iloc[0]
        else:
            sample_row = df_ml.sample(1, random_state=42).iloc[0]

        real_w = sample_row['peso_g']
        pred_w = sample_row['peso_predito_g']
        err_abs = sample_row['erro_absoluto_g']
        err_pct = sample_row['erro_relativo_pct']

        q25 = df_ml[df_ml['idade'] == age]['peso_g'].quantile(0.25)
        q75 = df_ml[df_ml['idade'] == age]['peso_g'].quantile(0.75)
        if real_w < q25:
            meta_status = 'Abaixo da Meta'
        elif real_w > q75:
            meta_status = 'Acima da Meta'
        else:
            meta_status = 'Na Meta'

        c15_val = sample_row['c15'] if 'c15' in sample_row else 42.0

        simulations.append({
            'Simulação': f'#{i:02d}',
            'Lote_ID': str(sample_row['lote_composto'])[:14],
            'Idade (dias)': int(sample_row['idade']),
            'Peso 1d (c15)': f"{c15_val:.1f}g",
            'Peso 35d': f"{sample_row['peso_dia_35']:.0f}g",
            'Peso Real (g)': round(real_w, 1),
            'Peso Predito (g)': round(pred_w, 1),
            'Erro (g)': round(err_abs, 1),
            'Erro (%)': round(err_pct, 2),
            'Status Meta': meta_status
        })

    sim_df = pd.DataFrame(simulations)
    sim_csv_path = os.path.join(data_proc_dir, 'simulacoes_peso_abate_10_lotes.csv')
    sim_df.to_csv(sim_csv_path, index=False)
    logger.info(f"10 Simulations saved to {sim_csv_path}")

    # 4. Plot 10 Simulations Predito vs Real Comparison
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    fig, ax = plt.subplots(figsize=(12, 6))

    x_indices = np.arange(len(sim_df))
    width = 0.35

    rects1 = ax.bar(x_indices - width/2, sim_df['Peso Real (g)'], width, label='Peso Real Observado (g)', color='#1f77b4', alpha=0.9)
    rects2 = ax.bar(x_indices + width/2, sim_df['Peso Predito (g)'], width, label='Peso Predito pelo Modelo (g)', color='#ff7f0e', alpha=0.9)

    for i in range(len(sim_df)):
        err_val = sim_df.loc[i, 'Erro (g)']
        err_pct = sim_df.loc[i, 'Erro (%)']
        max_bar = max(sim_df.loc[i, 'Peso Real (g)'], sim_df.loc[i, 'Peso Predito (g)'])
        ax.text(i, max_bar + 40, f"Δ {err_val:.0f}g\n({err_pct:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#d62728')

    ax.set_title("Simulação Comparativa de Peso Predito vs Real em 10 Lotes de Abate", fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel("Lote de Abate Simulado (Idade)", fontsize=12)
    ax.set_ylabel("Peso Corporal no Abate (g)", fontsize=12)
    ax.set_xticks(x_indices)
    labels_x = [f"{row['Simulação']}\n({row['Idade (dias)']}d)" for _, row in sim_df.iterrows()]
    ax.set_xticklabels(labels_x, fontsize=10)
    ax.set_ylim(1500, max(sim_df['Peso Real (g)'].max(), sim_df['Peso Predito (g)'].max()) + 250)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, 'simulacoes_predito_vs_real_10lotes.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info("Simulations and plot created successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_slaughter_simulations()
