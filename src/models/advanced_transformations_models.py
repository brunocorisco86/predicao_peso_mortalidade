# src/models/advanced_transformations_models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import logger

def run_advanced_experiments():
    logger.info("Starting Advanced Feature Transformations & Multi-Model Comparison (Slaughter Age)...")

    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    data_proc_dir = os.path.join('data', 'processed')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(cleaned_csv, low_memory=False)
    logger.info(f"Loaded cleaned_data.csv with {len(df)} slaughter records.")

    # --- 1. Feature Engineering & Transformations --- #
    # A. Zootecnic ratios and percentages
    df['mortalidade_pct'] = (df['mortalidade'] / df['cab_alojadas'].replace(0, np.nan)) * 100.0
    df['descartados_pct'] = (df['descartados'] / df['cab_alojadas'].replace(0, np.nan)) * 100.0
    df['taxa_perda_total'] = df['mortalidade_pct'] + df['descartados_pct']

    # B. Non-linear logarithmic transformations
    df['log_x02_distancia'] = np.log1p(df['x02'].clip(lower=0))
    df['log_cab_alojadas'] = np.log1p(df['cab_alojadas'].clip(lower=0))

    # C. Quadratic mortality effect (severe outbreak non-linearity)
    df['mortalidade_pct_sq'] = df['mortalidade_pct'] ** 2

    # D. Categorical Encodings & Target Encoding for Lineage/Supplier
    cat_cols = ['c16', 'c17']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    # E. Deviation of chick weight relative to lineage mean
    if 'c15' in df.columns and 'c16' in df.columns:
        lineage_c15_mean = df.groupby('c16')['c15'].transform('mean')
        df['c15_dev_lineage'] = df['c15'] - lineage_c15_mean

    feature_candidates = [
        'idade', 'cab_alojadas', 'log_cab_alojadas', 'mortalidade', 'descartados',
        'mortalidade_pct', 'descartados_pct', 'taxa_perda_total', 'mortalidade_pct_sq',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'c15_dev_lineage', 'c16', 'c17',
        'f07', 'f15', 'x02', 'log_x02_distancia'
    ]
    avail_features = [c for c in feature_candidates if c in df.columns]
    logger.info(f"Engineered transformed feature set size: {len(avail_features)}")

    df_ml = df[['lote_composto', 'peso_g'] + avail_features].dropna().copy().reset_index(drop=True)
    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # PowerTransformer scaling for Neural Net & distance models
    pt = PowerTransformer(method='yeo-johnson')
    X_scaled = pd.DataFrame(pt.fit_transform(X), columns=avail_features)

    # --- 2. Advanced Multi-Model Evaluation (5-Fold GroupKFold) --- #
    gkf = GroupKFold(n_splits=5)

    base_models = {
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.1, min_samples_leaf=10, l2_regularization=1.0, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, min_child_samples=15, random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        'NeuralNetwork_MLP': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=250, random_state=42, early_stopping=True),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    }

    eval_results = []
    oof_predictions = {name: np.zeros(len(df_ml)) for name in base_models.keys()}
    oof_predictions['StackingEnsemble'] = np.zeros(len(df_ml))

    for name, model in base_models.items():
        rmse_l, mae_l, r2_l = [], [], []

        for train_idx, test_idx in gkf.split(X, y, groups):
            if name == 'NeuralNetwork_MLP':
                X_tr, X_te = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            else:
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]

            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            oof_predictions[name][test_idx] = preds

            rmse_l.append(np.sqrt(mean_squared_error(y_te, preds)))
            mae_l.append(mean_absolute_error(y_te, preds))
            r2_l.append(r2_score(y_te, preds))

        m_rmse, m_mae, m_r2 = np.mean(rmse_l), np.mean(mae_l), np.mean(r2_l)
        logger.info(f"Model {name:22s} -> RMSE: {m_rmse:.2f}g | MAE: {m_mae:.2f}g | R²: {m_r2:.4f}")
        eval_results.append({'Modelo': name, 'RMSE (g)': m_rmse, 'MAE (g)': m_mae, 'R2': m_r2})

    # --- 3. Stacking Ensemble Meta-Model --- #
    logger.info("Evaluating Stacking Ensemble Regressor (Combining LightGBM + XGBoost + HistGB + MLP)...")

    stack_rmse_l, stack_mae_l, stack_r2_l = [], [], []
    estimators = [
        ('hgb', HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.1, min_samples_leaf=10, random_state=42)),
        ('lgb', lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, random_state=42, verbose=-1)),
        ('xgb', xgb.XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1))
    ]

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        stack = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(),
            n_jobs=-1
        )
        stack.fit(X_tr, y_tr)
        preds = stack.predict(X_te)
        oof_predictions['StackingEnsemble'][test_idx] = preds

        stack_rmse_l.append(np.sqrt(mean_squared_error(y_te, preds)))
        stack_mae_l.append(mean_absolute_error(y_te, preds))
        stack_r2_l.append(r2_score(y_te, preds))

    s_rmse, s_mae, s_r2 = np.mean(stack_rmse_l), np.mean(stack_mae_l), np.mean(stack_r2_l)
    logger.info(f"Model StackingEnsemble      -> RMSE: {s_rmse:.2f}g | MAE: {s_mae:.2f}g | R²: {s_r2:.4f}")
    eval_results.append({'Modelo': 'StackingEnsemble', 'RMSE (g)': s_rmse, 'MAE (g)': s_mae, 'R2': s_r2})

    eval_df = pd.DataFrame(eval_results).sort_values(by='RMSE (g)').reset_index(drop=True)
    eval_df.to_csv(os.path.join(data_proc_dir, 'advanced_transformations_models_results.csv'), index=False)

    best_row = eval_df.iloc[0]
    logger.info(f"\n🏆 BEST PERFORMING MODEL: {best_row['Modelo']} -> RMSE: {best_row['RMSE (g)']:.2f}g | MAE: {best_row['MAE (g)']:.2f}g | R²: {best_row['R2']:.4f}")

    # --- 4. Plot Multi-Model Performance Comparison --- #
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Barplot RMSE
    sns.barplot(data=eval_df, x='RMSE (g)', y='Modelo', ax=ax1, palette='mako')
    ax1.set_title("Comparativo de Erro Quadrático Médio (RMSE em g)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("RMSE (g) - Quanto menor, melhor", fontsize=11)

    for i, row in eval_df.iterrows():
        ax1.text(row['RMSE (g)'] + 1, i, f"{row['RMSE (g)']:.1f}g", va='center', fontweight='bold', fontsize=10)

    # Barplot R2
    sns.barplot(data=eval_df, x='R2', y='Modelo', ax=ax2, palette='viridis')
    ax2.set_title("Comparativo do Coeficiente de Determinação ($R^2$)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("$R^2$ - Quanto maior, melhor", fontsize=11)

    for i, row in eval_df.iterrows():
        ax2.text(row['R2'] + 0.005, i, f"{row['R2']:.4f}", va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'comparativo_modelos_avancados.png'), dpi=300)
    plt.close()

    logger.info("Advanced Transformations & Multi-Model Comparison Completed Successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_advanced_experiments()
