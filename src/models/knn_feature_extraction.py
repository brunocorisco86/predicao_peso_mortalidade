# src/models/knn_feature_extraction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import logger

def run_knn_feature_extraction():
    logger.info("Starting KNN Feature Extraction & Neighborhood-Enhanced Modeling for Slaughter Weight...")

    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    data_proc_dir = os.path.join('data', 'processed')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(cleaned_csv, low_memory=False)
    logger.info(f"Loaded cleaned_data.csv with {len(df)} slaughter age records.")

    # 1. Base Feature Engineering
    df['mortalidade_pct'] = (df['mortalidade'] / df['cab_alojadas'].replace(0, np.nan)) * 100.0
    df['descartados_pct'] = (df['descartados'] / df['cab_alojadas'].replace(0, np.nan)) * 100.0
    df['taxa_perda_total'] = df['mortalidade_pct'] + df['descartados_pct']

    cat_cols = ['c16', 'c17']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    base_features = [
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'mortalidade_pct', 'descartados_pct', 'taxa_perda_total',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'c16', 'c17', 'f07', 'f15', 'x02'
    ]
    avail_features = [c for c in base_features if c in df.columns]

    df_ml = df[['lote_composto', 'peso_g'] + avail_features].dropna().copy().reset_index(drop=True)
    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # 2. Out-Of-Fold KNN Feature Extraction (Avoid Data Leakage)
    logger.info("Extracting Out-Of-Fold KNN Neighborhood Features (K=15 and K=30)...")
    gkf = GroupKFold(n_splits=5)

    knn_feat_g_k15 = np.zeros(len(df_ml))
    knn_feat_std_k15 = np.zeros(len(df_ml))
    knn_feat_g_k30 = np.zeros(len(df_ml))
    knn_dist_nearest = np.zeros(len(df_ml))

    scaler = StandardScaler()

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features for distance calculations
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # KNN K=15
        knn_k15 = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)
        knn_k15.fit(X_tr_scaled, y_tr)
        knn_feat_g_k15[test_idx] = knn_k15.predict(X_te_scaled)

        # KNN K=30
        knn_k30 = KNeighborsRegressor(n_neighbors=30, weights='distance', n_jobs=-1)
        knn_k30.fit(X_tr_scaled, y_tr)
        knn_feat_g_k30[test_idx] = knn_k30.predict(X_te_scaled)

        # Nearest Neighbors Stats
        nn = NearestNeighbors(n_neighbors=15, n_jobs=-1)
        nn.fit(X_tr_scaled)
        distances, indices = nn.kneighbors(X_te_scaled)

        knn_dist_nearest[test_idx] = distances[:, 0] # Distance to 1st nearest neighbor
        # Std dev of neighbor target weights
        for i, row_indices in enumerate(indices):
            knn_feat_std_k15[test_idx[i]] = np.std(y_tr.iloc[row_indices])

    # Append KNN Features to Dataset
    df_ml['knn_pred_weight_k15'] = knn_feat_g_k15
    df_ml['knn_pred_weight_k30'] = knn_feat_g_k30
    df_ml['knn_neighbor_std_k15'] = knn_feat_std_k15
    df_ml['knn_dist_nearest'] = knn_dist_nearest

    knn_features = ['knn_pred_weight_k15', 'knn_pred_weight_k30', 'knn_neighbor_std_k15', 'knn_dist_nearest']
    enhanced_features = avail_features + knn_features
    logger.info(f"Enhanced feature set size: {len(enhanced_features)} (including 4 KNN features).")

    # 3. Train & Evaluate Model WITH KNN Features via 5-Fold GroupKFold
    X_enhanced = df_ml[enhanced_features]

    cv_rmse_base, cv_mae_base, cv_r2_base = [], [], []
    cv_rmse_knn, cv_mae_knn, cv_r2_knn = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        # Baseline model without KNN features
        X_tr_b, X_te_b = X.iloc[train_idx], X.iloc[test_idx]
        y_tr_b, y_te_b = y.iloc[train_idx], y.iloc[test_idx]

        hgb_base = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.1, min_samples_leaf=10, l2_regularization=1.0, random_state=42)
        hgb_base.fit(X_tr_b, y_tr_b)
        p_base = hgb_base.predict(X_te_b)

        cv_rmse_base.append(np.sqrt(mean_squared_error(y_te_b, p_base)))
        cv_mae_base.append(mean_absolute_error(y_te_b, p_base))
        cv_r2_base.append(r2_score(y_te_b, p_base))

        # Enhanced model WITH KNN features
        X_tr_e, X_te_e = X_enhanced.iloc[train_idx], X_enhanced.iloc[test_idx]

        hgb_knn = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.1, min_samples_leaf=10, l2_regularization=1.0, random_state=42)
        hgb_knn.fit(X_tr_e, y_tr_b)
        p_knn = hgb_knn.predict(X_te_e)

        cv_rmse_knn.append(np.sqrt(mean_squared_error(y_te_b, p_knn)))
        cv_mae_knn.append(mean_absolute_error(y_te_b, p_knn))
        cv_r2_knn.append(r2_score(y_te_b, p_knn))

        logger.info(f"Fold {fold} -> Base R²: {cv_r2_base[-1]:.4f} | KNN-Enhanced R²: {cv_r2_knn[-1]:.4f} | KNN MAE: {cv_mae_knn[-1]:.2f}g")

    mean_rmse_base, mean_mae_base, mean_r2_base = np.mean(cv_rmse_base), np.mean(cv_mae_base), np.mean(cv_r2_base)
    mean_rmse_knn, mean_mae_knn, mean_r2_knn = np.mean(cv_rmse_knn), np.mean(cv_mae_knn), np.mean(cv_r2_knn)

    logger.info(f"\n--- PERFORMANCE COMPARISON ---")
    logger.info(f"Modelo Sem KNN   -> RMSE: {mean_rmse_base:.2f}g | MAE: {mean_mae_base:.2f}g | R²: {mean_r2_base:.4f}")
    logger.info(f"Modelo COM KNN   -> RMSE: {mean_rmse_knn:.2f}g | MAE: {mean_mae_knn:.2f}g | R²: {mean_r2_knn:.4f}")

    # Save summary
    comp_df = pd.DataFrame([
        {'Abordagem': 'HistGradientBoosting (Sem KNN)', 'RMSE (g)': mean_rmse_base, 'MAE (g)': mean_mae_base, 'R2': mean_r2_base},
        {'Abordagem': 'HistGradientBoosting + KNN Features', 'RMSE (g)': mean_rmse_knn, 'MAE (g)': mean_mae_knn, 'R2': mean_r2_knn}
    ])
    comp_path = os.path.join(data_proc_dir, 'knn_feature_extraction_results.csv')
    comp_df.to_csv(comp_path, index=False)

    # 4. Fit Final KNN-Enhanced Model and Save
    hgb_final = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.1, min_samples_leaf=10, l2_regularization=1.0, random_state=42)
    hgb_final.fit(X_enhanced, y)
    joblib.dump(hgb_final, os.path.join(models_dir, 'knn_enhanced_slaughter_model.pkl'))

    # 5. Visualization Plot (Comparison)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ['Sem KNN (Baseline)', 'COM KNN Features (Extração)']
    rmses = [mean_rmse_base, mean_rmse_knn]
    maes = [mean_mae_base, mean_mae_knn]

    x_bar = np.arange(len(methods))
    width = 0.35

    ax.bar(x_bar - width/2, rmses, width, label='RMSE (g)', color='#d62728')
    ax.bar(x_bar + width/2, maes, width, label='MAE (g)', color='#1f77b4')

    for i in range(len(methods)):
        ax.text(i - width/2, rmses[i] + 2, f'{rmses[i]:.1f}g', ha='center', va='bottom', fontweight='bold')
        ax.text(i + width/2, maes[i] + 2, f'{maes[i]:.1f}g', ha='center', va='bottom', fontweight='bold')

    ax.set_title("Impacto da Extração de Features via KNN no Peso de Abate", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(methods, fontweight='bold')
    ax.set_ylabel("Erro (gramas)", fontsize=11)
    ax.set_ylim(0, max(rmses) * 1.2)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'knn_feature_extraction_impact.png'), dpi=300)
    plt.close()

    logger.info("KNN Feature Extraction successfully completed!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_knn_feature_extraction()
