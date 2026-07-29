# src/models/fine_tune_slaughter_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
from src.utils.logger import logger

def run_fine_tuning():
    logger.info("Starting Hyperparameter Fine-Tuning for Slaughter Weight Prediction...")

    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    data_proc_dir = os.path.join('data', 'processed')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(cleaned_csv, low_memory=False)
    logger.info(f"Loaded cleaned_data.csv with {len(df)} slaughter age records.")

    # 1. Advanced Feature Engineering
    df['mortalidade_pct'] = (df['mortalidade'] / df['cab_alojadas'].replace(0, np.nan)) * 100.0
    df['descartados_pct'] = (df['descartados'] / df['cab_alojadas'].replace(0, np.nan)) * 100.0
    df['taxa_perda_total'] = df['mortalidade_pct'] + df['descartados_pct']

    # Categorical Frequency / One-Hot Encoding if available
    cat_cols = ['c16', 'c17']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    feature_candidates = [
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'mortalidade_pct', 'descartados_pct', 'taxa_perda_total',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'c16', 'c17', 'f07', 'f15', 'x02'
    ]
    avail_features = [c for c in feature_candidates if c in df.columns]
    logger.info(f"Engineered features for tuning: {avail_features}")

    df_ml = df[['lote_composto', 'peso_g'] + avail_features].dropna().copy()
    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # 2. Compare Algorithms via 5-Fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    models = {
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    }

    results = []
    best_model_name = None
    best_overall_rmse = float('inf')
    best_model_obj = None

    for name, model in models.items():
        rmse_list, mae_list, r2_list = [], [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            rmse_list.append(np.sqrt(mean_squared_error(y_te, preds)))
            mae_list.append(mean_absolute_error(y_te, preds))
            r2_list.append(r2_score(y_te, preds))

        mean_rmse = np.mean(rmse_list)
        mean_mae = np.mean(mae_list)
        mean_r2 = np.mean(r2_list)

        logger.info(f"Model {name} -> RMSE: {mean_rmse:.2f}g | MAE: {mean_mae:.2f}g | R²: {mean_r2:.4f}")
        results.append({'Modelo': name, 'RMSE (g)': mean_rmse, 'MAE (g)': mean_mae, 'R2': mean_r2})

        if mean_rmse < best_overall_rmse:
            best_overall_rmse = mean_rmse
            best_model_name = name
            best_model_obj = model

    results_df = pd.DataFrame(results).sort_values(by='RMSE (g)')
    results_df.to_csv(os.path.join(data_proc_dir, 'algorithm_comparison_slaughter.csv'), index=False)

    # 3. Fine-Tuning Hyperparameters for the Best Model (HistGradientBoosting / GradientBoosting)
    logger.info(f"Fine-Tuning Hyperparameters for Best Model: {best_model_name}...")

    if best_model_name == 'HistGradientBoosting':
        param_dist = {
            'max_iter': [100, 200, 300],
            'learning_rate': [0.03, 0.05, 0.1],
            'max_depth': [6, 10, 15, None],
            'min_samples_leaf': [10, 20, 30, 50],
            'l2_regularization': [0.0, 0.1, 1.0]
        }
        search_model = HistGradientBoostingRegressor(random_state=42)
    elif best_model_name == 'GradientBoosting':
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.03, 0.05, 0.1],
            'max_depth': [5, 8, 12],
            'min_samples_split': [5, 10, 20],
            'subsample': [0.7, 0.8, 1.0]
        }
        search_model = GradientBoostingRegressor(random_state=42)
    else:
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        search_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)

    random_search = RandomizedSearchCV(
        search_model,
        param_distributions=param_dist,
        n_iter=15,
        cv=gkf.split(X, y, groups),
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X, y)

    best_tuned_model = random_search.best_estimator_
    logger.info(f"Best Hyperparameters: {random_search.best_params_}")

    # Evaluate Tuned Model with 5-Fold GroupKFold
    tuned_rmse, tuned_mae, tuned_r2 = [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        best_tuned_model.fit(X_tr, y_tr)
        preds = best_tuned_model.predict(X_te)

        tuned_rmse.append(np.sqrt(mean_squared_error(y_te, preds)))
        tuned_mae.append(mean_absolute_error(y_te, preds))
        tuned_r2.append(r2_score(y_te, preds))

    final_rmse = float(np.mean(tuned_rmse))
    final_mae = float(np.mean(tuned_mae))
    final_r2 = float(np.mean(tuned_r2))

    logger.info(f"FINAL TUNED MODEL ({best_model_name}) -> RMSE: {final_rmse:.2f}g | MAE: {final_mae:.2f}g | R²: {final_r2:.4f}")

    # Save tuned model
    joblib.dump(best_tuned_model, os.path.join(models_dir, 'fine_tuned_slaughter_model.pkl'))

    # Save fine tuning report
    tuning_summary = pd.DataFrame([{
        'Modelo_Base': 'Random Forest Baseline',
        'RMSE_Anterior (g)': 170.56,
        'MAE_Anterior (g)': 127.15,
        'R2_Anterior': 0.2925,
        'Modelo_Tuned': best_model_name,
        'RMSE_Novo (g)': final_rmse,
        'MAE_Novo (g)': final_mae,
        'R2_Novo': final_r2,
        'Melhoria_RMSE (%)': ((170.56 - final_rmse) / 170.56) * 100.0
    }])
    tuning_summary.to_csv(os.path.join(data_proc_dir, 'fine_tuning_summary.csv'), index=False)

    # 4. Generate Plot of Tuned Predictions vs Real
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Train-test split for visualization
    split_idx = int(len(X) * 0.8)
    X_tr_vis, X_te_vis = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr_vis, y_te_vis = y.iloc[:split_idx], y.iloc[split_idx:]

    best_tuned_model.fit(X_tr_vis, y_tr_vis)
    preds_vis = best_tuned_model.predict(X_te_vis)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_te_vis, preds_vis, alpha=0.25, color='#1f77b4', s=16)
    min_v, max_v = min(y_te_vis.min(), preds_vis.min()), max(y_te_vis.max(), preds_vis.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Predição Perfeita (1:1)')
    ax.set_title(f"Modelo Fine-Tuned ({best_model_name})\nRMSE: {final_rmse:.1f}g | MAE: {final_mae:.1f}g | R²: {final_r2:.4f}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Peso Real no Abate (g)", fontsize=11)
    ax.set_ylabel("Peso Predito no Abate (g)", fontsize=11)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predito_vs_observado_peso.png'), dpi=300)
    plt.close()

    logger.info("Fine-Tuning completed successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_fine_tuning()
