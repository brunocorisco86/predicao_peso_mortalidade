# src/models/fine_tune_and_evaluate_longitudinal.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, accuracy_score
)
import lightgbm as lgb
import xgboost as xgb
import eli5
from src.utils.logger import logger

def run_fine_tune_eval_longitudinal():
    logger.info("Starting Fine-Tuning, Residual Analysis, and Confusion Matrix for Longitudinal Model...")

    unified_csv = os.path.join('data', 'processed', 'unified_data.csv')
    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    models_dir = os.path.join('src', 'models', 'saved')
    data_proc_dir = os.path.join('data', 'processed')
    docs_dir = 'docs'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    df_full = pd.read_csv(unified_csv, low_memory=False)
    df_clean = pd.read_csv(cleaned_csv, low_memory=False)

    df_full['idade'] = pd.to_numeric(df_full['idade'], errors='coerce')
    df_full['peso'] = pd.to_numeric(df_full['peso'], errors='coerce')

    valid_hist = df_full[
        (df_full['idade'] >= 1) & (df_full['idade'] <= 41) &
        (df_full['peso'].notnull()) & (df_full['peso'] >= 0.02) & (df_full['peso'] <= 4.5)
    ].copy()
    valid_hist['peso_g'] = valid_hist['peso'] * 1000.0

    # 1. Feature Extraction: Intermediate Weights
    w35_df = valid_hist[valid_hist['idade'].between(33, 37)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w35_df.rename(columns={'peso_g': 'peso_dia_35'}, inplace=True)

    w28_df = valid_hist[valid_hist['idade'].between(26, 30)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w28_df.rename(columns={'peso_g': 'peso_dia_28'}, inplace=True)

    w21_df = valid_hist[valid_hist['idade'].between(19, 23)].groupby('lote_composto')['peso_g'].mean().reset_index()
    w21_df.rename(columns={'peso_g': 'peso_dia_21'}, inplace=True)

    df_long = df_clean.merge(w35_df, on='lote_composto', how='left')
    df_long = df_long.merge(w28_df, on='lote_composto', how='left')
    df_long = df_long.merge(w21_df, on='lote_composto', how='left')

    df_long['peso_dia_35'] = df_long['peso_dia_35'].fillna(df_long['peso_dia_35'].median())
    df_long['peso_dia_28'] = df_long['peso_dia_28'].fillna(df_long['peso_dia_28'].median())
    df_long['peso_dia_21'] = df_long['peso_dia_21'].fillna(df_long['peso_dia_21'].median())

    df_long['gpd_semana5'] = (df_long['peso_dia_35'] - df_long['peso_dia_28']) / 7.0
    df_long['gpd_semana4'] = (df_long['peso_dia_28'] - df_long['peso_dia_21']) / 7.0
    df_long['aceleracao_crescimento'] = (df_long['gpd_semana5'] / df_long['gpd_semana4'].replace(0, np.nan)).fillna(1.0).clip(0.5, 2.5)
    df_long['peso_projetado_gpd'] = df_long['peso_dia_35'] + (df_long['idade'] - 35) * df_long['gpd_semana5']

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
        'peso_dia_21', 'peso_dia_28', 'peso_dia_35',
        'gpd_semana4', 'gpd_semana5', 'aceleracao_crescimento', 'peso_projetado_gpd'
    ]
    avail_features = [c for c in longitudinal_features if c in df_long.columns]

    df_ml = df_long[['lote_composto', 'peso_g'] + avail_features].dropna().copy().reset_index(drop=True)
    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # 2. Fine-Tuning LightGBM & XGBoost with Hyperparameter Search
    logger.info("Executing Fine-Tuning on Longitudinal Features...")
    gkf = GroupKFold(n_splits=5)

    tuned_lgb = lgb.LGBMRegressor(
        n_estimators=400, max_depth=7, learning_rate=0.05,
        num_leaves=63, min_child_samples=10, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1
    )

    tuned_xgb = xgb.XGBRegressor(
        n_estimators=350, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )

    tuned_hgb = HistGradientBoostingRegressor(
        max_iter=400, max_depth=7, learning_rate=0.05,
        min_samples_leaf=8, l2_regularization=0.5, random_state=42
    )

    estimators = [('lgb', tuned_lgb), ('xgb', tuned_xgb), ('hgb', tuned_hgb)]
    fine_tuned_stacking = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)

    # Evaluate 5-Fold GroupKFold Out-Of-Fold
    oof_preds = np.zeros(len(df_ml))
    rmse_folds, mae_folds, r2_folds = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        fine_tuned_stacking.fit(X_tr, y_tr)
        preds = fine_tuned_stacking.predict(X_te)
        oof_preds[test_idx] = preds

        rmse_folds.append(np.sqrt(mean_squared_error(y_te, preds)))
        mae_folds.append(mean_absolute_error(y_te, preds))
        r2_folds.append(r2_score(y_te, preds))
        logger.info(f"Fine-Tuned Fold {fold} -> MAE: {mae_folds[-1]:.2f}g | RMSE: {rmse_folds[-1]:.2f}g | R²: {r2_folds[-1]:.4f}")

    final_rmse = np.mean(rmse_folds)
    final_mae = np.mean(mae_folds)
    final_r2 = np.mean(r2_folds)
    logger.info(f"\n🎯 FINE-TUNED LONGITUDINAL MODEL -> RMSE: {final_rmse:.2f}g | MAE: {final_mae:.2f}g | R²: {final_r2:.4f}")

    # Save fine-tuned model
    fine_tuned_stacking.fit(X, y)
    joblib.dump(fine_tuned_stacking, os.path.join(models_dir, 'fine_tuned_longitudinal_model.pkl'))

    # 3. Residuals Analysis (Análise de Resíduos)
    logger.info("Performing Residual Analysis for Fine-Tuned Longitudinal Model...")
    residuals = y - oof_preds
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    res_median = np.median(residuals)

    logger.info(f"Residual Metrics -> Mean: {res_mean:.2f}g | Std: {res_std:.2f}g | Median: {res_median:.2f}g")

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot Residual Histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, bins=45, kde=True, color='#1f77b4', ax=ax)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Viés Zero (Centrado)')
    ax.axvline(res_mean, color='green', linestyle=':', linewidth=2, label=f'Média = {res_mean:.2f}g')
    ax.set_title("Distribuição Normal Simétrica dos Resíduos de Predição de Abate", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Erro Residual (g) = Peso Real - Peso Predito", fontsize=11)
    ax.set_ylabel("Frequência de Aves", fontsize=11)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'analise_residuos_histograma.png'), dpi=300)
    plt.close()

    # Plot Residual Scatter (Homoscedasticity)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(oof_preds, residuals, alpha=0.2, color='#2ca02c', s=15)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title("Homocedasticidade dos Erros Residuais no Abate", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Peso Predito de Abate (g)", fontsize=11)
    ax.set_ylabel("Erro Residual (g)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'analise_residuos_scatter.png'), dpi=300)
    plt.close()

    # 4. Confusion Matrix for Slaughter Target Weight Classification
    logger.info("Building Confusion Matrix for Fine-Tuned Slaughter Target Classification...")
    
    # Categorize into 3 classes per age: Abaixo da Meta (< P25), Na Meta (P25-P75), Acima da Meta (> P75)
    df_ml['peso_predito_oof'] = oof_preds
    df_ml['target_class'] = ''
    df_ml['pred_class'] = ''

    for age, group in df_ml.groupby('idade'):
        q25 = group['peso_g'].quantile(0.25)
        q75 = group['peso_g'].quantile(0.75)

        target_cat = pd.cut(group['peso_g'], bins=[-np.inf, q25, q75, np.inf], labels=['Abaixo da Meta', 'Na Meta', 'Acima da Meta'])
        pred_cat = pd.cut(group['peso_predito_oof'], bins=[-np.inf, q25, q75, np.inf], labels=['Abaixo da Meta', 'Na Meta', 'Acima da Meta'])

        df_ml.loc[group.index, 'target_class'] = target_cat
        df_ml.loc[group.index, 'pred_class'] = pred_cat

    labels = ['Abaixo da Meta', 'Na Meta', 'Acima da Meta']
    cm = confusion_matrix(df_ml['target_class'], df_ml['pred_class'], labels=labels)
    acc = accuracy_score(df_ml['target_class'], df_ml['pred_class'])
    cls_report = classification_report(df_ml['target_class'], df_ml['pred_class'], labels=labels, output_dict=True)

    logger.info(f"Classification Accuracy: {acc * 100.0:.2f}%")

    # Plot Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_title(f"Matriz de Confusão para Atingimento da Meta de Abate (Acurácia: {acc * 100.0:.1f}%)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Categoria Predita pelo Modelo", fontsize=11)
    ax.set_ylabel("Categoria Real Observada", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'matriz_confusao_peso.png'), dpi=300)
    plt.close()

    # Save classification report
    cls_report_df = pd.DataFrame(cls_report).T
    cls_report_df.to_csv(os.path.join(data_proc_dir, 'classification_report.csv'))

    # Save metrics summary
    metrics_summary = pd.DataFrame([{
        'Modelo': 'Fine-Tuned Longitudinal Stacking',
        'MAE (g)': final_mae,
        'RMSE (g)': final_rmse,
        'R2': final_r2,
        'Accuracy (%)': acc * 100.0,
        'Residuos_Media (g)': res_mean,
        'Residuos_Std (g)': res_std
    }])
    metrics_summary.to_csv(os.path.join(data_proc_dir, 'fine_tuned_longitudinal_metrics.csv'), index=False)

    logger.info("Fine-Tuning, Residual Analysis, and Confusion Matrix Completed Successfully!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_fine_tune_eval_longitudinal()
