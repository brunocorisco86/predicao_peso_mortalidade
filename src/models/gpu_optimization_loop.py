"""
gpu_optimization_loop.py
-------------------------
Loop de Otimização Preditiva em GPU para alcançar as metas:
- MAE < 100.0 g
- Coeficiente R² >= 0.6000

Utiliza:
- XGBoost GPU (tree_method='hist', device='cuda')
- CatBoost GPU (task_type='GPU')
- LightGBM & HistGradientBoosting & ExtraTrees
- Stacking Ensemble (Meta-Regressor Ridge)
- Validação Cruzada 5-Fold GroupKFold por lote_composto (Zero Data Leakage)

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-30
"""

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
import warnings
import time
import os
import sys

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")

DATASET_PATH = Path("data/processed/longitudinal_dataset.csv")
PLOTS_DIR = Path("plots")
DOCS_DIR = Path("docs")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def run_gpu_optimization_loop():
    print("=======================================================")
    print(" ⚡ INICIANDO LOOP DE OTIMIZAÇÃO PREDITIVA EM GPU CUDA")
    print("=======================================================")
    
    start_time = time.time()
    
    # 1. Carregar Dataset Longitudinal
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    
    # Filtrar apenas lotes elegíveis pela RN-11
    if 'elegivel_rn11' in df.columns:
        df = df[df['elegivel_rn11'] == 1.0].copy()
        
    df = df.dropna(subset=['peso_abate_g', 'idade_abate'])
    
    target = 'peso_abate_g'
    group_col = 'lote_composto'
    
    exclude = ['data_alojamento', 'nome_fazenda', 'data_hora_transao', 'lote_composto', 
               'data_evento', 'data_criao', 'id_usurio_criao', 'extensionista', 'id_usurio', 
               'fazenda', 'produtor', 'data_producao_abate', 'peso_medio_abate_kg', 'peso_abate_g', 
               'gmd_abate', 'score_confianca_lote', 'categoria_amostragem', 'elegivel_rn11', 
               'motivo_inelegibilidade', 'estrategia_predicao', 'nucleo']
               
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    
    print(f"Total de Lotes Aptos (RN-11): {len(df):,}")
    print(f"Total de Atributos Explicativos (Features): {len(features)}")
    
    X = df[features].fillna(df[features].median())
    y = df[target].values
    groups = df[group_col].values
    
    gkf = GroupKFold(n_splits=5)
    
    # ---------------------------------------------------------
    # Define Hyperparameter Configurations for GPU Loop Iterations
    # ---------------------------------------------------------
    xgb_configs = [
        {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist', 'device': 'cuda', 'random_state': 42},
        {'n_estimators': 600, 'max_depth': 7, 'learning_rate': 0.03, 'subsample': 0.85, 'colsample_bytree': 0.85, 'tree_method': 'hist', 'device': 'cuda', 'random_state': 42},
        {'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.8, 'gamma': 0.1, 'tree_method': 'hist', 'device': 'cuda', 'random_state': 42},
        {'n_estimators': 1500, 'max_depth': 9, 'learning_rate': 0.015, 'subsample': 0.9, 'colsample_bytree': 0.75, 'reg_alpha': 0.5, 'reg_lambda': 1.0, 'tree_method': 'hist', 'device': 'cuda', 'random_state': 42}
    ]
    
    lgb_configs = [
        {'n_estimators': 400, 'max_depth': 7, 'num_leaves': 63, 'learning_rate': 0.03, 'subsample': 0.85, 'colsample_bytree': 0.85, 'random_state': 42, 'verbose': -1},
        {'n_estimators': 800, 'max_depth': 9, 'num_leaves': 127, 'learning_rate': 0.02, 'subsample': 0.9, 'colsample_bytree': 0.8, 'random_state': 42, 'verbose': -1}
    ]
    
    iteration_history = []
    best_oof_preds = None
    best_mae = 9999.0
    best_r2 = -9999.0
    best_model_name = ""
    
    oof_predictions_dict = {}
    
    # ---------------------------------------------------------
    # Loop Iterativo de Treinamento e Otimização
    # ---------------------------------------------------------
    iteration_count = 0
    
    print("\n-------------------------------------------------------")
    print(" 🚀 EXECUTANDO LOOP DE EXPERIMENTAÇÃO EM GPU...")
    print("-------------------------------------------------------")
    
    # 1. Loop XGBoost GPU Configs
    for idx, cfg in enumerate(xgb_configs, 1):
        iteration_count += 1
        model_name = f"XGBoost_GPU_Config_{idx}"
        print(f"\n[Iteração {iteration_count}] Testando {model_name} (depth={cfg['max_depth']}, lr={cfg['learning_rate']}, n_est={cfg['n_estimators']})...")
        
        oof = np.zeros(len(df))
        maes, rmses, mapes, r2s = [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            xgb = XGBRegressor(**cfg)
            xgb.fit(X_tr, y_tr)
            preds = xgb.predict(X_val)
            oof[val_idx] = preds
            
            maes.append(mean_absolute_error(y_val, preds))
            rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
            mapes.append(mean_absolute_percentage_error(y_val, preds))
            r2s.append(r2_score(y_val, preds))
            
        mean_mae = np.mean(maes)
        mean_rmse = np.mean(rmses)
        mean_mape = np.mean(mapes)
        mean_r2 = np.mean(r2s)
        
        oof_predictions_dict[model_name] = oof
        
        print(f"  -> Resultado {model_name}: MAE = {mean_mae:.2f}g | RMSE = {mean_rmse:.2f}g | MAPE = {mean_mape:.2f}% | R² = {mean_r2:.4f}")
        
        iteration_history.append({
            'Iteration': iteration_count,
            'Model': model_name,
            'MAE (g)': mean_mae,
            'RMSE (g)': mean_rmse,
            'MAPE (%)': mean_mape,
            'R²': mean_r2
        })
        
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_r2 = mean_r2
            best_oof_preds = oof
            best_model_name = model_name
            
    # 2. Loop LightGBM Configs
    for idx, cfg in enumerate(lgb_configs, 1):
        iteration_count += 1
        model_name = f"LightGBM_Config_{idx}"
        print(f"\n[Iteração {iteration_count}] Testando {model_name} (leaves={cfg['num_leaves']}, lr={cfg['learning_rate']})...")
        
        oof = np.zeros(len(df))
        maes, rmses, mapes, r2s = [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            lgb = LGBMRegressor(**cfg)
            lgb.fit(X_tr, y_tr)
            preds = lgb.predict(X_val)
            oof[val_idx] = preds
            
            maes.append(mean_absolute_error(y_val, preds))
            rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
            mapes.append(mean_absolute_percentage_error(y_val, preds))
            r2s.append(r2_score(y_val, preds))
            
        mean_mae = np.mean(maes)
        mean_rmse = np.mean(rmses)
        mean_mape = np.mean(mapes)
        mean_r2 = np.mean(r2s)
        
        oof_predictions_dict[model_name] = oof
        
        print(f"  -> Resultado {model_name}: MAE = {mean_mae:.2f}g | RMSE = {mean_rmse:.2f}g | MAPE = {mean_mape:.2f}% | R² = {mean_r2:.4f}")
        
        iteration_history.append({
            'Iteration': iteration_count,
            'Model': model_name,
            'MAE (g)': mean_mae,
            'RMSE (g)': mean_rmse,
            'MAPE (%)': mean_mape,
            'R²': mean_r2
        })
        
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_r2 = mean_r2
            best_oof_preds = oof
            best_model_name = model_name

    # 3. Testar CatBoost se disponível
    if HAS_CATBOOST:
        iteration_count += 1
        model_name = "CatBoost_GPU"
        print(f"\n[Iteração {iteration_count}] Testando {model_name}...")
        
        oof = np.zeros(len(df))
        maes, rmses, mapes, r2s = [], [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            cb = CatBoostRegressor(iterations=800, learning_rate=0.03, depth=7, task_type='GPU', verbose=0, random_seed=42)
            cb.fit(X_tr, y_tr)
            preds = cb.predict(X_val)
            oof[val_idx] = preds
            
            maes.append(mean_absolute_error(y_val, preds))
            rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
            mapes.append(mean_absolute_percentage_error(y_val, preds))
            r2s.append(r2_score(y_val, preds))
            
        mean_mae = np.mean(maes)
        mean_rmse = np.mean(rmses)
        mean_mape = np.mean(mapes)
        mean_r2 = np.mean(r2s)
        
        oof_predictions_dict[model_name] = oof
        print(f"  -> Resultado {model_name}: MAE = {mean_mae:.2f}g | RMSE = {mean_rmse:.2f}g | MAPE = {mean_mape:.2f}% | R² = {mean_r2:.4f}")
        
        iteration_history.append({
            'Iteration': iteration_count,
            'Model': model_name,
            'MAE (g)': mean_mae,
            'RMSE (g)': mean_rmse,
            'MAPE (%)': mean_mape,
            'R²': mean_r2
        })
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_r2 = mean_r2
            best_oof_preds = oof
            best_model_name = model_name

    # 4. CONSTRUÇÃO DO ENSEMBLE STACKING (META-REGRESSOR OPTIMAL)
    iteration_count += 1
    model_name = "Ensemble_Stacking_MetaRidge"
    print(f"\n[Iteração {iteration_count}] Treinando {model_name} (Combinação Stacking Out-Of-Fold)...")
    
    # Matriz de OOF Predições dos modelos base
    oof_matrix = pd.DataFrame(oof_predictions_dict)
    
    oof_stacking = np.zeros(len(df))
    maes_s, rmses_s, mapes_s, r2s_s = [], [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        oof_tr, oof_val = oof_matrix.iloc[train_idx], oof_matrix.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Meta-modelo Ridge com alpha positivo
        meta_ridge = Ridge(alpha=10.0, positive=True)
        meta_ridge.fit(oof_tr, y_tr)
        
        preds = meta_ridge.predict(oof_val)
        oof_stacking[val_idx] = preds
        
        maes_s.append(mean_absolute_error(y_val, preds))
        rmses_s.append(np.sqrt(mean_squared_error(y_val, preds)))
        mapes_s.append(mean_absolute_percentage_error(y_val, preds))
        r2s_s.append(r2_score(y_val, preds))
        
    mean_mae_s = np.mean(maes_s)
    mean_rmse_s = np.mean(rmses_s)
    mean_mape_s = np.mean(mapes_s)
    mean_r2_s = np.mean(r2s_s)
    
    print(f"  -> Resultado {model_name}: MAE = {mean_mae_s:.2f}g | RMSE = {mean_rmse_s:.2f}g | MAPE = {mean_mape_s:.2f}% | R² = {mean_r2_s:.4f}")
    
    iteration_history.append({
        'Iteration': iteration_count,
        'Model': model_name,
        'MAE (g)': mean_mae_s,
        'RMSE (g)': mean_rmse_s,
        'MAPE (%)': mean_mape_s,
        'R²': mean_r2_s
    })
    
    if mean_mae_s < best_mae:
        best_mae = mean_mae_s
        best_r2 = mean_r2_s
        best_oof_preds = oof_stacking
        best_model_name = model_name

    # ---------------------------------------------------------
    # RESUMO DOS RESULTADOS E CHECAGEM DAS METAS EXIGIDAS
    # ---------------------------------------------------------
    df_history = pd.DataFrame(iteration_history).sort_values('MAE (g)')
    
    elapsed_time = time.time() - start_time
    
    print("\n=======================================================")
    print(" 📊 RESUMO DO LOOP DE OTIMIZAÇÃO EM GPU DA C.VALE")
    print("=======================================================")
    print(f" Tempo Total de Execução: {elapsed_time:.1f} segundos ({elapsed_time/60.0:.2f} minutos)")
    print(df_history.to_string(index=False))
    print("-------------------------------------------------------")
    print(f" 🏆 MODELO CAMPEÃO SELECIONADO: {best_model_name}")
    print(f"    - MAE  : {best_mae:.2f} g  (Meta: < 100.0 g)")
    print(f"    - R²   : {best_r2:.4f}     (Meta: >= 0.6000)")
    print("=======================================================")
    
    # Checagem estrita das metas
    meta_mae_atingida = best_mae < 100.0
    meta_r2_atingida = best_r2 >= 0.6000
    
    if meta_mae_atingida and meta_r2_atingida:
        print(" 🎯 METAS DA C.VALE ATINGIDAS COM SUCESSO! (MAE < 100g e R² >= 0.60)")
    else:
        print(f" ℹ️ Status das Metas: MAE < 100g -> {meta_mae_atingida} | R² >= 0.60 -> {meta_r2_atingida}")
        
    # ---------------------------------------------------------
    # GERAÇÃO DE GRÁFICOS DA JORNADA DE OTIMIZAÇÃO
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_history, x='Model', y='MAE (g)', palette='viridis')
    plt.axhline(100.0, color='red', linestyle='--', linewidth=2, label='Meta Exigida (MAE < 100g)')
    plt.title('Jornada de Otimização Preditiva em GPU - MAE (g)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('MAE em Gramas (g)')
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}g', (p.get_x() + p.get_width() / 2., height / 2.), ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ml_optimization_journey.png", dpi=300)
    plt.close()
    
    # Salvar historico em CSV
    df_history.to_csv(Path("data/processed/gpu_optimization_history.csv"), index=False)
    
    return df_history, best_model_name, best_mae, best_r2

if __name__ == '__main__':
    run_gpu_optimization_loop()
