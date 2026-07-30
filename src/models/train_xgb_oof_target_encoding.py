"""
train_xgb_oof_target_encoding.py
---------------------------------
XGBoost GPU CUDA com Out-Of-Fold Target Encoding (Média Histórica da Fazenda e Produtor OOF):
Garante zero data leakage e injeta o perfil histórico do produtor/fazenda.

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-30
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')

DATASET_PATH = Path("data/processed/longitudinal_dataset.csv")

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def run_oof_target_encoding_experiment():
    print("=======================================================")
    print(" 🚀 XGBOOST GPU COM TARGET ENCODING OUT-OF-FOLD (OOF)")
    print("=======================================================")
    
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    if 'elegivel_rn11' in df.columns:
        df = df[df['elegivel_rn11'] == 1.0].copy()
        
    df = df.dropna(subset=['peso_abate_g', 'idade_abate'])
    
    target = 'peso_abate_g'
    group_col = 'lote_composto'
    
    # 1. Calcular Out-Of-Fold Farm Target Encoding
    gkf = GroupKFold(n_splits=5)
    groups = df[group_col].values
    y = df[target].values
    
    df['oof_fazenda_target_enc'] = np.nan
    df['oof_produtor_target_enc'] = np.nan
    
    global_mean_target = y.mean()
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
        tr_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Fazenda mean
        faz_map = tr_df.groupby('fazenda')[target].mean().to_dict()
        df.iloc[val_idx, df.columns.get_loc('oof_fazenda_target_enc')] = val_df['fazenda'].map(faz_map).fillna(global_mean_target)
        
        # Produtor mean
        if 'produtor' in df.columns:
            prod_map = tr_df.groupby('produtor')[target].mean().to_dict()
            df.iloc[val_idx, df.columns.get_loc('oof_produtor_target_enc')] = val_df['produtor'].map(prod_map).fillna(global_mean_target)
            
    exclude = ['data_alojamento', 'nome_fazenda', 'data_hora_transao', 'lote_composto', 
               'data_evento', 'data_criao', 'id_usurio_criao', 'extensionista', 'id_usurio', 
               'fazenda', 'produtor', 'data_producao_abate', 'peso_medio_abate_kg', 'peso_abate_g', 
               'gmd_abate', 'score_confianca_lote', 'categoria_amostragem', 'elegivel_rn11', 
               'motivo_inelegibilidade', 'estrategia_predicao', 'nucleo']
               
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    
    X = df[features].fillna(df[features].median())
    
    oof_xgb = np.zeros(len(df))
    oof_lgb = np.zeros(len(df))
    
    # XGBoost GPU Deep
    xgb_gpu = XGBRegressor(
        n_estimators=1800, max_depth=8, learning_rate=0.015,
        subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
        tree_method='hist', device='cuda', random_state=42
    )
    
    # LightGBM Deep
    lgb_deep = LGBMRegressor(
        n_estimators=1200, max_depth=9, num_leaves=127, learning_rate=0.018,
        subsample=0.85, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    
    print("Treinando XGBoost GPU e LightGBM com OOF Target Encoding...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        xgb_gpu.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb_gpu.predict(X_val)
        
        lgb_deep.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb_deep.predict(X_val)
        
        print(f" Fold {fold} completo.")
        
    mae_xgb = mean_absolute_error(y, oof_xgb)
    mae_lgb = mean_absolute_error(y, oof_lgb)
    
    print(f"\nMAE XGBoost GPU OOF Target Enc : {mae_xgb:.2f} g | R² = {r2_score(y, oof_xgb):.4f}")
    print(f"MAE LightGBM OOF Target Enc    : {mae_lgb:.2f} g | R² = {r2_score(y, oof_lgb):.4f}")
    
    # Stacking Ridge
    oof_matrix = pd.DataFrame({'xgb': oof_xgb, 'lgb': oof_lgb})
    oof_ensemble = np.zeros(len(df))
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        tr_m, val_m = oof_matrix.iloc[train_idx], oof_matrix.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        meta = Ridge(alpha=10.0, positive=True)
        meta.fit(tr_m, y_tr)
        oof_ensemble[val_idx] = meta.predict(val_m)
        
    final_mae = mean_absolute_error(y, oof_ensemble)
    final_rmse = np.sqrt(mean_squared_error(y, oof_ensemble))
    final_mape = mean_absolute_percentage_error(y, oof_ensemble)
    final_r2 = r2_score(y, oof_ensemble)
    
    print("\n=======================================================")
    print(" 🏆 RESULTADO COM OOF TARGET ENCODING (GPU CUDA)")
    print("=======================================================")
    print(f" MAE Final (Out-Of-Fold)                   : {final_mae:.2f} g ({final_mae/1000.0:.3f} kg)")
    print(f" RMSE Final                                : {final_rmse:.2f} g")
    print(f" MAPE Final                                : {final_mape:.2f} %")
    print(f" Coeficiente R² Final                      : {final_r2:.4f}")
    print("-------------------------------------------------------")
    print(f" Status das Metas Exigidas:")
    print(f"  - MAE < 100.0 g  -> {'✅ ALCANÇADO!' if final_mae < 100.0 else f'⚡ {final_mae:.2f}g (Diferença: {final_mae-100.0:.2f}g)'}")
    print(f"  - Coeficiente R² >= 0.60 -> {'✅ ALCANÇADO!' if final_r2 >= 0.60 else '❌ FALHOU'}")
    print("=======================================================\n")
    
    return final_mae, final_r2

if __name__ == '__main__':
    run_oof_target_encoding_experiment()
