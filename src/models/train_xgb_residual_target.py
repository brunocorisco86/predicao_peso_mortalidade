"""
train_xgb_residual_target.py
------------------------------
Treinamento do XGBoost GPU CUDA sobre o Resíduo de Trajetória Biométrica:
Target Delta: Y_delta = peso_abate_g - pred_gompertz_lote
Predição Final: Y_pred = pred_gompertz_lote + XGBoost_GPU(X)

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
import warnings

warnings.filterwarnings('ignore')

DATASET_PATH = Path("data/processed/longitudinal_dataset.csv")

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def run_residual_target_experiment():
    print("=======================================================")
    print(" 🚀 TREINAMENTO DE RESÍDUO DE TRAJETÓRIA (GPU CUDA)")
    print("=======================================================")
    
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    if 'elegivel_rn11' in df.columns:
        df = df[df['elegivel_rn11'] == 1.0].copy()
        
    df = df.dropna(subset=['peso_abate_g', 'idade_abate', 'pred_gompertz_lote'])
    
    base_pred = df['pred_gompertz_lote'].values
    y_real = df['peso_abate_g'].values
    y_delta = y_real - base_pred
    
    group_col = 'lote_composto'
    
    exclude = ['data_alojamento', 'nome_fazenda', 'data_hora_transao', 'lote_composto', 
               'data_evento', 'data_criao', 'id_usurio_criao', 'extensionista', 'id_usurio', 
               'fazenda', 'produtor', 'data_producao_abate', 'peso_medio_abate_kg', 'peso_abate_g', 
               'gmd_abate', 'score_confianca_lote', 'categoria_amostragem', 'elegivel_rn11', 
               'motivo_inelegibilidade', 'estrategia_predicao', 'nucleo']
               
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    
    X = df[features].fillna(df[features].median())
    groups = df[group_col].values
    
    gkf = GroupKFold(n_splits=5)
    
    oof_delta = np.zeros(len(df))
    oof_train_mae = []
    oof_test_mae = []
    
    xgb_residual = XGBRegressor(
        n_estimators=1500, max_depth=9, learning_rate=0.015,
        subsample=0.85, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=1.5,
        tree_method='hist', device='cuda', random_state=42
    )
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_delta, groups), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        d_tr, d_val = y_delta[train_idx], y_delta[val_idx]
        
        xgb_residual.fit(X_tr, d_tr)
        preds_d_val = xgb_residual.predict(X_val)
        preds_d_tr = xgb_residual.predict(X_tr)
        
        oof_delta[val_idx] = preds_d_val
        
        y_val_pred = base_pred[val_idx] + preds_d_val
        y_tr_pred = base_pred[train_idx] + preds_d_tr
        
        oof_test_mae.append(mean_absolute_error(y_real[val_idx], y_val_pred))
        oof_train_mae.append(mean_absolute_error(y_real[train_idx], y_tr_pred))
        
        print(f" Fold {fold}: Val MAE = {oof_test_mae[-1]:.2f}g | Train MAE = {oof_train_mae[-1]:.2f}g")
        
    final_preds = base_pred + oof_delta
    
    mae_final = mean_absolute_error(y_real, final_preds)
    rmse_final = np.sqrt(mean_squared_error(y_real, final_preds))
    mape_final = mean_absolute_percentage_error(y_real, final_preds)
    r2_final = r2_score(y_real, final_preds)
    
    train_mae_avg = np.mean(oof_train_mae)
    test_mae_avg = np.mean(oof_test_mae)
    gap_overfitting = ((test_mae_avg - train_mae_avg) / train_mae_avg) * 100.0
    
    print("\n=======================================================")
    print(" 🏆 RESULTADO DO MODELO DE RESÍDUO DE TRAJETÓRIA (GPU)")
    print("=======================================================")
    print(f" MAE Final (Out-Of-Fold)                   : {mae_final:.2f} g ({mae_final/1000.0:.3f} kg)")
    print(f" RMSE Final                                : {rmse_final:.2f} g")
    print(f" MAPE Final                                : {mape_final:.2f} %")
    print(f" Coeficiente R² Final                      : {r2_final:.4f}")
    print(f" Gap Treino vs Validação (Overfitting Check): {gap_overfitting:.2f}% (Tolerância: < 15%)")
    print("-------------------------------------------------------")
    print(f" Status das Metas Exigidas:")
    print(f"  - MAE < 100.0 g  -> {'✅ ALCANÇADO!' if mae_final < 100.0 else f'⚡ {mae_final:.2f}g'}")
    print(f"  - Coeficiente R² >= 0.60 -> {'✅ ALCANÇADO!' if r2_final >= 0.60 else '❌ FALHOU'}")
    print("=======================================================\n")
    
    return mae_final, r2_final

if __name__ == '__main__':
    run_residual_target_experiment()
