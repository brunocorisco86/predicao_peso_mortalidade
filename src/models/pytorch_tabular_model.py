"""
pytorch_tabular_model.py
-------------------------
Rede Neural Profunda Tabular (MLP / ResNet Tabular) em PyTorch com suporte a GPU CUDA.
Treina modelo de aprendizado profundo nos atributos longitudinais e gera predições Out-Of-Fold (OOF).

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-30
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Configuração de Dispositivo (GPU CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.2):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        out = self.head(x)
        return out

def run_pytorch_model(df_long=None):
    print("=======================================================")
    print(f" 🧠 TREINANDO REDE NEURAL TABULAR PYTORCH (DEVICE: {device})")
    print("=======================================================")
    
    if df_long is None:
        df_long = pd.read_csv("data/processed/longitudinal_dataset.csv", low_memory=False)
        
    if 'elegivel_rn11' in df_long.columns:
        df_long = df_long[df_long['elegivel_rn11'] == 1.0].copy()
        
    df_long = df_long.dropna(subset=['peso_abate_g', 'w_last_obs', 'idade_abate'])
    
    target = 'peso_abate_g'
    group_col = 'lote_composto'
    
    exclude = ['data_alojamento', 'nome_fazenda', 'data_hora_transao', 'lote_composto', 
               'data_evento', 'data_criao', 'id_usurio_criao', 'extensionista', 'id_usurio', 
               'fazenda', 'produtor', 'data_producao_abate', 'peso_medio_abate_kg', 'peso_abate_g', 
               'gmd_abate', 'score_confianca_lote', 'categoria_amostragem', 'elegivel_rn11', 
               'motivo_inelegibilidade', 'estrategia_predicao', 'nucleo']
               
    features = [c for c in df_long.columns if c not in exclude and df_long[c].dtype in [np.float64, np.int64]]
    
    X = df_long[features].fillna(df_long[features].median())
    y = df_long[target].values
    groups = df_long[group_col].values
    
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(df_long))
    
    epochs = 40
    batch_size = 256
    
    maes, rmses, r2s = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        train_dataset = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = TabularResNet(input_dim=len(features)).to(device)
        criterion = nn.L1Loss() # MAE Loss
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    val_loss += criterion(out, by).item() * len(by)
            val_loss /= len(val_dataset)
            scheduler.step(val_loss)
            
        # Predict on validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                out = model(bx)
                val_preds_list.append(out.cpu().numpy())
        fold_preds = np.vstack(val_preds_list).flatten()
        oof_preds[val_idx] = fold_preds
        
        mae = mean_absolute_error(y_val, fold_preds)
        rmse = np.sqrt(mean_squared_error(y_val, fold_preds))
        r2 = r2_score(y_val, fold_preds)
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        print(f" Fold {fold} PyTorch: MAE = {mae:.2f}g | RMSE = {rmse:.2f}g | R² = {r2:.4f}")
        
    mean_mae = np.mean(maes)
    mean_rmse = np.mean(rmses)
    mean_r2 = np.mean(r2s)
    
    print("-------------------------------------------------------")
    print(f" PyTorch Tabular ResNet -> MAE: {mean_mae:.2f}g | RMSE: {mean_rmse:.2f}g | R²: {mean_r2:.4f}")
    print("=======================================================\n")
    
    return oof_preds, mean_mae, mean_rmse, mean_r2

if __name__ == '__main__':
    run_pytorch_model()
