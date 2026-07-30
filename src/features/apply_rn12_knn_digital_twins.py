"""
apply_rn12_knn_digital_twins.py
---------------------------------
Aplica a Regra de Negócio RN-12 na base de dados SQLite (prediction_data.db):
1. Imputação contextual de valores nulos nas tabelas de dimensão/manejo (variables e constantes) via KNNImputer.
2. Extração das features de Gêmeos Digitais (K=15 e K=30) via NearestNeighbors Out-Of-Fold (GroupKFold por lote_composto).
3. Preservação estrita das pesagens reais de campo (Zero contaminação sintética em biometrias).
4. Persistência na tabela SQLite 'lote_rn12_digital_twins' e exportação para CSV.

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-30
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.model_selection import GroupKFold
import os
import sys

# Configuração de Caminhos
DB_PATH = Path("database/prediction_data.db")
OUTPUT_CSV = Path("data/processed/lote_rn12_digital_twins.csv")

def apply_rn12_to_database():
    print("=======================================================")
    print(" 🚀 APLICANDO RN-12: GÊMEOS DIGITAIS E IMPUTAÇÃO KNN NO DB")
    print("=======================================================")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Carregar Tabelas do Banco SQLite
    df_mtech = pd.read_sql_query("SELECT * FROM extracao_mtech_data", conn)
    df_var = pd.read_sql_query("SELECT * FROM variables", conn)
    df_const = pd.read_sql_query("SELECT * FROM constantes", conn)
    df_conf = pd.read_sql_query("SELECT * FROM lote_sampling_confidence", conn)
    df_abate = pd.read_sql_query("SELECT * FROM peso_abate", conn)
    
    print(f"Carregados do SQLite: mtech ({len(df_mtech)}), variables ({len(df_var)}), constantes ({len(df_const)}), confianca ({len(df_conf)}), abate ({len(df_abate)}).")
    
    # 2. Pilar 1: Imputação KNN de Variáveis de Infraestrutura e Manejo (variables & constantes)
    print("\n📌 Pilar 1: Imputando valores nulos residuais em variables e constantes via KNNImputer...")
    
    # Selecionar colunas numéricas de variables
    num_cols_var = df_var.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols_var) > 0 and df_var[num_cols_var].isnull().sum().sum() > 0:
        imputer_var = KNNImputer(n_neighbors=5, weights='distance')
        df_var[num_cols_var] = imputer_var.fit_transform(df_var[num_cols_var])
        print(f"  - Variables: Nulos imputados com sucesso nas colunas numéricas.")
    else:
        print("  - Variables: Sem nulos numéricos pendentes.")
        
    # Selecionar colunas numéricas de constantes
    num_cols_const = df_const.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols_const) > 0 and df_const[num_cols_const].isnull().sum().sum() > 0:
        imputer_const = KNNImputer(n_neighbors=5, weights='distance')
        df_const[num_cols_const] = imputer_const.fit_transform(df_const[num_cols_const])
        print(f"  - Constantes: Nulos imputados com sucesso nas colunas numéricas (ex: i10, i11).")
    else:
        print("  - Constantes: Sem nulos numéricos pendentes.")
        
    # Atualizar tabelas imputadas no SQLite
    df_var.to_sql('variables', conn, if_exists='replace', index=False)
    df_const.to_sql('constantes', conn, if_exists='replace', index=False)
    print("  - Tabelas 'variables' e 'constantes' atualizadas e salvas no SQLite.")
    
    # 3. Pilar 3: Extração de Features de Gêmeos Digitais (KNN Matching)
    print("\n📌 Pilar 3: Extraindo Features de Gêmeos Digitais (K=15 e K=30) por Lote...")
    
    # Montar dataset unificado no nível do lote
    df_lote_base = df_conf.merge(df_abate[['lote_composto', 'idade_abate', 'peso_abate_g']], on='lote_composto', how='inner')
    if 'fazenda' not in df_lote_base.columns:
        df_lote_base['fazenda'] = df_lote_base['lote_composto'].apply(lambda x: int(str(x).split('-')[0]) if '-' in str(x) else None)
        
    df_lote_base = df_lote_base.merge(df_var, on='lote_composto', how='left', suffixes=('', '_var'))
    df_lote_base = df_lote_base.merge(df_const, on='fazenda', how='left', suffixes=('', '_const'))
    
    # Tratar colunas categóricas e numéricas para distância
    feature_cols = [
        'qtd_pesagens', 'score_confianca_lote',
        'c05', 'c06', 'c15', 'f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f15',
        'a01', 'a02', 'a03', 'a08', 'a09', 'x02'
    ]
    avail_cols = [c for c in feature_cols if c in df_lote_base.columns]
    
    # Preencher nulos das features explicativas com a mediana para cálculo de distância KNN
    X_mat = df_lote_base[avail_cols].copy()
    for col in X_mat.columns:
        X_mat[col] = pd.to_numeric(X_mat[col], errors='coerce')
        X_mat[col] = X_mat[col].fillna(X_mat[col].median())
        
    y_vec = df_lote_base['peso_abate_g'].values
    groups = df_lote_base['lote_composto'].values
    
    # Out-Of-Fold KNN Neighborhood Calculation (5-Fold GroupKFold)
    gkf = GroupKFold(n_splits=5)
    
    knn_pred_k15 = np.zeros(len(df_lote_base))
    knn_pred_k30 = np.zeros(len(df_lote_base))
    knn_std_k15 = np.zeros(len(df_lote_base))
    knn_dist_nearest = np.zeros(len(df_lote_base))
    
    scaler = StandardScaler()
    
    for train_idx, test_idx in gkf.split(X_mat, y_vec, groups):
        X_tr, X_te = X_mat.iloc[train_idx], X_mat.iloc[test_idx]
        y_tr, y_te = y_vec[train_idx], y_vec[test_idx]
        
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        # Fit KNeighborsRegressor K=15
        knn_k15 = KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1)
        knn_k15.fit(X_tr_scaled, y_tr)
        knn_pred_k15[test_idx] = knn_k15.predict(X_te_scaled)
        
        # Fit KNeighborsRegressor K=30
        knn_k30 = KNeighborsRegressor(n_neighbors=30, weights='distance', n_jobs=-1)
        knn_k30.fit(X_tr_scaled, y_tr)
        knn_pred_k30[test_idx] = knn_k30.predict(X_te_scaled)
        
        # NearestNeighbors para estatísticas de vizinhança
        nn = NearestNeighbors(n_neighbors=15, n_jobs=-1)
        nn.fit(X_tr_scaled)
        distances, indices = nn.kneighbors(X_te_scaled)
        
        knn_dist_nearest[test_idx] = distances[:, 0]
        for i, row_indices in enumerate(indices):
            knn_std_k15[test_idx[i]] = np.std(y_tr[row_indices])
            
    df_lote_base['knn_pred_weight_k15'] = np.round(knn_pred_k15, 2)
    df_lote_base['knn_pred_weight_k30'] = np.round(knn_pred_k30, 2)
    df_lote_base['knn_neighbor_std_k15'] = np.round(knn_std_k15, 2)
    df_lote_base['knn_dist_nearest'] = np.round(knn_dist_nearest, 4)
    
    # Criar Tabela de Gêmeos Digitais da RN-12
    df_rn12_result = df_lote_base[[
        'lote_composto', 'fazenda', 'score_confianca_lote', 'categoria_amostragem', 'elegivel_rn11',
        'knn_pred_weight_k15', 'knn_pred_weight_k30', 'knn_neighbor_std_k15', 'knn_dist_nearest'
    ]].copy()
    
    # 4. Salvar Tabela no SQLite e Exportar CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_rn12_result.to_csv(OUTPUT_CSV, index=False)
    
    df_rn12_result.to_sql('lote_rn12_digital_twins', conn, if_exists='replace', index=False)
    conn.close()
    
    print("\n=======================================================")
    print(" ✅ RN-12 APLICADA COM SUCESSO NO BANCO DE DADOS SQLITE!")
    print("=======================================================")
    print(f" Total de Lotes com Gêmeos Digitais Gerados: {len(df_rn12_result):,}")
    print(f" Tabela Criada no SQLite: 'lote_rn12_digital_twins'")
    print(f" Arquivo CSV Exportado: {OUTPUT_CSV}")
    print(" Sample de 3 Lotes com Gêmeos Digitais:")
    print(df_rn12_result[['lote_composto', 'knn_pred_weight_k15', 'knn_neighbor_std_k15', 'knn_dist_nearest']].head(3).to_string(index=False))
    print("=======================================================\n")
    return df_rn12_result

if __name__ == "__main__":
    apply_rn12_to_database()
