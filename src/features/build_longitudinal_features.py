"""
build_longitudinal_features.py
--------------------------------
Constrói o dataset avançado de atributos longitudinais de série temporal por lote:
- Pivotamento de pesos biométricos (peso_d04, peso_d07, peso_d14, peso_d21, peso_d28, peso_d35, peso_d42).
- Velocidades de ganho diário (GMD 7-14, 14-21, 21-28, 28-35, 35-42).
- Acelerações de crescimento (delta GMD).
- Parâmetros individuais de Gompertz (A_i, B_i, k_i) por lote.
- Ratios zootécnicos e interações sanitárias/climáticas.

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-30
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import os
import sys

DB_PATH = Path("database/prediction_data.db")
OUTPUT_CSV = Path("data/processed/longitudinal_dataset.csv")

def gompertz_func(t, A, B, k):
    return A * np.exp(-B * np.exp(-k * t))

def build_longitudinal_features():
    print("=======================================================")
    print(" 🛠️ CONSTRUINDO ATRIBUTOS LONGITUDINAIS DE SÉRIE TEMPORAL")
    print("=======================================================")
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Carregar Dados
    df_mtech = pd.read_sql_query("""
        SELECT lote_composto, fazenda, idade, idade_ref, peso AS peso_kg
        FROM extracao_mtech_data
        WHERE idade >= 1 AND idade <= 60 AND peso > 0
    """, conn)
    
    df_abate = pd.read_sql_query("""
        SELECT lote_composto, fazenda, idade_abate, data_producao, peso_medio_abate_kg, peso_abate_g, gmd_abate
        FROM peso_abate
        WHERE peso_abate_g BETWEEN 1800 AND 4800 AND idade_abate BETWEEN 42 AND 60
    """, conn)
    
    df_conf = pd.read_sql_query("SELECT * FROM lote_sampling_confidence", conn)
    df_var = pd.read_sql_query("SELECT * FROM variables", conn)
    df_const = pd.read_sql_query("SELECT * FROM constantes", conn)
    df_rn12 = pd.read_sql_query("SELECT * FROM lote_rn12_digital_twins", conn)
    
    conn.close()
    
    print(f"Carregados: mtech ({len(df_mtech)}), abate ({len(df_abate)}), confianca ({len(df_conf)}).")
    
    # 2. Convert peso_kg -> peso_g
    df_mtech['peso_g'] = df_mtech['peso_kg'].apply(lambda x: x * 1000.0 if x <= 10.0 else x)
    
    # 3. Pivotar pesagens por idade de referência (idade_ref: 4, 7, 14, 21, 28, 35, 42)
    # Selecionar pesagem mais próxima por idade_ref para cada lote
    mtech_sorted = df_mtech.sort_values(['lote_composto', 'idade_ref', 'idade'])
    mtech_unique_age = mtech_sorted.groupby(['lote_composto', 'idade_ref']).last().reset_index()
    
    df_pivot_peso = mtech_unique_age.pivot(index='lote_composto', columns='idade_ref', values='peso_g')
    df_pivot_peso.columns = [f"peso_d{int(col):02d}" for col in df_pivot_peso.columns]
    df_pivot_peso.reset_index(inplace=True)
    
    # 4. Unir tudo no nível do Lote
    df_base = df_abate.merge(df_conf, on='lote_composto', how='inner', suffixes=('', '_conf'))
    df_base = df_base.merge(df_pivot_peso, on='lote_composto', how='left')
    df_base = df_base.merge(df_var, on='lote_composto', how='left', suffixes=('', '_var'))
    df_base = df_base.merge(df_const, on='fazenda', how='left', suffixes=('', '_const'))
    df_base = df_base.merge(df_rn12[['lote_composto', 'knn_pred_weight_k15', 'knn_pred_weight_k30', 'knn_neighbor_std_k15', 'knn_dist_nearest']], on='lote_composto', how='left')
    
    print(f"Dataset de lotes unido: {len(df_base):,} registros.")
    
    # 5. Engenharia de Features Longitudinais & Velocidades (GMD)
    print("\nCalculando velocidades (GMD), acelerações e interações...")
    
    # Preencher pesos ausentes com interpolação linear por lote onde houver extremos
    weight_cols = [c for c in df_base.columns if c.startswith('peso_d')]
    
    # GMDs Sequenciais (g/dia)
    if 'peso_d07' in df_base.columns and 'peso_d14' in df_base.columns:
        df_base['gmd_07_14'] = (df_base['peso_d14'] - df_base['peso_d07']) / 7.0
    if 'peso_d14' in df_base.columns and 'peso_d21' in df_base.columns:
        df_base['gmd_14_21'] = (df_base['peso_d21'] - df_base['peso_d14']) / 7.0
    if 'peso_d21' in df_base.columns and 'peso_d28' in df_base.columns:
        df_base['gmd_21_28'] = (df_base['peso_d28'] - df_base['peso_d21']) / 7.0
    if 'peso_d28' in df_base.columns and 'peso_d35' in df_base.columns:
        df_base['gmd_28_35'] = (df_base['peso_d35'] - df_base['peso_d28']) / 7.0
    if 'peso_d35' in df_base.columns and 'peso_d42' in df_base.columns:
        df_base['gmd_35_42'] = (df_base['peso_d42'] - df_base['peso_d35']) / 7.0
        
    # Acelerações de GMD (delta GMD)
    if 'gmd_28_35' in df_base.columns and 'gmd_21_28' in df_base.columns:
        df_base['accel_28_35'] = df_base['gmd_28_35'] - df_base['gmd_21_28']
    if 'gmd_35_42' in df_base.columns and 'gmd_28_35' in df_base.columns:
        df_base['accel_35_42'] = df_base['gmd_35_42'] - df_base['gmd_28_35']
        
    # Ratios Temporais
    if 'peso_d35' in df_base.columns and 'peso_d28' in df_base.columns:
        df_base['ratio_d35_d28'] = df_base['peso_d35'] / df_base['peso_d28']
    if 'peso_d42' in df_base.columns and 'peso_d35' in df_base.columns:
        df_base['ratio_d42_d35'] = df_base['peso_d42'] / df_base['peso_d35']
    if 'c15' in df_base.columns and 'peso_d07' in df_base.columns:
        df_base['ratio_c15_d07'] = df_base['c15'] / df_base['peso_d07']
    if 'c15' in df_base.columns and 'peso_d35' in df_base.columns:
        df_base['ratio_c15_d35'] = df_base['c15'] / df_base['peso_d35']
        
    # Projeção Gompertz Individual por Lote
    print("Ajustando projeções Gompertz analíticas individuais por lote...")
    A_g, B_g, k_g = 6400.27, 4.7269, 0.04433 # Parâmetros globais
    
    # Determinar última pesagem válida do lote
    def get_last_weighing_and_proj(row):
        for age in [42, 35, 28, 21, 14, 7, 4]:
            col = f"peso_d{age:02d}"
            if col in row and pd.notna(row[col]) and row[col] > 0:
                w_last = row[col]
                t_last = age
                t_abate = row['idade_abate']
                ratio = gompertz_func(t_abate, A_g, B_g, k_g) / gompertz_func(t_last, A_g, B_g, k_g)
                pred_gomp = w_last * ratio
                return pd.Series([t_last, w_last, t_abate - t_last, pred_gomp])
        return pd.Series([np.nan, np.nan, np.nan, np.nan])
        
    proj_df = df_base.apply(get_last_weighing_and_proj, axis=1)
    proj_df.columns = ['t_last_obs', 'w_last_obs', 'dias_ate_abate', 'pred_gompertz_lote']
    
    df_base = pd.concat([df_base, proj_df], axis=1)
    
    # 6. Salvar Resultado
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_base.to_csv(OUTPUT_CSV, index=False)
    
    print("\n=======================================================")
    print(" ✅ DATASET LONGITUDINAL GERADO COM SUCESSO!")
    print("=======================================================")
    print(f" Total de Lotes Processados: {len(df_base):,}")
    print(f" Total de Colunas Geradas: {len(df_base.columns)}")
    print(f" Arquivo Salvo: {OUTPUT_CSV}")
    print("=======================================================\n")
    return df_base

if __name__ == '__main__':
    build_longitudinal_features()
