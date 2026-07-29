"""
field_weighing_weight_model.py
--------------------------------
Modelo de Predição de Peso de Abate baseado estritamente nas pesagens amostrais
de campo (extracao_mtech_data) e projeção de trajetória zootécnica (GMD / Gompertz).

Este modelo atua como a BASE PRIMÁRIA de predição antes de receber segundas opiniões
de modelos de Machine Learning (Ensemble / Fallback / Complementaridade).

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-29
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("database/prediction_data.db")
OUTPUT_CSV = Path("data/processed/predictions_field_model.csv")


def normalize_peso_kg(peso_val):
    """Uniformiza a escala de peso para Quilogramas (kg), convertendo gramas (> 10.0) para kg."""
    if pd.isna(peso_val) or peso_val <= 0:
        return np.nan
    if peso_val > 10.0:  # Registrado em gramas (ex: 2350.0 g -> 2.35 kg)
        return peso_val / 1000.0
    return peso_val


def load_data(db_path=DB_PATH):
    """Carrega pesagens de campo e alvos de abate do SQLite."""
    conn = sqlite3.connect(db_path)
    
    # Query de pesagens de campo
    query_mtech = """
    SELECT 
        lote_composto,
        fazenda,
        idade,
        peso AS peso_campo_raw,
        _mortalidade AS taxa_mortalidade_pct,
        _descartados AS taxa_descarte_pct
    FROM extracao_mtech_data
    WHERE idade >= 1 AND idade <= 60 AND peso > 0
    ORDER BY lote_composto, idade ASC
    """
    df_mtech = pd.read_sql_query(query_mtech, conn)
    df_mtech['peso_campo_kg'] = df_mtech['peso_campo_raw'].apply(normalize_peso_kg)
    # Filtro de plausibilidade biológica por idade
    df_mtech = df_mtech[df_mtech['peso_campo_kg'].between(0.03, 6.0)]
    
    # Query do target de abate
    query_abate = """
    SELECT 
        lote_composto,
        fazenda,
        idade_abate,
        data_producao,
        peso_medio_abate_kg,
        peso_abate_g,
        gmd_abate
    FROM peso_abate
    WHERE peso_abate_g BETWEEN 1800 AND 4800
      AND idade_abate BETWEEN 42 AND 60
    """
    df_abate = pd.read_sql_query(query_abate, conn)
    conn.close()
    
    logger.info(f"Carregados {len(df_mtech)} registros de pesagem de campo normalizados e {len(df_abate)} lotes de abate.")
    return df_mtech, df_abate


def extract_field_features(df_mtech, df_abate):
    """
    Extrai as últimas pesagens de campo por lote e calcula a dinâmica de crescimento (GMD).
    """
    # Cruzamento base por lote_composto
    lotes_validos = df_abate.copy()
    
    # Para cada lote, extrair última pesagem (ex: 35 ou 42 dias) e pesagem anterior (ex: 28 dias)
    mtech_grouped = df_mtech.groupby('lote_composto')
    
    records = []
    for idx, row in lotes_validos.iterrows():
        lote = row['lote_composto']
        idade_abate = row['idade_abate']
        peso_abate_g = row['peso_abate_g']
        
        if lote not in mtech_grouped.groups:
            continue
            
        group = mtech_grouped.get_group(lote).sort_values('idade')
        
        # Considerar apenas pesagens anteriores à data/idade de abate
        group_pre = group[group['idade'] < idade_abate]
        if len(group_pre) == 0:
            continue
            
        # Última pesagem disponível
        last_row = group_pre.iloc[-1]
        t_last = last_row['idade']
        w_last = last_row['peso_campo_kg'] # kg
        
        # Penúltima pesagem se existir (para GMD de campo)
        if len(group_pre) >= 2:
            prev_row = group_pre.iloc[-2]
            t_prev = prev_row['idade']
            w_prev = prev_row['peso_campo_kg']
            dt = t_last - t_prev
            gmd_campo_g = ((w_last - w_prev) / dt) * 1000.0 if dt > 0 else 60.0 # g/dia
        else:
            t_prev = np.nan
            w_prev = np.nan
            gmd_campo_g = 62.0 # Valor zootécnico padrão
            
        # Limitar GMD biológico plausível (30g/dia a 100g/dia)
        gmd_campo_g = np.clip(gmd_campo_g, 30.0, 100.0)
        
        # Projeção Linear via GMD de Campo: W_abate = W_last + GMD * (t_abate - t_last)
        dias_restantes = idade_abate - t_last
        pred_peso_gmd_g = (w_last * 1000.0) + (gmd_campo_g * dias_restantes)
        
        # Projeção Curva Gompertz Zootécnica Padrão (A=4800, B=4.3, k=0.051)
        A, B, k = 4800.0, 4.3, 0.051
        ratio_gompertz = np.exp(-B * np.exp(-k * idade_abate)) / np.exp(-B * np.exp(-k * t_last))
        pred_peso_gompertz_g = (w_last * 1000.0) * ratio_gompertz
        
        # Modelo Híbrido Base de Campo (Média Ponderada GMD + Gompertz)
        pred_peso_campo_base_g = 0.65 * pred_peso_gmd_g + 0.35 * pred_peso_gompertz_g
        
        records.append({
            'lote_composto': lote,
            'fazenda': row['fazenda'],
            'idade_abate': idade_abate,
            'peso_abate_g_real': peso_abate_g,
            'ultima_idade_campo': t_last,
            'ultimo_peso_campo_g': w_last * 1000.0,
            'gmd_campo_g': gmd_campo_g,
            'dias_para_abate': dias_restantes,
            'pred_peso_gmd_g': pred_peso_gmd_g,
            'pred_peso_gompertz_g': pred_peso_gompertz_g,
            'pred_peso_campo_base_g': pred_peso_campo_base_g,
            'erro_g': pred_peso_campo_base_g - peso_abate_g,
            'erro_abs_g': abs(pred_peso_campo_base_g - peso_abate_g),
            'erro_pct': ((pred_peso_campo_base_g - peso_abate_g) / peso_abate_g) * 100.0
        })
        
    df_results = pd.DataFrame(records)
    logger.info(f"Processadas feições de campo e projeções para {len(df_results)} lotes.")
    return df_results


def evaluate_model(df_results):
    """Calcula e exibe as métricas globais e por idade de amostragem."""
    y_true = df_results['peso_abate_g_real']
    y_pred = df_results['pred_peso_campo_base_g']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    r2 = r2_score(y_true, y_pred)
    
    acc_3pct = np.mean(df_results['erro_pct'].abs() <= 3.0) * 100.0
    acc_5pct = np.mean(df_results['erro_pct'].abs() <= 5.0) * 100.0
    
    print("\n=======================================================")
    print(" 📊 AVALIAÇÃO DO MODELO BASE DE PESAGENS DE CAMPO (MTECH)")
    print("=======================================================")
    print(f" Total de Lotes Avaliados: {len(df_results):,}")
    print(f" MAE (Erro Médio Absoluto): {mae:.2f} g ({mae/1000.0:.3f} kg)")
    print(f" RMSE (Raiz do Erro Quadrático): {rmse:.2f} g ({rmse/1000.0:.3f} kg)")
    print(f" MAPE (Erro Percentual Médio): {mape:.2f} %")
    print(f" R² (Coeficiente de Determinação): {r2:.4f}")
    print(f" Acurácia dentro de ±3% de Margem: {acc_3pct:.2f} %")
    print(f" Acurácia dentro de ±5% de Margem: {acc_5pct:.2f} %")
    print("-------------------------------------------------------")
    
    # Desempenho por Idade da Última Pesagem no Aviário
    print("\n 📍 MAE POR IDADE DA ÚLTIMA PESAGEM DE CAMPO:")
    age_groups = df_results.groupby('ultima_idade_campo').agg(
        qtd_lotes=('lote_composto', 'count'),
        mae_g=('erro_abs_g', 'mean'),
        mape_pct=('erro_pct', lambda x: np.mean(np.abs(x)))
    ).reset_index()
    
    for idx, row in age_groups[age_groups['qtd_lotes'] >= 10].sort_values('ultima_idade_campo').iterrows():
        print(f"  - Última amostragem no dia {int(row['ultima_idade_campo']):02d}: MAE = {row['mae_g']:.1f} g ({row['mae_g']/1000.0:.3f} kg) | MAPE = {row['mape_pct']:.2f}% ({int(row['qtd_lotes'])} lotes)")
        
    print("=======================================================\n")


def main():
    logger.info("Iniciando construção do Modelo Base de Pesagens de Campo...")
    df_mtech, df_abate = load_data()
    df_results = extract_field_features(df_mtech, df_abate)
    evaluate_model(df_results)
    
    # Salvar predições
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Predições do Modelo Base salvas com sucesso em {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
