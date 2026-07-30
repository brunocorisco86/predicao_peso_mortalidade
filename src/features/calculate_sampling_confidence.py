"""
calculate_sampling_confidence.py
----------------------------------
Calcula os critérios de elegibilidade (RN-09) e o Índice de Confiança de Amostragem (RN-10)
para cada LoteComposto e Fazenda no banco de dados SQLite.

RN-09:
  - Mínimo de 3 pesagens de campo durante o ciclo.
  - Pesagem amostral aos 35 dias (janela 33 a 37d) é OBRIGATÓRIA.
  - Pesagem aos 42 dias é desejável (janela 40 a 44d).

RN-10:
  - Nível de confiança do Lote/Fazenda (0.0 a 10.0).

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-29
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("database/prediction_data.db")
OUTPUT_CSV = Path("data/processed/lote_sampling_confidence.csv")


def calculate_confidence():
    conn = sqlite3.connect(DB_PATH)
    
    # Query de pesagens de campo limpas
    df_mtech = pd.read_sql_query("""
    SELECT lote_composto, fazenda, idade, peso
    FROM extracao_mtech_data
    WHERE idade >= 1 AND idade <= 60 AND peso > 0
    """, conn)
    
    logger.info(f"Analisando pesagens de campo para {df_mtech['lote_composto'].nunique()} lotes únicos...")
    
    # Agrupar por lote_composto
    records = []
    grouped = df_mtech.groupby('lote_composto')
    
    for lote, group in grouped:
        fazenda = group['fazenda'].iloc[0]
        idades = group['idade'].unique()
        qtd_pesagens = len(idades)
        
        # Presença de marcos críticos
        tem_35d = int(any((idades >= 33) & (idades <= 37)))
        tem_42d = int(any((idades >= 40) & (idades <= 44)))
        tem_28d = int(any((idades >= 26) & (idades <= 30)))
        tem_21d = int(any((idades >= 19) & (idades <= 23)))
        
        # RN-09: Elegibilidade para Modelagem
        # Mínimo 3 pesagens E presença obrigatória do marco dos 35d
        elegivel_modelo = int((qtd_pesagens >= 3) and (tem_35d == 1))
        
        # Categoria de Maturidade de Amostragem (RN-09)
        if tem_35d == 1 and tem_42d == 1 and qtd_pesagens >= 4:
            categoria_amostragem = "Ouro (Preferível: 35d + 42d + Histórico)"
        elif tem_35d == 1 and tem_42d == 1:
            categoria_amostragem = "Prata (Razoável: 35d + 42d)"
        elif tem_35d == 1 and qtd_pesagens >= 3:
            categoria_amostragem = "Bronze (Básico: 35d + 2 Pesagens)"
        else:
            categoria_amostragem = "Inelegível (Sem 35d ou < 3 pesagens)"
            
        # RN-10: Cálculo do Score de Confiança do Lote (0.0 a 10.0)
        # 1. Quantidade de pesagens (máximo 4.5 pts)
        pts_qtd = min(qtd_pesagens, 5) * 0.9
        # 2. Marco obrigatório dos 35d (3.5 pts)
        pts_35d = 3.5 if tem_35d == 1 else 0.0
        # 3. Marco dos 42d (2.0 pts)
        pts_42d = 2.0 if tem_42d == 1 else 0.0
        
        score_confianca_lote = round(pts_qtd + pts_35d + pts_42d, 2)

        # RN-11: Delineamento Amostral Mínimo para Modelagem Preditiva Direta
        # Requisitos: >=3 pesagens, presenca de 35d E score >= 7.5 (Ouro ou Prata)
        motivos = []
        if tem_35d == 0:
            motivos.append("Ausência de pesagem aos 35d")
        if qtd_pesagens < 3:
            motivos.append("Menos de 3 pesagens válidas")
        if score_confianca_lote < 7.5:
            motivos.append(f"Score de confiança insuficiente ({score_confianca_lote:.2f} < 7.5)")
            
        if len(motivos) == 0:
            elegivel_rn11 = 1
            motivo_inelegibilidade = "Nenhum (Lote Conforme RN-11)"
            estrategia_predicao = "Modelo Direto (Gompertz / ML)"
        else:
            elegivel_rn11 = 0
            motivo_inelegibilidade = " | ".join(motivos)
            estrategia_predicao = "Fallback Conservador (Média da Fazenda / Histórico)"
        
        records.append({
            'lote_composto': lote,
            'fazenda': fazenda,
            'qtd_pesagens': qtd_pesagens,
            'tem_pesagem_21d': tem_21d,
            'tem_pesagem_28d': tem_28d,
            'tem_pesagem_35d': tem_35d,
            'tem_pesagem_42d': tem_42d,
            'elegivel_modelo': elegivel_modelo,
            'categoria_amostragem': categoria_amostragem,
            'score_confianca_lote': score_confianca_lote,
            'elegivel_rn11': elegivel_rn11,
            'motivo_inelegibilidade': motivo_inelegibilidade,
            'estrategia_predicao': estrategia_predicao
        })
        
    df_lotes = pd.DataFrame(records)
    
    # Calcular score médio da Fazenda (confiança do aviário)
    score_fazenda = df_lotes.groupby('fazenda')['score_confianca_lote'].mean().round(2).reset_index()
    score_fazenda.rename(columns={'score_confianca_lote': 'score_confianca_fazenda'}, inplace=True)
    
    # Merge no dataframe final
    df_lotes = df_lotes.merge(score_fazenda, on='fazenda', how='left')
    
    # Salvar resultado em CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_lotes.to_csv(OUTPUT_CSV, index=False)
    
    # Atualizar / criar tabela no SQLite database
    df_lotes.to_sql('lote_sampling_confidence', conn, if_exists='replace', index=False)
    conn.close()
    
    # Estatísticas de Elegibilidade RN-11
    tot_lotes = len(df_lotes)
    tot_rn11 = df_lotes['elegivel_rn11'].sum()
    pct_rn11 = (tot_rn11 / tot_lotes) * 100.0
    
    logger.info("=======================================================")
    logger.info(" 🎯 RESULTADOS DA APLICAÇÃO DA RN-11 (DELINEAMENTO MÍNIMO)")
    logger.info("=======================================================")
    logger.info(f" Total de Lotes Analisados: {tot_lotes:,}")
    logger.info(f" Lotes Elegíveis para Modelo Direto (RN-11): {tot_rn11:,} ({pct_rn11:.2f}%)")
    logger.info(f" Lotes Direcionados para Fallback (RN-11): {tot_lotes - tot_rn11:,} ({100-pct_rn11:.2f}%)")
    logger.info("-------------------------------------------------------")
    logger.info(" Distribuição por Estratégia de Predição:")
    for est, count in df_lotes['estrategia_predicao'].value_counts().items():
        logger.info(f"  - {est}: {count:,} lotes ({count/tot_lotes*100:.1f}%)")
    logger.info("=======================================================")
    
    return df_lotes


if __name__ == '__main__':
    calculate_confidence()
