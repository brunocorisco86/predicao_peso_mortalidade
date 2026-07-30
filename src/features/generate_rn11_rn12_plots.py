"""
generate_rn11_rn12_plots.py
----------------------------
Gera o Dashboard Visual Integrado da RN-11 (Delineamento Mínimo) e RN-12 (Gêmeos Digitais KNN).
Salva os gráficos em alta resolução na pasta plots/.

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-30
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurações de Estilo Gráfico Premium
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

DB_PATH = Path("database/prediction_data.db")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_dashboard():
    print("Gerando Dashboard Visual da RN-11 e RN-12...")
    conn = sqlite3.connect(DB_PATH)
    
    # Query Integrada de Lotes, Abate, Confiança (RN-11) e Gêmeos Digitais (RN-12)
    query = """
    SELECT 
        sc.lote_composto,
        sc.fazenda,
        sc.qtd_pesagens,
        sc.categoria_amostragem,
        sc.score_confianca_lote,
        sc.elegivel_rn11,
        sc.estrategia_predicao,
        pa.peso_abate_g AS peso_real_g,
        rn.knn_pred_weight_k15,
        rn.knn_pred_weight_k30,
        rn.knn_neighbor_std_k15,
        rn.knn_dist_nearest
    FROM lote_sampling_confidence sc
    INNER JOIN peso_abate pa ON sc.lote_composto = pa.lote_composto
    INNER JOIN lote_rn12_digital_twins rn ON sc.lote_composto = rn.lote_composto
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Dados carregados para o dashboard: {len(df):,} lotes.")
    
    # Criar Figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ---------------------------------------------------------
    # Gráfico 1: Peso Predito por Gêmeos Digitais (K=15) vs Peso Real de Abate
    # ---------------------------------------------------------
    sns.scatterplot(
        data=df.sample(min(3000, len(df)), random_state=42),
        x='peso_real_g', y='knn_pred_weight_k15',
        hue='categoria_amostragem', alpha=0.5, ax=axes[0, 0], s=30, palette='Set2'
    )
    # Linha de 45 graus (Identidade Perfeita)
    min_val = min(df['peso_real_g'].min(), df['knn_pred_weight_k15'].min())
    max_val = max(df['peso_real_g'].max(), df['knn_pred_weight_k15'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Concordância Perfeita (1:1)')
    axes[0, 0].set_title('1. Gêmeos Digitais (K=15) vs Peso Real no Abate', fontweight='bold')
    axes[0, 0].set_xlabel('Peso Real de Abate no Frigorífico (g)')
    axes[0, 0].set_ylabel('Peso Estimado por Gêmeos Digitais (g)')
    axes[0, 0].legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    
    # ---------------------------------------------------------
    # Gráfico 2: Estabilidade/Variabilidade dos Gêmeos por Categoria RN-11
    # ---------------------------------------------------------
    sns.boxplot(
        data=df, x='categoria_amostragem', y='knn_neighbor_std_k15',
        ax=axes[0, 1], palette='Blues_r'
    )
    axes[0, 1].set_title('2. Variabilidade Interna dos Gêmeos (Std Dev em Gramas)', fontweight='bold')
    axes[0, 1].set_xlabel('Categoria de Maturidade Amostral (RN-09/RN-11)')
    axes[0, 1].set_ylabel('Desvio Padrão dos Gêmeos (g)')
    axes[0, 1].tick_params(axis='x', rotation=15)
    
    # ---------------------------------------------------------
    # Gráfico 3: Distribuição da Distância ao Gêmeo Mais Próximo
    # ---------------------------------------------------------
    sns.histplot(df['knn_dist_nearest'], kde=True, ax=axes[1, 0], color='#27ae60', bins=35)
    axes[1, 0].set_title('3. Densidade de Similaridade (Distância ao Gêmeo Mais Próximo)', fontweight='bold')
    axes[1, 0].set_xlabel('Distância Euclidiana Padronizada')
    axes[1, 0].set_ylabel('Quantidade de Lotes')
    
    # ---------------------------------------------------------
    # Gráfico 4: Distribuição por Estratégia de Predição (RN-11)
    # ---------------------------------------------------------
    strat_counts = df['estrategia_predicao'].value_counts()
    colors = ['#2980b9', '#e67e22']
    wedges, texts, autotexts = axes[1, 1].pie(
        strat_counts, labels=strat_counts.index, autopct='%1.1f%%',
        startangle=140, colors=colors, explode=(0.05, 0), textprops=dict(color="black")
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    axes[1, 1].set_title('4. Proporção de Estratégias Preditivas (RN-11)', fontweight='bold')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / "dashboard_rn11_rn12_digital_twins.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Dashboard salvo com sucesso em: {output_path}")

if __name__ == "__main__":
    generate_dashboard()
