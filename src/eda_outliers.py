import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings
from src.utils.logger import logger

def calculate_descriptive_stats(df, columns):
    """Calcule estatística descritiva completa para colunas especificadas."""
    stats_list = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        n_total = len(df)
        n_null = df[col].isnull().sum() + (len(df[col]) - len(series))
        pct_null = (n_null / n_total) * 100

        if len(series) == 0:
            continue

        mean = series.mean()
        std = series.std()
        median = series.median()
        min_val = series.min()
        max_val = series.max()
        p25 = series.quantile(0.25)
        p75 = series.quantile(0.75)
        iqr = p75 - p25
        skew = series.skew()
        kurt = series.kurtosis()

        stats_list.append({
            'Variavel': col,
            'N': len(series),
            'Nulos (%)': round(pct_null, 2),
            'Media': round(mean, 4),
            'Mediana': round(median, 4),
            'Desv_Pad': round(std, 4),
            'Min': round(min_val, 4),
            'Max': round(max_val, 4),
            'P25': round(p25, 4),
            'P75': round(p75, 4),
            'IQR': round(iqr, 4),
            'Assimetria': round(skew, 4),
            'Curtose': round(kurt, 4)
        })
    return pd.DataFrame(stats_list)

def main():
    logger.info("Iniciando processo de EDA, tratamento de outliers e geração de gráficos...")

    # Direct paths
    raw_csv_path = settings.UNIFIED_CSV_PATH
    plots_dir = os.path.join(settings.BASE_DIR, 'plots')
    processed_dir = settings.PROCESSED_DIR
    cleaned_csv_path = os.path.join(processed_dir, 'cleaned_data.csv')

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Carregar Dados
    logger.info(f"Lendo dados de {raw_csv_path}")
    df_raw = pd.read_csv(raw_csv_path, low_memory=False)
    initial_rows = len(df_raw)
    logger.info(f"Linhas carregadas: {initial_rows}")

    # Definir colunas principais
    key_numeric_cols = [
        'peso', 'idade', 'mortalidade', '_mortalidade', 'descartados', '_descartados',
        '_mortalidade__descartados', 'cab_alojadas', 'estoque_aves'
    ]

    # Calcular estatística descritiva bruta
    df_stats_raw = calculate_descriptive_stats(df_raw, key_numeric_cols)
    logger.info("\n--- Estatística Descritiva Bruta ---\n" + df_stats_raw.to_string())

    # 2. Tratamento e Correções
    df = df_raw.copy()

    # Ajuste de escala do peso (valores entre 10 e 5000 foram digitados em gramas; convertendo para kg)
    peso_g_count = ((df['peso'] > 10) & (df['peso'] <= 5000)).sum()
    df['peso_kg'] = df['peso'].apply(lambda x: x/1000.0 if (10 < x <= 5000) else x)
    df['peso_g'] = df['peso_kg'] * 1000.0  # coluna em gramas para conveniência

    # Criar coluna de semana de alojamento
    df['semana'] = np.ceil(df['idade'] / 7.0).astype(float)
    df['taxa_mortalidade_pct'] = (df['mortalidade'] / df['cab_alojadas']) * 100.0
    df['taxa_descartados_pct'] = (df['descartados'] / df['cab_alojadas']) * 100.0

    # 3. Aplicar Filtros Biológicos e Estatísticos
    # Filter 1: Idade entre 1 e 60 dias
    mask_age = (df['idade'] >= 1) & (df['idade'] <= 60)
    removed_age = (~mask_age).sum()

    # Filter 2: Cabeças alojadas > 0
    mask_cab = df['cab_alojadas'] > 0
    removed_cab = (~mask_cab & mask_age).sum()

    # Filter 3: Mortalidade plausível (mortalidade >= 0, mortalidade <= cab_alojadas, taxa_mortalidade <= 50%)
    mask_mort = (df['mortalidade'] >= 0) & (df['mortalidade'] <= df['cab_alojadas']) & (df['taxa_mortalidade_pct'] <= 50.0)
    removed_mort = (~mask_mort & mask_age & mask_cab).sum()

    # Filter 4: Peso positivo e plausível (peso_kg > 0.020 e <= 5.0kg, isto é 20g a 5000g)
    mask_peso_basic = (df['peso_kg'] > 0.020) & (df['peso_kg'] <= 5.0)
    removed_peso_basic = (~mask_peso_basic & mask_age & mask_cab & mask_mort).sum()

    df_filtered_base = df[mask_age & mask_cab & mask_mort & mask_peso_basic].copy()

    # Filter 5: IQR Outliers em Peso por faixa de Idade (Remoção de discrepâncias extremas por idade)
    outlier_indices = []
    for id_val, grp in df_filtered_base.groupby('idade'):
        if len(grp) < 10:
            continue
        q1 = grp['peso_kg'].quantile(0.25)
        q3 = grp['peso_kg'].quantile(0.75)
        iqr = q3 - q1
        # Usando 3.0 * IQR para descarte de outliers biológicos/operacionais extremos sem mutilar a variação natural
        lower_b = q1 - 3.0 * iqr
        upper_b = q3 + 3.0 * iqr
        out_idx = grp[(grp['peso_kg'] < lower_b) | (grp['peso_kg'] > upper_b)].index
        outlier_indices.extend(out_idx)

    removed_peso_iqr = len(outlier_indices)
    df_cleaned = df_filtered_base.drop(index=outlier_indices).copy()

    # Padronizar coluna peso principal em cleaned_data
    df_cleaned['peso'] = df_cleaned['peso_kg']

    final_rows = len(df_cleaned)
    total_removed = initial_rows - final_rows
    logger.info(f"Processo de filtragem concluído: {initial_rows} -> {final_rows} linhas (Removidos total: {total_removed})")

    # Save cleaned dataset
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    logger.info(f"Dataset limpo salvo em: {cleaned_csv_path}")

    # Estatística Descritiva Pós-Tratamento
    df_stats_cleaned = calculate_descriptive_stats(df_cleaned, key_numeric_cols + ['peso_kg', 'peso_g', 'taxa_mortalidade_pct'])
    logger.info("\n--- Estatística Descritiva Limpa ---\n" + df_stats_cleaned.to_string())

    # Configuração estética do Seaborn / Matplotlib
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300
    })

    # -------------------------------------------------------------
    # Gráfico 1: plots/distribuicao_peso_por_idade.png
    # Dispersão / Curva Mediana de Peso por Idade com Faixa de Percentis
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    # Agrupar estatísticas de peso por idade em dias
    age_stats = df_cleaned.groupby('idade')['peso_g'].agg(
        median='median',
        p10=lambda x: np.percentile(x, 10),
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75),
        p90=lambda x: np.percentile(x, 90)
    ).reset_index()

    # Scatter plot de uma amostragem dos pontos para clareza visual
    sample_points = df_cleaned.sample(n=min(10000, len(df_cleaned)), random_state=42)
    ax.scatter(sample_points['idade'], sample_points['peso_g'], alpha=0.15, color='#4c72b0', s=12, label='Amostra de Lotes (Medições)')

    # Curvas de Mediana e Percentis
    ax.plot(age_stats['idade'], age_stats['median'], color='#c44e52', linewidth=3, label='Mediana de Peso (g)', marker='o', markersize=5)
    ax.fill_between(age_stats['idade'], age_stats['p25'], age_stats['p75'], color='#c44e52', alpha=0.25, label='Intervalo Interquartil (P25 - P75)')
    ax.fill_between(age_stats['idade'], age_stats['p10'], age_stats['p90'], color='#c44e52', alpha=0.12, label='Faixa P10 - P90')

    ax.set_title('Evolução e Distribuição do Peso Corporal por Idade (Dias)', pad=15, fontweight='bold')
    ax.set_xlabel('Idade das Aves (Dias)')
    ax.set_ylabel('Peso Corporal (Gramas)')
    ax.legend(loc='upper left', frameon=True)
    ax.set_xlim(0, 56)
    plt.tight_layout()

    g1_path = os.path.join(plots_dir, 'distribuicao_peso_por_idade.png')
    plt.savefig(g1_path, dpi=300)
    plt.close()
    logger.info(f"Gráfico 1 salvo em: {g1_path}")

    # -------------------------------------------------------------
    # Gráfico 2: plots/boxplots_outliers_peso.png
    # Boxplot de peso Antes e Depois por faixa de idade (Semanas 1 a 6)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Preparar datasets com indicação de semana
    df_raw_valid_age = df_raw[(df_raw['idade'] >= 1) & (df_raw['idade'] <= 60) & (df_raw['peso'] > 0)].copy()
    # Para o raw, converter a escala de g para kg apenas para comparabilidade
    df_raw_valid_age['peso_kg'] = df_raw_valid_age['peso'].apply(lambda x: x/1000.0 if (10 < x <= 5000) else x)
    df_raw_valid_age['semana'] = np.ceil(df_raw_valid_age['idade'] / 7.0).astype(int)

    # Filtrar semanas 1 a 6
    df_raw_semanas = df_raw_valid_age[df_raw_valid_age['semana'].between(1, 6)]
    df_clean_semanas = df_cleaned[df_cleaned['semana'].between(1, 6)]

    sns.boxplot(data=df_raw_semanas, x='semana', y='peso_kg', ax=axes[0], color='#8c8c8c', flierprops=dict(marker='o', markersize=3, alpha=0.4))
    axes[0].set_title('Antes do Tratamento de Outliers (Bruto)', fontweight='bold')
    axes[0].set_xlabel('Semana de Alojamento')
    axes[0].set_ylabel('Peso corporal (kg)')

    sns.boxplot(data=df_clean_semanas, x='semana', y='peso_kg', ax=axes[1], color='#55a868', flierprops=dict(marker='o', markersize=3, alpha=0.3))
    axes[1].set_title('Após Tratamento de Outliers (Limpo)', fontweight='bold')
    axes[1].set_xlabel('Semana de Alojamento')
    axes[1].set_ylabel('')

    plt.suptitle('Comparativo de Boxplots do Peso Corporal por Semana de Alojamento', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    g2_path = os.path.join(plots_dir, 'boxplots_outliers_peso.png')
    plt.savefig(g2_path, dpi=300)
    plt.close()
    logger.info(f"Gráfico 2 salvo em: {g2_path}")

    # -------------------------------------------------------------
    # Gráfico 3: plots/distribuicao_mortalidade.png
    # Histograma e Boxplot de mortalidade (cabeças e taxa %)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Subplot 1: Histograma de Mortalidade Absoluta (cabeças)
    sns.histplot(df_cleaned['mortalidade'], kde=True, ax=axes[0, 0], color='#4c72b0', bins=40)
    axes[0, 0].set_title('Distribuição da Mortalidade Absoluta (Cabeças)', fontweight='bold')
    axes[0, 0].set_xlabel('Nº de Aves Mortas (Cabeças)')
    axes[0, 0].set_ylabel('Frequência')

    # Subplot 2: Boxplot de Mortalidade Absoluta
    sns.boxplot(x=df_cleaned['mortalidade'], ax=axes[0, 1], color='#4c72b0')
    axes[0, 1].set_title('Boxplot da Mortalidade Absoluta (Cabeças)', fontweight='bold')
    axes[0, 1].set_xlabel('Nº de Aves Mortas (Cabeças)')

    # Subplot 3: Histograma da Taxa de Mortalidade (%)
    sns.histplot(df_cleaned['taxa_mortalidade_pct'], kde=True, ax=axes[1, 0], color='#dd8452', bins=40)
    axes[1, 0].set_title('Distribuição da Taxa de Mortalidade (%)', fontweight='bold')
    axes[1, 0].set_xlabel('Taxa de Mortalidade (%)')
    axes[1, 0].set_ylabel('Frequência')

    # Subplot 4: Boxplot da Taxa de Mortalidade (%)
    sns.boxplot(x=df_cleaned['taxa_mortalidade_pct'], ax=axes[1, 1], color='#dd8452')
    axes[1, 1].set_title('Boxplot da Taxa de Mortalidade (%)', fontweight='bold')
    axes[1, 1].set_xlabel('Taxa de Mortalidade (%)')

    plt.suptitle('Análise de Distribuição e Dispersão da Mortalidade Avícola', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()

    g3_path = os.path.join(plots_dir, 'distribuicao_mortalidade.png')
    plt.savefig(g3_path, dpi=300)
    plt.close()
    logger.info(f"Gráfico 3 salvo em: {g3_path}")

    # -------------------------------------------------------------
    # Gráfico 4: plots/matriz_correlacao_features.png
    # Heatmap das principais correlações de variáveis numéricas
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))

    # Selecionar variáveis numéricas relevantes
    corr_cols = [
        'peso', 'idade', 'mortalidade', 'taxa_mortalidade_pct',
        'descartados', 'taxa_descartados_pct', 'cab_alojadas', 'estoque_aves',
        'f01', 'f02', 'f03', 'a01', 'a02', 'd01', 'h01'
    ]
    # Filtrar apenas as colunas existentes
    existing_corr_cols = [c for c in corr_cols if c in df_cleaned.columns]

    corr_matrix = df_cleaned[existing_corr_cols].corr(method='spearman')

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title('Matriz de Correlação de Spearman entre Variáveis Zootécnicas', pad=15, fontweight='bold')
    plt.tight_layout()

    g4_path = os.path.join(plots_dir, 'matriz_correlacao_features.png')
    plt.savefig(g4_path, dpi=300)
    plt.close()
    logger.info(f"Gráfico 4 salvo em: {g4_path}")

    logger.info("Execução finalizada com sucesso!")

if __name__ == '__main__':
    main()
