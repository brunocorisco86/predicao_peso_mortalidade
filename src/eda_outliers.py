# src/eda_outliers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import logger, setup_logging
from config.settings import settings

def run_eda_and_clean():
    logger.info("Starting EDA and Outlier Cleaning process for SLAUGHTER WEIGHT DATASET...")

    input_csv = settings.UNIFIED_CSV_PATH
    cleaned_csv = os.path.join(settings.PROCESSED_DIR, 'cleaned_data.csv')
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    logger.info(f"Loaded unified_data.csv with {len(df)} rows and {len(df.columns)} columns.")

    # 1. Prioritize Ground Truth Slaughter Weight (export_peso_abate_2023_2026.xlsx)
    has_abate = df['peso_abate_g'].notnull() & (df['peso_abate_g'] > 0)
    df['peso_kg'] = np.where(has_abate, df['peso_medio_abate_kg'], pd.to_numeric(df['peso'], errors='coerce'))
    df['peso_g'] = np.where(has_abate, df['peso_abate_g'], df['peso_kg'] * 1000.0)
    df['idade'] = np.where(has_abate, df['idade_abate'], pd.to_numeric(df['idade'], errors='coerce'))
    df['gmd_abate'] = pd.to_numeric(df['gmd_abate'], errors='coerce')

    df['mortalidade'] = pd.to_numeric(df['mortalidade'], errors='coerce')
    df['descartados'] = pd.to_numeric(df['descartados'], errors='coerce')
    df['cab_alojadas'] = pd.to_numeric(df['cab_alojadas'], errors='coerce')

    initial_len = len(df)

    # 2. Filter Biological Ranges according to RN-01 and RN-02 for Slaughter Batches
    # RN-01: Idade de Abate entre 42 e 60 dias
    # RN-02: Peso de Abate entre 1.80 kg e 4.80 kg (1800g a 4800g)
    # RN-03: cab_alojadas > 0
    mask = (
        (df['idade'] >= 42) & (df['idade'] <= 60) &
        (df['peso_kg'].notnull()) & (df['peso_kg'] >= 1.80) & (df['peso_kg'] <= 4.80) &
        (df['cab_alojadas'] > 0)
    )
    df_clean = df[mask].copy()

    # Deduplicate by lote_composto to guarantee unique slaughter flock records
    df_clean = df_clean.drop_duplicates(subset=['lote_composto'], keep='first').copy()

    # 3. IQR Outlier Filter per Slaughter Age (RN-05)
    outlier_indices = set()
    for age, group in df_clean.groupby('idade'):
        if len(group) >= 20:
            q1 = group['peso_kg'].quantile(0.25)
            q3 = group['peso_kg'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = max(1.50, q1 - 3.0 * iqr)
            upper_bound = min(5.00, q3 + 3.0 * iqr)
            outliers = group[(group['peso_kg'] < lower_bound) | (group['peso_kg'] > upper_bound)].index
            outlier_indices.update(outliers)

    df_clean = df_clean.drop(index=list(outlier_indices)).reset_index(drop=True)
    final_len = len(df_clean)
    logger.info(f"Slaughter age dataset filtered: kept {final_len} unique slaughter flock records.")

    # Save cleaned dataset
    df_clean.to_csv(cleaned_csv, index=False)
    logger.info(f"Cleaned slaughter dataset saved to {cleaned_csv}")

    # 4. Descriptive Statistics Summary for Slaughter
    stats_cols = ['idade', 'peso_kg', 'peso_g', 'gmd_abate', 'mortalidade', 'descartados', 'cab_alojadas', '_mortalidade', '_descartados']
    existing_stats_cols = [c for c in stats_cols if c in df_clean.columns]
    desc_stats = df_clean[existing_stats_cols].describe().T
    desc_stats['median'] = df_clean[existing_stats_cols].median()
    desc_stats['skewness'] = df_clean[existing_stats_cols].skew()
    desc_stats['kurtosis'] = df_clean[existing_stats_cols].kurtosis()
    desc_stats['null_pct'] = (df_clean[existing_stats_cols].isnull().sum() / len(df_clean)) * 100

    desc_stats_path = os.path.join(settings.PROCESSED_DIR, 'descriptive_statistics.csv')
    desc_stats.to_csv(desc_stats_path)
    logger.info(f"Descriptive statistics saved to {desc_stats_path}")

    # 5. Visualization Plots for Slaughter Age
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Weight Distribution at Slaughter Ages (42 to 60 days)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_clean, x='idade', y='peso_kg', ax=ax, palette='YlOrRd')
    ax.set_title("Distribuição do Peso Corporal nos Dias de Abate (42 a 60 Dias)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Idade das Aves no Abate (dias)", fontsize=12)
    ax.set_ylabel("Peso Corporal Real de Abate (kg)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distribuicao_peso_por_idade.png'), dpi=300)
    plt.savefig(os.path.join(plots_dir, 'boxplots_outliers_peso.png'), dpi=300)
    plt.close()

    # Plot 2: Mortality Distribution at Slaughter Age
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    mort_data = df_clean['mortalidade'].dropna()
    mort_filtered = mort_data[mort_data < mort_data.quantile(0.99)]
    sns.histplot(mort_filtered, bins=35, kde=True, ax=ax1, color='#2ca02c')
    ax1.set_title("Distribuição da Mortalidade Acumulada no Abate", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Mortes (cabeças)", fontsize=11)
    ax1.set_ylabel("Frequência", fontsize=11)

    mort_rate_col = '_mortalidade' if '_mortalidade' in df_clean.columns else 'mortalidade'
    mort_rate = df_clean[mort_rate_col].dropna()
    mort_rate_filtered = mort_rate[mort_rate < mort_rate.quantile(0.99)]
    sns.boxplot(x=mort_rate_filtered, ax=ax2, color='#9467bd')
    ax2.set_title("Boxplot da Taxa de Mortalidade no Abate (%)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Taxa (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distribuicao_mortalidade.png'), dpi=300)
    plt.close()

    # Plot 3: Heatmap Correlation Matrix for Slaughter Variables
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cols_of_interest = ['idade', 'peso_kg', 'gmd_abate', 'mortalidade', 'descartados', 'cab_alojadas', 'f01', 'f04', 'c05', 'c15', 'x02']
    valid_corr_cols = [c for c in cols_of_interest if c in num_cols]
    if len(valid_corr_cols) > 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df_clean[valid_corr_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, cbar=True, square=True)
        ax.set_title("Matriz de Correlação entre Variáveis no Momento do Abate", fontsize=14, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'matriz_correlacao_features.png'), dpi=300)
        plt.close()

    logger.info("EDA and Outlier Cleaning for Slaughter Age successfully completed!")
    return desc_stats

if __name__ == "__main__":
    setup_logging()
    run_eda_and_clean()
