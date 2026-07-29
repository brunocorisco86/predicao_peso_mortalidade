# src/eda_outliers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.logger import logger

def run_eda_and_clean():
    logger.info("Starting EDA and Outlier Cleaning process...")

    input_csv = os.path.join('data', 'processed', 'unified_data.csv')
    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)

    df = pd.read_csv(input_csv, low_memory=False)
    logger.info(f"Loaded unified_data.csv with {len(df)} rows and {len(df.columns)} columns.")

    # 1. Numeric Conversions
    df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
    df['peso'] = pd.to_numeric(df['peso'], errors='coerce')
    df['mortalidade'] = pd.to_numeric(df['mortalidade'], errors='coerce')
    df['descartados'] = pd.to_numeric(df['descartados'], errors='coerce')
    df['cab_alojadas'] = pd.to_numeric(df['cab_alojadas'], errors='coerce')

    initial_len = len(df)

    # 2. Filter Biological Ranges
    # Idade de frangos de corte: 1 a 60 dias
    # Peso (kg): 0.02 kg (20g) a 5.0 kg (5000g)
    mask = (
        (df['idade'] >= 1) & (df['idade'] <= 60) &
        (df['peso'].notnull()) & (df['peso'] >= 0.02) & (df['peso'] <= 5.0)
    )
    df_clean = df[mask].copy()

    # 3. IQR Outlier Filter per Age Group (3.0 * IQR to only remove extreme errors)
    outlier_indices = set()
    for age, group in df_clean.groupby('idade'):
        if len(group) >= 30:
            q1 = group['peso'].quantile(0.25)
            q3 = group['peso'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = max(0.01, q1 - 3.0 * iqr)
            upper_bound = q3 + 3.0 * iqr
            outliers = group[(group['peso'] < lower_bound) | (group['peso'] > upper_bound)].index
            outlier_indices.update(outliers)

    df_clean = df_clean.drop(index=list(outlier_indices))
    final_len = len(df_clean)
    logger.info(f"Removed {initial_len - final_len} outlier/invalid rows ({((initial_len - final_len)/initial_len)*100:.2f}%). Cleaned dataset size: {final_len} rows.")

    # Convert peso to grams for standard poultry modeling (peso_g) while keeping peso_kg
    df_clean['peso_kg'] = df_clean['peso']
    df_clean['peso_g'] = df_clean['peso'] * 1000.0

    # Save cleaned dataset
    df_clean.to_csv(cleaned_csv, index=False)
    logger.info(f"Cleaned dataset saved to {cleaned_csv}")

    # 4. Descriptive Statistics Summary
    stats_cols = ['idade', 'peso_kg', 'peso_g', 'mortalidade', 'descartados', 'cab_alojadas', '_mortalidade', '_descartados']
    existing_stats_cols = [c for c in stats_cols if c in df_clean.columns]
    desc_stats = df_clean[existing_stats_cols].describe().T
    desc_stats['median'] = df_clean[existing_stats_cols].median()
    desc_stats['skewness'] = df_clean[existing_stats_cols].skew()
    desc_stats['kurtosis'] = df_clean[existing_stats_cols].kurtosis()
    desc_stats['null_pct'] = (df_clean[existing_stats_cols].isnull().sum() / len(df_clean)) * 100

    desc_stats_path = os.path.join('data', 'processed', 'descriptive_statistics.csv')
    desc_stats.to_csv(desc_stats_path)
    logger.info(f"Descriptive statistics saved to {desc_stats_path}")

    # 5. Visualization Plots
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Weight Distribution by Age (Scatter + Quantiles in kg)
    fig, ax = plt.subplots(figsize=(10, 6))
    age_stats = df_clean.groupby('idade')['peso_kg'].agg(['mean', 'median', lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90)]).reset_index()
    age_stats.columns = ['idade', 'mean', 'median', 'p10', 'p90']

    ax.scatter(df_clean['idade'], df_clean['peso_kg'], alpha=0.03, s=8, color='#1f77b4', label='Observações (kg)')
    ax.plot(age_stats['idade'], age_stats['median'], color='#d62728', linewidth=2.5, label='Mediana')
    ax.fill_between(age_stats['idade'], age_stats['p10'], age_stats['p90'], color='#d62728', alpha=0.2, label='Intervalo P10-P90')
    ax.set_title("Distribuição e Evolução do Peso Corporal por Idade (Frangos de Corte)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Idade das Aves (dias)", fontsize=12)
    ax.set_ylabel("Peso Corporal (kg)", fontsize=12)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distribuicao_peso_por_idade.png'), dpi=300)
    plt.close()

    # Plot 2: Boxplots of weight per key weekly age group
    sample_ages = [7, 14, 21, 28, 35, 42, 49]
    df_sample = df_clean[df_clean['idade'].isin(sample_ages)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_sample, x='idade', y='peso_kg', ax=ax, palette='Blues')
    ax.set_title("Boxplot do Peso Corporal por Idade Semanal (Filtrado)", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Idade (dias)", fontsize=12)
    ax.set_ylabel("Peso Corporal (kg)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplots_outliers_peso.png'), dpi=300)
    plt.close()

    # Plot 3: Mortality Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    mort_data = df_clean['mortalidade'].dropna()
    mort_filtered = mort_data[mort_data < mort_data.quantile(0.99)]
    sns.histplot(mort_filtered, bins=35, kde=True, ax=ax1, color='#2ca02c')
    ax1.set_title("Distribuição da Mortalidade Diária (Filtrada P99)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Mortes (cabeças)", fontsize=11)
    ax1.set_ylabel("Frequência", fontsize=11)

    mort_rate_col = '_mortalidade' if '_mortalidade' in df_clean.columns else 'mortalidade'
    mort_rate = df_clean[mort_rate_col].dropna()
    mort_rate_filtered = mort_rate[mort_rate < mort_rate.quantile(0.99)]
    sns.boxplot(x=mort_rate_filtered, ax=ax2, color='#9467bd')
    ax2.set_title("Boxplot da Taxa de Mortalidade (%)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Taxa (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distribuicao_mortalidade.png'), dpi=300)
    plt.close()

    # Plot 4: Heatmap Correlation Matrix
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cols_of_interest = ['idade', 'peso_kg', 'mortalidade', 'descartados', 'cab_alojadas', 'f01', 'f04', 'c05', 'c15', 'x02']
    valid_corr_cols = [c for c in cols_of_interest if c in num_cols]
    if len(valid_corr_cols) > 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df_clean[valid_corr_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, cbar=True, square=True)
        ax.set_title("Matriz de Correlação entre Variáveis Selecionadas", fontsize=14, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'matriz_correlacao_features.png'), dpi=300)
        plt.close()

    logger.info("EDA and Outlier Cleaning successfully completed!")
    return desc_stats

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_eda_and_clean()
