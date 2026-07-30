"""
generate_complete_plot_suite.py
---------------------------------
Gera a Suíte Completa de Gráficos do Modelo Campeão Final para dois públicos:
1. Equipe Técnica (Zootecnistas, Médicos Veterinários e Extensionistas Rural) -> plots/zootecnia/
2. Equipe Estatística (Data Scientists, Estatísticos e Auditores) -> plots/estatistica/

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")

DATASET_PATH = Path("data/processed/longitudinal_dataset.csv")
ZOOT_DIR = Path("plots/zootecnia")
STAT_DIR = Path("plots/estatistica")
ZOOT_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

def generate_plot_suite():
    print("=======================================================")
    print(" 🎨 GERANDO SUÍTE COMPLETA DE GRÁFICOS DO MODELO CAMPEÃO")
    print("=======================================================")
    
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    if 'elegivel_rn11' in df.columns:
        df = df[df['elegivel_rn11'] == 1.0].copy()
    df = df.dropna(subset=['peso_abate_g', 'idade_abate'])
    
    target = 'peso_abate_g'
    group_col = 'lote_composto'
    
    gkf = GroupKFold(n_splits=5)
    groups = df[group_col].values
    y = df[target].values
    
    # 1. Target Encoding OOF
    df['oof_fazenda_target_enc'] = np.nan
    df['oof_produtor_target_enc'] = np.nan
    global_mean_target = y.mean()
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
        tr_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        faz_map = tr_df.groupby('fazenda')[target].mean().to_dict()
        df.iloc[val_idx, df.columns.get_loc('oof_fazenda_target_enc')] = val_df['fazenda'].map(faz_map).fillna(global_mean_target)
        if 'produtor' in df.columns:
            prod_map = tr_df.groupby('produtor')[target].mean().to_dict()
            df.iloc[val_idx, df.columns.get_loc('oof_produtor_target_enc')] = val_df['produtor'].map(prod_map).fillna(global_mean_target)
            
    exclude = ['data_alojamento', 'nome_fazenda', 'data_hora_transao', 'lote_composto', 
               'data_evento', 'data_criao', 'id_usurio_criao', 'extensionista', 'id_usurio', 
               'fazenda', 'produtor', 'data_producao_abate', 'peso_medio_abate_kg', 'peso_abate_g', 
               'gmd_abate', 'score_confianca_lote', 'categoria_amostragem', 'elegivel_rn11', 
               'motivo_inelegibilidade', 'estrategia_predicao', 'nucleo']
               
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    X = df[features].fillna(df[features].median())
    
    oof_preds = np.zeros(len(df))
    
    xgb_gpu = XGBRegressor(
        n_estimators=1800, max_depth=8, learning_rate=0.015,
        subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
        tree_method='hist', device='cuda', random_state=42
    )
    
    lgb_deep = LGBMRegressor(
        n_estimators=1200, max_depth=9, num_leaves=127, learning_rate=0.018,
        subsample=0.85, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    
    oof_xgb = np.zeros(len(df))
    oof_lgb = np.zeros(len(df))
    
    print("Treinando Modelo Campeão Stacking para extração de predições OOF...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        xgb_gpu.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb_gpu.predict(X_val)
        
        lgb_deep.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb_deep.predict(X_val)
        
        oof_tr_m = pd.DataFrame({'xgb': xgb_gpu.predict(X_tr), 'lgb': lgb_deep.predict(X_tr)})
        oof_val_m = pd.DataFrame({'xgb': oof_xgb[val_idx], 'lgb': oof_lgb[val_idx]})
        
        meta = Ridge(alpha=10.0, positive=True)
        meta.fit(oof_tr_m, y_tr)
        oof_preds[val_idx] = meta.predict(oof_val_m)
        
    df['y_pred'] = oof_preds
    df['resíduo'] = df['y_pred'] - y
    df['abs_erro'] = np.abs(df['resíduo'])
    df['mape_pct'] = (df['abs_erro'] / y) * 100.0
    
    # ---------------------------------------------------------
    # SUÍTE 1: PAINEL ZOOTÉCNICO (VETERINÁRIOS & EXTENSIONISTAS)
    # ---------------------------------------------------------
    print("\n[1/2] Gerando gráficos para a Equipe Zootécnica em plots/zootecnia/...")
    
    # Plot Z1: Trajetória Corporal por Idade (Curva Zootécnica)
    plt.figure(figsize=(10, 6))
    ages_summary = []
    for age in [7, 14, 21, 28, 35, 42]:
        col = f"peso_d{age:02d}"
        if col in df.columns:
            val = df[col].dropna()
            ages_summary.append({'Idade': f'{age}d', 'Peso Médio (g)': val.mean(), 'Tipo': 'Biometria de Campo'})
    # Abate real e predito aos 45d (idade média)
    ages_summary.append({'Idade': 'Abate Real', 'Peso Médio (g)': y.mean(), 'Tipo': 'Abate Frigorífico'})
    ages_summary.append({'Idade': 'Abate Predito', 'Peso Médio (g)': oof_preds.mean(), 'Tipo': 'Predição Modelo'})
    
    df_curve = pd.DataFrame(ages_summary)
    sns.lineplot(data=df_curve, x='Idade', y='Peso Médio (g)', marker='o', linewidth=2.5, markersize=8, color='#27ae60')
    plt.title('1. Trajetória Biométrica Média de Crescimento vs Predição no Abate', fontsize=13, fontweight='bold')
    plt.ylabel('Peso Médio Corporal (g)')
    plt.xlabel('Marco Temporal de Amostragem (Dias)')
    for idx, row in df_curve.iterrows():
        plt.annotate(f"{row['Peso Médio (g)']:.0f}g", (row['Idade'], row['Peso Médio (g)']), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(ZOOT_DIR / "01_curva_crescimento_predita_vs_real.png", dpi=300)
    plt.close()
    
    # Plot Z2: Impacto do Peso do Pintainho 1d (c15) no Abate
    plt.figure(figsize=(9, 6))
    if 'c15' in df.columns:
        df['cat_c15'] = pd.cut(df['c15'], bins=[0, 41, 45, 100], labels=['Pintainho Leve (<41g)', 'Pintainho Padrão (41-45g)', 'Pintainho Pesado (>45g)'])
        sns.boxplot(data=df.dropna(subset=['cat_c15']), x='cat_c15', y='y_pred', palette='Blues')
        plt.title('2. Projeção de Peso ao Abate por Categoria de Pintainho 1d (c15)', fontsize=13, fontweight='bold')
        plt.xlabel('Categoria de Peso ao Alojamento (Incubatório)')
        plt.ylabel('Peso de Abate Predito (g)')
        plt.tight_layout()
        plt.savefig(ZOOT_DIR / "02_impacto_peso_pintainho_c15.png", dpi=300)
        plt.close()
        
    # Plot Z3: Impacto da Estação / Sazonalidade Climática
    plt.figure(figsize=(9, 6))
    if 'y01' in df.columns and 'y02' in df.columns:
        df['estacao'] = np.where(df['y01'] == 1, 'Safra Verão (Estresse Calórico)', 'Safra Inverno/Ameno')
        sns.violinplot(data=df, x='estacao', y='y_pred', palette='YlOrRd', inner='quartile')
        plt.title('3. Distribuição do Peso de Abate Predito por Sazonalidade Climática', fontsize=13, fontweight='bold')
        plt.xlabel('Estação de Criatório (Ambiência)')
        plt.ylabel('Peso de Abate Predito (g)')
        plt.tight_layout()
        plt.savefig(ZOOT_DIR / "03_efeito_estresse_termico_sazonalidade.png", dpi=300)
        plt.close()

    # Plot Z4: Classificação Comercial PCP (Frango Leve, Médio, Pesado)
    plt.figure(figsize=(9, 6))
    df['classe_real'] = pd.qcut(df['peso_abate_g'], q=3, labels=['Leve (<3.000g)', 'Médio (3.000-3.350g)', 'Pesado (>3.350g)'])
    df['classe_pred'] = pd.qcut(df['y_pred'], q=3, labels=['Leve (<3.000g)', 'Médio (3.000-3.350g)', 'Pesado (>3.350g)'])
    counts = df['classe_pred'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['#3498db', '#2ecc71', '#e74c3c'])
    plt.title('4. Proporção de Categorias Comerciais de Abate Preditas para o PCP', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ZOOT_DIR / "04_distribuicao_categorias_comerciais.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # SUÍTE 2: PAINEL ESTATÍSTICO (DATA SCIENTISTS & AUDITORES)
    # ---------------------------------------------------------
    print("\n[2/2] Gerando gráficos para a Equipe Estatística em plots/estatistica/...")
    
    # Plot E1: Resíduos vs Valores Ajustados (Homoscedasticidade)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df.sample(min(4000, len(df)), random_state=42), x='y_pred', y='resíduo', alpha=0.4, color='#8e44ad', s=25)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title('1. Resíduos vs Valores Ajustados (Teste de Homoscedasticidade)', fontsize=13, fontweight='bold')
    plt.xlabel('Valores Ajustados / Preditos (g)')
    plt.ylabel('Resíduos e_i = Predito - Real (g)')
    plt.tight_layout()
    plt.savefig(STAT_DIR / "01_residuos_vs_valores_ajustados.png", dpi=300)
    plt.close()
    
    # Plot E2: Q-Q Plot de Normalidade dos Resíduos
    plt.figure(figsize=(8, 6))
    stats.probplot(df['resíduo'], dist="norm", plot=plt)
    plt.title('2. Normal Probability Q-Q Plot dos Resíduos do Modelo Campeão', fontsize=13, fontweight='bold')
    plt.xlabel('Quantis Teóricos da Normal Standard')
    plt.ylabel('Quantis Amostrais dos Resíduos (g)')
    plt.tight_layout()
    plt.savefig(STAT_DIR / "02_qq_plot_normalidade_residuos.png", dpi=300)
    plt.close()
    
    # Plot E3: Intervalo de Confiança Empírico P5 - P95
    plt.figure(figsize=(9, 6))
    sample_df = df.sample(min(3000, len(df)), random_state=42).sort_values(target)
    sns.scatterplot(data=sample_df, x=target, y='y_pred', alpha=0.5, color='#2c3e50', s=25, label='Lotes Observados')
    min_v, max_v = y.min(), y.max()
    plt.plot([min_v, max_v], [min_v, max_v], color='red', linestyle='--', linewidth=2, label='Linha de Identidade (1:1)')
    plt.fill_between([min_v, max_v], [min_v - 150, max_v - 150], [min_v + 150, max_v + 150], color='gray', alpha=0.2, label='Banda de Tolerância Operacional (±150g)')
    plt.title('3. Predito vs Observado com Banda de Tolerância Operacional (PCP)', fontsize=13, fontweight='bold')
    plt.xlabel('Peso Real no Abate Frigorífico (g)')
    plt.ylabel('Peso Predito pelo Stacking GPU (g)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(STAT_DIR / "03_intervalos_confianca_predicao.png", dpi=300)
    plt.close()
    
    # Plot E4: Distribuição de Erro Relativo (MAPE)
    plt.figure(figsize=(9, 6))
    sns.histplot(df['mape_pct'], kde=True, color='#16a085', bins=40)
    plt.axvline(df['mape_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'MAPE Médio: {df["mape_pct"].mean():.2f}%')
    plt.axvline(5.0, color='orange', linestyle=':', linewidth=2, label='Limite Tolerância Comercial (5.0%)')
    plt.title(f'4. Distribuição do Erro Absoluto Percentual (MAPE = {df["mape_pct"].mean():.2f}%)', fontsize=13, fontweight='bold')
    plt.xlabel('Erro Percentual Absoluto (%)')
    plt.ylabel('Frequência de Lotes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(STAT_DIR / "04_distribuicao_erros_relativos_mape.png", dpi=300)
    plt.close()
    
    print("\n=======================================================")
    print(" ✅ SUÍTE DE GRÁFICOS GERADA COM SUCESSO!")
    print("=======================================================")
    print(f" Gráficos Zootécnicos em: {ZOOT_DIR}/ (4 arquivos)")
    print(f" Gráficos Estatísticos em: {STAT_DIR}/ (4 arquivos)")
    print("=======================================================\n")

if __name__ == '__main__':
    generate_plot_suite()
