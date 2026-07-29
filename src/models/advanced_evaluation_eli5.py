# src/models/advanced_evaluation_eli5.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import eli5
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from src.utils.logger import logger

def run_advanced_evaluations():
    logger.info("Starting Advanced Evaluations: Residuals, Confusion Matrix, Cross-Validation, and ELI5...")

    cleaned_csv = os.path.join('data', 'processed', 'cleaned_data.csv')
    plots_dir = 'plots'
    docs_dir = 'docs'
    data_proc_dir = os.path.join('data', 'processed')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    df = pd.read_csv(cleaned_csv, low_memory=False)
    logger.info(f"Loaded cleaned_data.csv with {len(df)} rows.")

    # Select Features & Target
    feature_candidates = [
        'idade', 'cab_alojadas', 'mortalidade', 'descartados',
        'f01', 'f02', 'f03', 'f04', 'f05', 'f06',
        'c05', 'c06', 'c11', 'c12', 'c15', 'f07', 'f15', 'x02'
    ]
    avail_features = [c for c in feature_candidates if c in df.columns]

    df_ml = df[['lote_composto', 'peso_g'] + avail_features].dropna().copy()
    X = df_ml[avail_features]
    y = df_ml['peso_g']
    groups = df_ml['lote_composto']

    # --- 1. Cross-Validation (GroupKFold por Lote Composto) --- #
    logger.info("Running 5-Fold GroupKFold Cross-Validation by Batch (lote_composto)...")
    gkf = GroupKFold(n_splits=5)
    
    cv_r2, cv_mae, cv_rmse = [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train_f, X_test_f = X.iloc[train_idx], X.iloc[test_idx]
        y_train_f, y_test_f = y.iloc[train_idx], y.iloc[test_idx]

        rf_fold = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf_fold.fit(X_train_f, y_train_f)
        y_pred_f = rf_fold.predict(X_test_f)

        r2 = r2_score(y_test_f, y_pred_f)
        mae = mean_absolute_error(y_test_f, y_pred_f)
        rmse = np.sqrt(mean_squared_error(y_test_f, y_pred_f))

        cv_r2.append(r2)
        cv_mae.append(mae)
        cv_rmse.append(rmse)
        logger.info(f"Fold {fold} -> R²: {r2:.4f}, MAE: {mae:.2f}g, RMSE: {rmse:.2f}g")

    cv_results_df = pd.DataFrame({
        'Fold': [f'Fold {i}' for i in range(1, 6)] + ['Média', 'Desvio Padrão'],
        'R2': cv_r2 + [float(np.mean(cv_r2)), float(np.std(cv_r2))],
        'MAE (g)': cv_mae + [float(np.mean(cv_mae)), float(np.std(cv_mae))],
        'RMSE (g)': cv_rmse + [float(np.mean(cv_rmse)), float(np.std(cv_rmse))]
    })
    cv_results_path = os.path.join(data_proc_dir, 'cross_validation_results.csv')
    cv_results_df.to_csv(cv_results_path, index=False)
    logger.info(f"Cross-validation results saved to {cv_results_path}")

    # --- 2. Residuals Analysis --- #
    logger.info("Performing Residuals Analysis...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    residuals = y_test - y_pred

    # Residual Plot 1: Histogram & KDE
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(residuals, bins=50, kde=True, ax=ax, color='#1f77b4')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Resíduo Zero')
    ax.set_title(f"Distribuição dos Resíduos ($y_{{observado}} - y_{{predito}}$)\nMédia = {np.mean(residuals):.2f}g | Desv. Padrão = {np.std(residuals):.2f}g", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Erro Residual (g)", fontsize=11)
    ax.set_ylabel("Frequência", fontsize=11)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'analise_residuos_histograma.png'), dpi=300)
    plt.close()

    # Residual Plot 2: Scatter (Fitted vs Residuals)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred, residuals, alpha=0.15, s=12, color='#333333')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title("Análise de Resíduos vs Valores Preditos (Homocedasticidade)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Peso Predito (g)", fontsize=11)
    ax.set_ylabel("Resíduo (g)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'analise_residuos_scatter.png'), dpi=300)
    plt.close()

    # --- 3. Matriz de Confusão (Classificação por Desempenho do Lote) --- #
    logger.info("Building Confusion Matrix for Weight Performance Classification...")
    df_abate = df[df['idade'] >= 35].copy()
    
    # Vectorized quantile bounds per age
    quantiles_by_age = df_abate.groupby('idade')['peso_g'].agg(
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).to_dict(orient='index')

    def assign_category(row):
        q = quantiles_by_age.get(row['idade'])
        if not q:
            return 'Na Meta'
        if row['peso_g'] < q['q25']:
            return 'Abaixo da Meta'
        elif row['peso_g'] <= q['q75']:
            return 'Na Meta'
        else:
            return 'Acima da Meta'

    df_abate['categoria_peso'] = df_abate.apply(assign_category, axis=1)
    
    clf_features = [c for c in avail_features if c != 'idade']
    df_clf = df_abate[clf_features + ['categoria_peso']].dropna()

    X_clf = df_clf[clf_features]
    y_clf = df_clf['categoria_peso']
    labels = ['Abaixo da Meta', 'Na Meta', 'Acima da Meta']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)

    cm = confusion_matrix(y_test_c, y_pred_c, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_title("Matriz de Confusão: Classificação do Desempenho de Peso do Lote", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Classe Preditada", fontsize=11)
    ax.set_ylabel("Classe Real", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'matriz_confusao_peso.png'), dpi=300)
    plt.close()

    clf_rep = classification_report(y_test_c, y_pred_c, target_names=labels, output_dict=True)
    clf_rep_df = pd.DataFrame(clf_rep).T
    clf_rep_df.to_csv(os.path.join(data_proc_dir, 'classification_report.csv'))
    logger.info("Classification confusion matrix and report saved successfully.")

    # --- 4. ELI5 Explicabilidade de Variáveis --- #
    logger.info("Generating ELI5 Explanations...")
    eli5_weights = eli5.explain_weights_df(rf, feature_names=avail_features)
    eli5_weights.to_csv(os.path.join(data_proc_dir, 'eli5_feature_importance.csv'), index=False)

    html_explanation = eli5.format_as_html(eli5.explain_weights(rf, feature_names=avail_features))
    with open(os.path.join(docs_dir, 'explicabilidade_eli5.html'), 'w', encoding='utf-8') as f:
        f.write(html_explanation)

    # Markdown version of ELI5 table
    md_eli5 = "# Explicabilidade de Variáveis via ELI5\n\n"
    md_eli5 += "Esta tabela mostra o peso e a importância relativa de cada variável na decisão do modelo de regressão de peso corporal das aves:\n\n"
    md_eli5 += "| Rank | Variável | Descrição | Importância (Peso ELI5) | Desvio Padrão |\n"
    md_eli5 += "|---|---|---|---|---|\n"

    descriptions = {
        'idade': 'Idade das aves (dias)',
        'cab_alojadas': 'Cabeças de aves alojadas no lote',
        'c15': 'Peso inicial do pintainho (g)',
        'c05': 'Idade da matriz baixa (dias)',
        'c06': 'Idade da matriz alta (dias)',
        'f01': 'Vazio sanitário curto (14-18 dias)',
        'f02': 'Vazio sanitário médio (19-24 dias)',
        'f03': 'Vazio sanitário longo (>25 dias)',
        'f04': 'Número de camas (1-4)',
        'f05': 'Número de camas (5-9)',
        'f06': 'Número de camas (>10)',
        'c11': 'Fator peso 35d abaixo',
        'c12': 'Fator peso 35d acima',
        'f07': 'Lote medicado até 14 dias',
        'f15': 'Tempo de jejum pré-abate (h)',
        'x02': 'Distância até o abatedouro (km)',
        'mortalidade': 'Contagem de aves mortas',
        'descartados': 'Contagem de aves descartadas'
    }

    for idx, row in eli5_weights.iterrows():
        fname = row['feature']
        weight = row['weight']
        std = row['std']
        desc = descriptions.get(fname, fname)
        md_eli5 += f"| {idx+1} | `{fname}` | {desc} | **{weight:.4f}** | ±{std:.4f} |\n"

    with open(os.path.join(docs_dir, 'explicabilidade_eli5.md'), 'w', encoding='utf-8') as f:
        f.write(md_eli5)

    # Plot ELI5 Feature Importances Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    top_eli5 = eli5_weights.head(10)
    sns.barplot(data=top_eli5, x='weight', y='feature', ax=ax, palette='crest')
    ax.set_title("Explicabilidade ELI5: Top 10 Variáveis de Maior Importância", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Importância Média (ELI5)", fontsize=11)
    ax.set_ylabel("Variável", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'eli5_importancia_variaveis.png'), dpi=300)
    plt.close()

    logger.info("Advanced Evaluations successfully completed!")

if __name__ == "__main__":
    from src.utils.logger import setup_logging
    setup_logging()
    run_advanced_evaluations()
