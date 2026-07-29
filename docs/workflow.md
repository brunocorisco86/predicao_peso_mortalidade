# Roteiro do Projeto (Workflow): Predição de Peso e Mortalidade

Este documento descreve o fluxo de trabalho executado para a predição de peso e mortalidade de aves de corte.

## Fases do Projeto e Status

1. **Extração e Carga de Dados (ETL) - [CONCLUÍDO]**
   - Racionalização e carga dos arquivos Excel brutos em `database/prediction_data.db`.
   - Script `src/etl/export_unified_data.py` gera a base consolidada em `data/processed/unified_data.csv` (104.601 linhas, 68 colunas).

2. **Análise Exploratória e Tratamento de Outliers (EDA) - [CONCLUÍDO]**
   - Execução do script `src/eda_outliers.py`.
   - Remoção de incoerências biológicas de idade e peso em kg.
   - Filtragem via amplitude interquartil (3.0 x IQR). Base limpa resultante: 88.021 registros (`data/processed/cleaned_data.csv`).
   - Geração de plots estatísticos e matrizes de correlação em `plots/`.

3. **Modelagem Preditiva (Curvas de Crescimento & Machine Learning) - [CONCLUÍDO]**
   - Ajuste do modelo não-linear biológico de Gompertz: $W(t) = 6281.25 \cdot \exp(-4.7536 \cdot \exp(-0.0449 \cdot t))$ ($R^2 = 0.9848$).
   - Treinamento do modelo de Machine Learning (Random Forest): $R^2 = 0.9903$, $\text{MAE} = 66.59\text{ g}$.
   - Validação cruzada (5-Fold GroupKFold): $R^2 = 0.9897$, $\text{MAE} = 67.70\text{ g}$.
   - Salvamento dos modelos e visualizações em `src/models/saved/` e `plots/`.

4. **Diagnósticos Avançados e Explicabilidade ELI5 - [CONCLUÍDO]**
   - Análise de Resíduos ($\mu = -1.14\text{ g}$, homocedasticidade confirmada).
   - Matriz de Confusão para classificação da meta de peso de abate (acurácia $98.4\%$).
   - Explicabilidade de variáveis via ELI5 em `docs/explicabilidade_eli5.md`.

5. **Documentação e Repositório Remoto - [CONCLUÍDO]**
   - Documentação atualizada em `docs/` e `README.md`.
   - Sincronização e push efetuados para o repositório Git remoto.