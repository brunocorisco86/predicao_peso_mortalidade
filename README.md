# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Longitudinal Model](https://img.shields.io/badge/Modelo%20Longitudinal-Fine--Tuned-brightgreen.svg)](docs/sugestoes_melhoria_predicao.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução dedicada à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias).

---

## 📌 1. Principais Resultados e Métricas Otimizadas (Fine-Tuning Longitudinal)

Após **Engenharia de Séries Temporais Longitudinais** e **Fine-Tuning de Hiperparâmetros** (LightGBM + XGBoost + HistGB Stacking) via **5-Fold GroupKFold Cross-Validation**, obteve-se o melhor desempenho absoluto do projeto:

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Redução de Erro / Ganho |
|---|---|---|---|---|
| **Modelo Estático Baseline** | 0,3684 | 118,80 g | 160,74 g | Sem Séries Temporais |
| **Modelo Longitudinal Baseline** | 0,5359 | 96,92 g | 137,79 g | Séries Temporais (Item 2) |
| **Fine-Tuned Longitudinal Stacking (Best Model)** | **0,5405** | **96,07 g** | **137,11 g** | **🏆 Menor Erro Absoluto (MAE = 96g / Desvio 3,2%)** |
| **Classificador de Meta no Abate** | **98,4% (Acurácia)** | **F1-Score: 0,98** | **Precision/Recall: 0,98** | Categorias: `Abaixo`, `Na Meta`, `Acima` |

---

## 🔬 2. Diagnósticos Avançados do Modelo Otimizado

### A. Análise de Resíduos (Erro Residual = $y_{\text{real}} - y_{\text{predito}}$)
* **Média dos Resíduos:** $\mu = -0,48\text{ g}$ (centrado perfeitamente em zero, livre de viés).
* **Homocedasticidade:** Variância uniforme dos erros em todas as faixas de peso de abate ($1.800\text{g}$ a $4.800\text{g}$).
* 📊 Visualização: [plots/analise_residuos_histograma.png](plots/analise_residuos_histograma.png) e [plots/analise_residuos_scatter.png](plots/analise_residuos_scatter.png).

### B. Matriz de Confusão para Atingimento da Meta de Abate
* **Acurácia:** **98,4%** na classificação correta entre as faixas comerciais (`Abaixo da Meta`, `Na Meta`, `Acima da Meta`).
* 📊 Visualização: [plots/matriz_confusao_peso.png](plots/matriz_confusao_peso.png).
* 📄 Relatório Detalhado: [data/processed/classification_report.csv](data/processed/classification_report.csv).

---

## 📋 3. Regras de Negócio Implementadas ([`docs/regras_de_negocio_abate.md`](docs/regras_de_negocio_abate.md))

* **RN-01 (Janela de Abate):** $42 \le \text{idade} \le 60\text{ dias}$ ($15.416$ registros de abate analisados).
* **RN-02 (Faixa Comercial de Peso):** $1,80\text{ kg} \le \text{peso} \le 4,80\text{ kg}$ ($1.800\text{g}$ a $4.800\text{g}$).
* **RN-05 (Filtro IQR por Idade de Abate):** Remoção de anomalias extremas via $3,0 \times \text{IQR}$ por dia de abate.

---

## 📈 4. Galeria de Gráficos

| Gráfico | Descrição do Diagnóstico de Abate |
|---|---|
| [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) | Distribuição simétrica normal dos resíduos do modelo tunado ($\mu = -0,48\text{g}$). |
| [analise_residuos_scatter.png](plots/analise_residuos_scatter.png) | Gráfico de dispersão de resíduos para validação de homocedasticidade. |
| [matriz_confusao_peso.png](plots/matriz_confusao_peso.png) | Matriz de confusão para atingimento da meta de peso no abate (98,4% de acurácia). |
| [modelo_longitudinal_predito_vs_real.png](plots/modelo_longitudinal_predito_vs_real.png) | Dispersão do Modelo Longitudinal de Séries Temporais ($\text{MAE} = 96,07\text{g}$ / $R^2 = 0,5405$). |
| [importancia_features_longitudinal.png](plots/importancia_features_longitudinal.png) | Ranking de importância de variáveis com pesagens intermediárias e GPD. |
| [curva_crescimento_gompertz.png](plots/curva_crescimento_gompertz.png) | Curva Biológica de Gompertz ajustada em 1-60 dias. |

---

## 🧠 5. Grafo de Conhecimento (Graphify)

* 🕸️ **Visualização Interativa:** [graphify-out/graph.html](graphify-out/graph.html)
* 📄 **Relatório de Audit da Arquitetura:** [graphify-out/GRAPH_REPORT.md](graphify-out/GRAPH_REPORT.md)

---

## 🛠️ 6. Guia de Execução

```bash
source .venv/bin/activate

# 1. Executar EDA e Filtro de Abate (RN-01 a RN-05)
python3 -m src.eda_outliers

# 2. Treinar Modelo de Predição do Peso de Abate
python3 -m src.models.train_predict_weight

# 3. Executar Fine-Tuning Longitudinal, Resíduos e Matriz de Confusão
python3 -m src.models.fine_tune_and_evaluate_longitudinal
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
