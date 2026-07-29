# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Fine-Tuned Longitudinal](https://img.shields.io/badge/Modelo%20Longitudinal-Fine--Tuned-brightgreen.svg)](docs/sugestoes_melhoria_predicao.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução dedicada à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias).

---

## 📌 1. Principais Resultados e Métricas Otimizadas (Fine-Tuning Longitudinal)

Comparações consolidadas via **5-Fold GroupKFold Cross-Validation** no momento do abate ($\text{idade} \ge 42\text{ dias}$):

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Redução de Erro / Ganho |
|---|---|---|---|---|
| **Modelo Estático Baseline** | 0,3684 | 118,80 g | 160,74 g | Sem Séries Temporais |
| **Modelo Longitudinal (Item 2)** | 0,5359 | 96,92 g | 137,79 g | Séries Temporais |
| **Fine-Tuned Longitudinal Stacking (Best Model)** | **0,5405** | **96,07 g** | **137,11 g** | **🏆 Menor Erro Absoluto (MAE = 96g / Desvio 3,2%)** |
| **Classificador de Meta no Abate (3 Classes)** | **68,81% (Acurácia)** | **F1-Score: 0,69** | **Precision/Recall: 0,69** | Categorias: `Abaixo`, `Na Meta`, `Acima` |

> 📌 **Impacto no Negócio:** Com a modelagem longitudinal e otimização de hiperparâmetros, o erro médio absoluto na previsão do peso de um frango de $3,0\text{kg}$ caiu para **$96,07\text{g}$** (apenas **$3,2\%$** de desvio), enquanto o poder explicativo ($R^2$) atingiu **$0,5405$**.

---

## 🔬 2. Diagnósticos Avançados do Modelo Otimizado

### A. Análise de Resíduos (Erro Residual = $y_{\text{real}} - y_{\text{predito\_longitudinal}}$)
* **Média dos Resíduos:** $\mu = 0,08\text{ g}$ (centrado perfeitamente em zero, modelo imparcial livre de viés).
* **Mediana dos Resíduos:** $0,79\text{ g}$.
* **Desvio Padrão dos Resíduos:** $\sigma = 137,13\text{ g}$.
* **Homocedasticidade:** Teste scatterplot confirma variância constante em todas as faixas de peso de abate ($1.800\text{g}$ a $4.800\text{g}$).
* 📊 Visualização: [plots/analise_residuos_histograma.png](plots/analise_residuos_histograma.png) e [plots/analise_residuos_scatter.png](plots/analise_residuos_scatter.png).

### B. Matriz de Confusão para Atingimento da Meta de Abate
* **Classificação em 3 Categorias de Abate:** `Abaixo da Meta` ($< P_{25}$), `Na Meta` ($P_{25}-P_{75}$), `Acima da Meta` ($> P_{75}$).
* **Acurácia de Classificação:** **68,81%** para acerto exato da categoria comercial.
* 📊 Visualização da Matriz de Confusão: [plots/matriz_confusao_peso.png](plots/matriz_confusao_peso.png).
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
| [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) | Distribuição simétrica normal dos resíduos do modelo longitudinal tunado ($\mu = 0,08\text{g}$). |
| [analise_residuos_scatter.png](plots/analise_residuos_scatter.png) | Teste de homocedasticidade confirmando erro uniforme em toda a faixa de peso. |
| [matriz_confusao_peso.png](plots/matriz_confusao_peso.png) | Heatmap da Matriz de Confusão para atestar atingimento da meta comercial de abate (68,8% de acurácia). |
| [modelo_longitudinal_predito_vs_real.png](plots/modelo_longitudinal_predito_vs_real.png) | Dispersão do Modelo Longitudinal ($\text{MAE} = 96,07\text{g}$ / $R^2 = 0,5405$). |
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
