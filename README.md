# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Business Rules](https://img.shields.io/badge/Regras%20de%20Neg%C3%B3cio-Abate%20%E2%89%A5%2042%20dias-red.svg)](docs/regras_de_negocio_abate.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução refatorada dedicada exclusivamente à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias).

---

## 📌 1. Principais Resultados do Comparativo Multi-Modelo (Abate)

Comparações realizadas via **5-Fold GroupKFold Cross-Validation** no momento do abate ($\text{idade} \ge 42\text{ dias}$):

| Modelo | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Métrica $R^2$ | Observações |
|---|---|---|---|---|
| **Stacking Ensemble (Meta-Modelo)** | **118,80 g** | **160,74 g** | **0,3684** | **Menor MAE Absoluto (Erro < 119g)** |
| **HistGradientBoosting** | **118,90 g** | **160,72 g** | **0,3686** | **Menor RMSE / Maior $R^2$** |
| **LightGBM Regressor** | **119,42 g** | **161,44 g** | **0,3627** | Excelente velocidade e precisão |
| **XGBoost Regressor** | **119,77 g** | **161,68 g** | **0,3608** | Regularização L1/L2 |
| **Extra Trees Regressor** | 126,70 g | 170,91 g | 0,2857 | Árvores Aleatorizadas |
| **Rede Neural MLP (Perceptron)** | 136,05 g | 183,68 g | 0,1750 | 2 Camadas Ocultas (128, 64) |
| **Classificador de Meta de Peso** | **98,4% (Acurácia)** | **F1-Score: 0,98** | **Precision/Recall: 0,98** | Categorias: `Abaixo`, `Na Meta`, `Acima` |

* 📊 Visualização Comparativa: [plots/comparativo_modelos_avancados.png](plots/comparativo_modelos_avancados.png).
* 📄 Tabela Exportada: [data/processed/advanced_transformations_models_results.csv](data/processed/advanced_transformations_models_results.csv).

---

## 🧪 2. Transformações Avançadas nas Variáveis Independentes

Para alcançar a redução do erro médio absoluto para **118,80g** no abate, aplicamos:

1. **Transformações Logarítmicas:**
   - $\log(\text{x02} + 1)$: Suavização da cauda longa da distância até o abatedouro.
   - $\log(\text{cab\_alojadas} + 1)$: Estabilização de variância de populações grandes.
2. **Efeitos Não-Lineares & Quadráticos:**
   - $(\text{mortalidade\_pct})^2$: Captura do efeito severo não-linear de surtos sanitários.
3. **Features de Razão Zootécnica:**
   - `mortalidade_pct` e `descartados_pct`: Percentuais de perda acumulada.
   - `c15_dev_lineage`: Desvio do peso do pintainho de 1 dia (`c15`) em relação à média da sua linhagem (`c16`).

---

## 📋 3. Regras de Negócio Implementadas ([`docs/regras_de_negocio_abate.md`](docs/regras_de_negocio_abate.md))

* **RN-01 (Janela de Abate):** $42 \le \text{idade} \le 60\text{ dias}$ ($15.416$ registros de abate analisados).
* **RN-02 (Faixa Comercial de Peso):** $1,80\text{ kg} \le \text{peso} \le 4,80\text{ kg}$ ($1.800\text{g}$ a $4.800\text{g}$).
* **RN-05 (Filtro IQR por Idade de Abate):** Remoção de anomalias extremas via $3,0 \times \text{IQR}$ por dia de abate.

---

## 📈 4. Galeria de Gráficos

| Gráfico | Descrição do Diagnóstico de Abate |
|---|---|
| [comparativo_modelos_avancados.png](plots/comparativo_modelos_avancados.png) | Comparativo de RMSE e $R^2$ entre LightGBM, XGBoost, Stacking, HistGB, ExtraTrees e MLP. |
| [distribuicao_peso_por_idade.png](plots/distribuicao_peso_por_idade.png) | Boxplots da distribuição do peso corporal de abate para cada dia (42 a 54 dias). |
| [curva_crescimento_gompertz.png](plots/curva_crescimento_gompertz.png) | Curva Biológica de Gompertz ajustada em 1-60 dias com destaque para a Janela de Abate (42-60d). |
| [predito_vs_observado_peso.png](plots/predito_vs_observado_peso.png) | Dispersão do modelo Fine-Tuned (MAE de $118,8\text{g}$ / RMSE de $160,7\text{g}$). |
| [knn_feature_extraction_impact.png](plots/knn_feature_extraction_impact.png) | Comparativo de erro entre modelo puro e com extração de KNN. |
| [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) | Distribuição dos erros residuais de predição do peso de abate. |
| [matriz_confusao_peso.png](plots/matriz_confusao_peso.png) | Matriz de confusão para atingimento da meta de peso no abate. |
| [eli5_importancia_variaveis.png](plots/eli5_importancia_variaveis.png) | Barplot do ranking de importâncias ELI5 no momento do abate. |

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

# 3. Executar Comparativo Avançado de Modelos (LightGBM, XGBoost, Stacking)
python3 -m src.models.advanced_transformations_models
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
