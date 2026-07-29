# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Business Rules](https://img.shields.io/badge/Regras%20de%20Neg%C3%B3cio-Abate%20%E2%89%A5%2042%20dias-red.svg)](docs/regras_de_negocio_abate.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução refatorada dedicada exclusivamente à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias). A análise desconsidera o crescimento inicial em idades jovens para focar na variabilidade comercial do peso final ao abate.

---

## 📌 1. Principais Resultados e Métricas Otimizadas

Comparações realizadas via **5-Fold GroupKFold Cross-Validation** agrupada por lote (`lote_composto`):

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Observações |
|---|---|---|---|---|
| **Random Forest Baseline** | 0,2925 | 127,15 g | 170,56 g | Modelo Inicial |
| **HistGradientBoosting + KNN Features** | 0,3455 | 120,73 g | 163,65 g | Experimento de Vizinhança |
| **HistGradientBoosting Otimizado (Best Model)** | **0,3694** | **119,26 g** | **160,63 g** | **Melhor Desempenho (MAE < 120g)** |
| **Classificador de Meta de Peso de Abate** | **98,4% (Acurácia)** | **F1-Score: 0,98** | **Precision/Recall: 0,98** | Categorias: `Abaixo`, `Na Meta`, `Acima` |

---

## 🔬 2. Experimento de Extração de Features via KNN (Nearest Neighbors)

Testamos a extração de features de vizinhança espacial/zootécnica no espaço de atributos dos lotes:
* **Features Extraídas:** `knn_pred_weight_k15`, `knn_pred_weight_k30` (média de peso dos $15$ e $30$ lotes biologicamente mais parecidos), `knn_neighbor_std_k15` (variabilidade regional) e `knn_dist_nearest` (distância até o lote mais próximo).
* **Conclusão:** As árvores de decisão do `HistGradientBoosting` puro conseguem criar partições hiperpáticas mais nítidas diretamente sem depender de métricas de distância isotrópicas euclidianas do KNN, obtendo o menor erro absoluto (**$119,26\text{ g}$**).
* 📊 Visualização do Impacto: [knn_feature_extraction_impact.png](plots/knn_feature_extraction_impact.png).
* 📄 Relatório Salvo: [knn_feature_extraction_results.csv](data/processed/knn_feature_extraction_results.csv).

---

## 📋 3. Regras de Negócio Implementadas

As regras de negócio foram formalizadas no documento [`docs/regras_de_negocio_abate.md`](docs/regras_de_negocio_abate.md):

* **RN-01 (Janela de Abate):** Filtragem estrita de lotes com idade $42 \le \text{idade} \le 60\text{ dias}$ ($15.416$ registros de abate analisados).
* **RN-02 (Faixa Comercial de Peso):** Filtro biológico de peso no abate $1,80\text{ kg} \le \text{peso} \le 4,80\text{ kg}$ ($1.800\text{g}$ a $4.800\text{g}$).
* **RN-05 (Filtro IQR por Idade de Abate):** Remoção de anomalias extremas via $3,0 \times \text{IQR}$ especificamente para cada dia de abate.

---

## 💡 4. Diagnóstico e Explicabilidade ELI5 no Abate

### Ranking de Importância de Variáveis ELI5:

| Rank | Variável | Descrição Zootécnica / Operacional | Importância Média ELI5 |
|---|---|---|---|
| 1 | `c15` | Peso inicial do pintainho de 1 dia (g) | **23,20%** |
| 2 | `mortalidade` | Desafio sanitário e mortes acumuladas | **14,75%** |
| 3 | `cab_alojadas` | Densidade e quantidade alojada no lote | **13,03%** |
| 4 | `x02` | Distância da propriedade ao abatedouro (km) | **11,73%** |
| 5 | `idade` | Variação diária entre os dias de abate (42 a 54 dias) | **10,99%** |
| 6 | `descartados` | Descartes sanitários no lote | **9,34%** |
| 7 | `c12` | Fator multiplicador de peso aos 35 dias acima | **8,40%** |

* 📄 Relatório em Markdown: [docs/explicabilidade_eli5.md](docs/explicabilidade_eli5.md)
* 🌐 Relatório HTML Interativo: [docs/explicabilidade_eli5.html](docs/explicabilidade_eli5.html)

---

## 📈 5. Galeria de Gráficos

| Gráfico | Descrição do Diagnóstico de Abate |
|---|---|
| [distribuicao_peso_por_idade.png](plots/distribuicao_peso_por_idade.png) | Boxplots da distribuição do peso corporal de abate para cada dia (42 a 54 dias). |
| [curva_crescimento_gompertz.png](plots/curva_crescimento_gompertz.png) | Ajuste da curva de Gompertz na janela comercial de abate. |
| [predito_vs_observado_peso.png](plots/predito_vs_observado_peso.png) | Dispersão do modelo Fine-Tuned (MAE de $119\text{g}$ / RMSE de $160\text{g}$). |
| [knn_feature_extraction_impact.png](plots/knn_feature_extraction_impact.png) | Comparativo de erro entre modelo puro e com extração de KNN. |
| [boxplots_outliers_peso.png](plots/boxplots_outliers_peso.png) | Avaliação da variabilidade do peso por dia de abate após filtros biológicos. |
| [matriz_correlacao_features.png](plots/matriz_correlacao_features.png) | Correlação de Spearman entre variáveis operacionais e peso de abate. |
| [distribuicao_mortalidade.png](plots/distribuicao_mortalidade.png) | Distribuição da mortalidade acumulada no momento do abate. |
| [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) | Distribuição dos erros residuais de predição do peso de abate. |
| [analise_residuos_scatter.png](plots/analise_residuos_scatter.png) | Teste de homocedasticidade para a predição do peso de abate. |
| [matriz_confusao_peso.png](plots/matriz_confusao_peso.png) | Matriz de confusão para atingimento da meta de peso no abate. |
| [eli5_importancia_variaveis.png](plots/eli5_importancia_variaveis.png) | Barplot do ranking de importâncias ELI5 no momento do abate. |

---

## 🧠 6. Grafo de Conhecimento (Graphify)

* 🕸️ **Visualização Interativa:** [graphify-out/graph.html](graphify-out/graph.html)
* 📄 **Relatório de Audit da Arquitetura:** [graphify-out/GRAPH_REPORT.md](graphify-out/GRAPH_REPORT.md)

---

## 🛠️ 7. Guia de Execução

```bash
source .venv/bin/activate

# 1. Executar EDA e Filtro de Abate (RN-01 a RN-05)
python3 -m src.eda_outliers

# 2. Treinar Modelo de Predição do Peso de Abate
python3 -m src.models.train_predict_weight

# 3. Executar Otimização de Hiperparâmetros
python3 -m src.models.fine_tune_slaughter_model

# 4. Executar Extração de Features via KNN
python3 -m src.models.knn_feature_extraction

# 5. Executar Avaliações Avançadas e ELI5
python3 -m src.models.advanced_evaluation_eli5
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
