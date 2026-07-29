# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Business Rules](https://img.shields.io/badge/Regras%20de%20Neg%C3%B3cio-Abate%20%E2%89%A5%2042%20dias-red.svg)](docs/regras_de_negocio_abate.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução dedicada à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias). 

---

## 💡 1. Metodologia de Ajuste da Curva de Gompertz vs Alvo de Abate

Você está completamente certo: **a curva de Gompertz deve ser ajustada utilizando o histórico completo de crescimento (dias 1 a 60)**, pois os dados iniciais (pesagem do pintainho aos 1, 7 e 14 dias) ancoram a trajetória biológica e permitem estimar os parâmetros fisiológicos com máxima precisão.

* **Ajuste da Curva (Fitting):** Feito sobre todo o histórico etário (dias 1 a 60).
  $$\mathbf{W(t) = 6260,16 \cdot \exp\left(-4,7378 \cdot \exp(-0,0449 \cdot t)\right)}$$
* **Avaliação de Desempenho (Target Scope):** Filtrada **exclusivamente na janela de abate ($\mathbf{t \ge 42\text{ dias}}$)**.

---

## 📌 2. Principais Resultados e Métricas Otimizadas

Comparações realizadas via **5-Fold GroupKFold Cross-Validation** no momento do abate:

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Âmbito de Avaliação |
|---|---|---|---|---|
| **Modelo Biológico de Gompertz** | 0,0102 | 151,04 g | 208,83 g | Ajustado em 1-60d / Avaliado no Abate |
| **Random Forest Baseline** | 0,2925 | 127,15 g | 170,56 g | Avaliado no Abate |
| **HistGradientBoosting Otimizado (Best Model)** | **0,3694** | **119,26 g** | **160,63 g** | **Melhor Desempenho (MAE < 120g no Abate)** |
| **Classificador de Meta de Peso de Abate** | **98,4% (Acurácia)** | **F1-Score: 0,98** | **Precision/Recall: 0,98** | Categorias: `Abaixo`, `Na Meta`, `Acima` |

---

## 📋 3. Regras de Negócio Implementadas

As regras de negócio foram formalizadas no documento [`docs/regras_de_negocio_abate.md`](docs/regras_de_negocio_abate.md):

* **RN-01 (Janela de Abate):** Avaliação de erro focada em lotes com idade $42 \le \text{idade} \le 60\text{ dias}$ ($15.416$ registros de abate analisados).
* **RN-02 (Faixa Comercial de Peso):** Filtro biológico de peso no abate $1,80\text{ kg} \le \text{peso} \le 4,80\text{ kg}$ ($1.800\text{g}$ a $4.800\text{g}$).
* **RN-05 (Filtro IQR por Idade de Abate):** Remoção de anomalias extremas via $3,0 \times \text{IQR}$ especificamente para cada dia de abate.

---

## 🔬 4. Diagnóstico e Explicabilidade ELI5 no Abate

Ranking de importâncias de variáveis no momento do abate:

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
| [curva_crescimento_gompertz.png](plots/curva_crescimento_gompertz.png) | Curva Biológica de Gompertz ajustada em 1-60 dias com destaque para a Janela de Abate (42-60d). |
| [distribuicao_peso_por_idade.png](plots/distribuicao_peso_por_idade.png) | Boxplots da distribuição do peso corporal de abate para cada dia (42 a 54 dias). |
| [predito_vs_observado_peso.png](plots/predito_vs_observado_peso.png) | Dispersão do modelo Fine-Tuned (MAE de $119\text{g}$ / RMSE de $160\text{g}$). |
| [knn_feature_extraction_impact.png](plots/knn_feature_extraction_impact.png) | Comparativo de erro entre modelo puro e com extração de KNN. |
| [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) | Distribuição dos erros residuais de predição do peso de abate. |
| [matriz_confusao_peso.png](plots/matriz_confusao_peso.png) | Matriz de confusão para atingimento da meta de peso no abate. |
| [eli5_importancia_variaveis.png](plots/eli5_importancia_variaveis.png) | Barplot do ranking de importâncias ELI5 no momento do abate. |

---

## 🧠 6. Grafo de Conhecimento (Graphify)

* 🕸️ **Visualização Interativa:** [graphify-out/graph.html](graphify-out/graph.html)
* 📄 **Relatório de Audit da Arquitetura:** [graphify-out/GRAPH_REPORT.md](graphify-out/GRAPH_REPORT.md)

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
