# 🐥 Predição de Peso e Mortalidade em Aves de Corte

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Data%20Science-Gompertz%20%7C%20RandomForest%20%7C%20ELI5-green.svg)](docs/premissas.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução completa de **engenharia de dados (ETL)**, **análise exploratória de dados (EDA)**, **tratamento biológico de outliers**, **modelagem preditiva de peso corporal de frangos de corte (Gompertz & Machine Learning)** e **avaliações diagnósticas avançadas (Resíduos, Matriz de Confusão, Validação Cruzada 5-Fold e Explicabilidade via ELI5)**.

---

## 📌 1. Principais Resultados e Métricas dos Modelos

A avaliação foi realizada utilizando **5-Fold GroupKFold Cross-Validation** agrupada por lote de produção (`lote_composto`), garantindo a eliminação de *data leakage*.

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Observações |
|---|---|---|---|---|
| **Curva Não-Linear de Gompertz** | **0,9848** | **81,81 g** | **126,81 g** | Modelo Zootécnico Biológico Diário |
| **Random Forest (Regressão)** | **0,9903** | **66,59 g** | **102,00 g** | Treinamento com 18 Features |
| **Random Forest (Validação Cruzada 5-Fold)** | **0,9897 ± 0,0004** | **67,70 g ± 0,81 g** | **105,55 g ± 2,08 g** | Validação Agrupada por Lote |
| **Classificador de Desempenho ao Abate** | **98,4% (Acurácia)** | **F1-Score: 0,98** | **Precision/Recall: 0,98** | Matriz de Confusão (3 Classes de Peso) |

### 📐 Equação Ajustada da Curva de Gompertz
$$W(t) = 6281,25 \cdot \exp\left(-4,7536 \cdot \exp(-0,0449 \cdot t)\right)$$
* $W(t)$: Peso corporal previsto em gramas no dia $t$.
* $A = 6281,25\text{ g}$: Peso assintótico teórico em maturidade.
* $b = 4,7536$: Constante de integração ligada ao peso inicial.
* $k = 0,0449\text{ dia}^{-1}$: Taxa relativa de maturação.

---

## 🔬 2. Diagnóstico do Modelo & Explicabilidade ELI5

### A. Análise de Resíduos
A análise dos resíduos ($y_{\text{observado}} - y_{\text{predito}}$) confirma a solidez estatística do modelo:
* **Média dos Resíduos:** $\mu = -1,14\text{ g}$ (centrada em zero, livre de viés de sub ou superestimativa).
* **Homocedasticidade:** Variância constante ao longo da evolução etária das aves.
* 📊 Visualizações: [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) e [analise_residuos_scatter.png](plots/analise_residuos_scatter.png).

### B. Matriz de Confusão para Meta de Abate
Categorizamos os lotes a partir do 35º dia em 3 faixas de desempenho (`Abaixo da Meta`, `Na Meta`, `Acima da Meta`).
* 📊 Visualização: [matriz_confusao_peso.png](plots/matriz_confusao_peso.png).
* 📄 Relatório Detalhado: [classification_report.csv](data/processed/classification_report.csv).

### C. Explicabilidade de Variáveis via ELI5
Utilizamos a biblioteca **ELI5** para quantificar a contribuição individual de cada variável do modelo:

| Rank | Variável | Descrição Zootécnica / Operacional | Peso ELI5 (Importância) | Desvio Padrão |
|---|---|---|---|---|
| 1 | `idade` | Idade cronológica das aves (dias) | **0,9923** | $\pm 0,0002$ |
| 2 | `c15` | Peso inicial do pintainho de 1 dia (g) | **0,0030** | $\pm 0,0001$ |
| 3 | `c12` | Fator multiplicador de peso aos 35 dias acima | **0,0013** | $\pm 0,0001$ |
| 4 | `mortalidade` | Contagem acumulada de aves mortas no lote | **0,0007** | $\pm 0,0001$ |
| 5 | `cab_alojadas` | Densidade / Total de cabeças alojadas | **0,0007** | $\pm 0,0001$ |
| 6 | `x02` | Distância entre a propriedade e o abatedouro (km) | **0,0006** | $\pm 0,0001$ |
| 7 | `descartados` | Contagem de descartes sanitários | **0,0005** | $\pm 0,0001$ |
| 8 | `c11` | Fator multiplicador de peso aos 35 dias abaixo | **0,0005** | $\pm 0,0001$ |
| 9 | `c05` | Idade da matriz baixa (semanas) | **0,0001** | $\pm 0,0000$ |
| 10 | `f06` | Número de reutilizações da cama ($>10$) | **0,0001** | $\pm 0,0000$ |

* 📄 Relatório Markdown: [explicabilidade_eli5.md](docs/explicabilidade_eli5.md)
* 🌐 Relatório HTML Interativo: [explicabilidade_eli5.html](docs/explicabilidade_eli5.html)
* 📊 Gráfico de Importâncias: [eli5_importancia_variaveis.png](plots/eli5_importancia_variaveis.png)

---

## 📈 3. Galeria de Visualizações Geradas

| Gráfico | Descrição Zootécnica / Analítica |
|---|---|
| [distribuicao_peso_por_idade.png](plots/distribuicao_peso_por_idade.png) | Curva de ganho de peso corporal observado vs Mediana e Intervalo P10-P90. |
| [curva_crescimento_gompertz.png](plots/curva_crescimento_gompertz.png) | Ajuste da curva biológica não-linear de Gompertz aos dados de pesagem. |
| [predito_vs_observado_peso.png](plots/predito_vs_observado_peso.png) | Dispersão de peso predito vs peso real do modelo Random Forest ($R^2 = 0,9903$). |
| [boxplots_outliers_peso.png](plots/boxplots_outliers_peso.png) | Boxplots semanais do peso corporal após tratamento de outliers. |
| [matriz_correlacao_features.png](plots/matriz_correlacao_features.png) | Heatmap de correlação de Spearman entre variáveis de lote e peso. |
| [distribuicao_mortalidade.png](plots/distribuicao_mortalidade.png) | Histograma e boxplot da distribuição de mortalidade e taxa de mortalidade (%). |
| [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) | Distribuição normal simétrica dos erros de predição centrada em zero ($\mu = -1,14\text{g}$). |
| [analise_residuos_scatter.png](plots/analise_residuos_scatter.png) | Teste de homocedasticidade (Resíduos vs Valores Preditos). |
| [matriz_confusao_peso.png](plots/matriz_confusao_peso.png) | Matriz de confusão de classificação de atingimento da meta de peso ao abate. |
| [eli5_importancia_variaveis.png](plots/eli5_importancia_variaveis.png) | Barplot do ranking de importâncias ELI5. |

---

## 🧠 4. Grafo de Conhecimento do Repositório (Graphify)

O repositório foi mapeado em um **Grafo de Conhecimento Persistente** via **Graphify**:

* 🕸️ **Visualização Interativa:** [graphify-out/graph.html](graphify-out/graph.html)
* 📄 **Relatório de Audit da Arquitetura:** [graphify-out/GRAPH_REPORT.md](graphify-out/GRAPH_REPORT.md)
* 📊 **Métricas:** 32 Nós, 42 Arestas e 7 Comunidades Mapeadas.

---

## 📁 5. Estrutura do Repositório

```
.
├── .venv/                      # Ambiente virtual Python
├── config/
│   └── settings.py             # Configurações globais e caminhos do projeto
├── data/
│   ├── raw/
│   │   ├── extracao_mtech/     # Arquivos semanais de dados brutos
│   │   └── features/           # BANCO_VARIAVEIS.xlsx
│   └── processed/
│       ├── unified_data.csv    # Base unificada consolidada (104.601 linhas)
│       ├── cleaned_data.csv    # Base limpa sem outliers (88.021 linhas)
│       ├── descriptive_statistics.csv
│       ├── cross_validation_results.csv
│       ├── classification_report.csv
│       └── eli5_feature_importance.csv
├── database/
│   └── prediction_data.db      # Banco de dados SQLite
├── docs/
│   ├── modelo_entidade_relacionamento.md # MER e diagrama Mermaid
│   ├── explicabilidade_eli5.md           # Tabela de importâncias ELI5
│   ├── explicabilidade_eli5.html         # Relatório HTML interativo ELI5
│   ├── premissas.md                      # Premissas do projeto
│   ├── db_schema.sql                     # Esquema exportado do banco
│   └── workflow.md                       # Roteiro e status das fases
├── graphify-out/
│   ├── graph.html              # Grafo de conhecimento interativo HTML
│   ├── graph.json              # Dados estruturados em JSON
│   └── GRAPH_REPORT.md         # Relatório do Grafo de Conhecimento
├── plots/                      # Galeria de gráficos de alta resolução (300 DPI)
├── src/
│   ├── etl/
│   │   ├── extract_mtech_data.py
│   │   ├── extract_excel_to_db.py
│   │   └── export_unified_data.py
│   ├── eda_outliers.py         # Análise exploratória e limpeza de dados
│   ├── models/
│   │   ├── train_predict_weight.py        # Treinamento do modelo Gompertz e ML
│   │   ├── advanced_evaluation_eli5.py    # Resíduos, Confusão, CV e ELI5
│   │   └── saved/                         # Artefatos (.pkl) salvos
│   └── utils/
│       └── logger.py
├── requirements.txt
└── README.md
```

---

## 🛠️ 6. Guia de Execução Passo a Passo

### 1. Configurar o Ambiente Virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Executar o Pipeline de ETL e Unificação
```bash
python3 -m src.etl.extract_excel_to_db
python3 -m src.etl.extract_mtech_data
python3 -m src.etl.export_unified_data
```

### 3. Executar Limpeza de Outliers e Análise Exploratória (EDA)
```bash
python3 -m src.eda_outliers
```

### 4. Treinar os Modelos de Predição de Peso (Gompertz & ML)
```bash
python3 -m src.models.train_predict_weight
```

### 5. Executar Avaliações Avançadas e ELI5
```bash
python3 -m src.models.advanced_evaluation_eli5
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
