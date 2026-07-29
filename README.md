# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Business Rules](https://img.shields.io/badge/Regras%20de%20Neg%C3%B3cio-Abate%20%E2%89%A5%2042%20dias-red.svg)](docs/regras_de_negocio_abate.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução refatorada dedicada exclusivamente à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias). A análise desconsidera o crescimento inicial em idades jovens para focar na variabilidade comercial do peso final ao abate.

---

## 📌 1. Principais Resultados e Métricas Otimizadas (Fine-Tuning)

Após **Engenharia de Features** (criando percentuais de perda `mortalidade_pct`, `descartados_pct`, `taxa_perda_total` e suporte a linhagens `c16`/fornecedores `c17`) e **Otimização de Hiperparâmetros (RandomizedSearchCV)** via **5-Fold GroupKFold Cross-Validation**, obteve-se uma redução consistente no erro preditivo:

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Redução de Erro / Ganho |
|---|---|---|---|---|
| **Random Forest Baseline (Abate)** | 0,2925 | 127,15 g | 170,56 g | Modelo Inicial |
| **Extra Trees Regressor** | 0,3148 | 121,49 g | 167,42 g | Algoritmo Alternativo |
| **HistGradientBoosting Otimizado (Tuned)** | **0,3694** | **119,26 g** | **160,63 g** | **Melhoria de ~10g no RMSE e MAE < 120g** |
| **Classificador de Meta de Peso de Abate** | **98,4% (Acurácia)** | **F1-Score: 0,98** | **Precision/Recall: 0,98** | Categorias: `Abaixo`, `Na Meta`, `Acima` |

### ⚙️ Hiperparâmetros Selecionados (`HistGradientBoosting`):
```json
{
  "learning_rate": 0.1,
  "max_depth": 6,
  "max_iter": 300,
  "min_samples_leaf": 10,
  "l2_regularization": 1.0
}
```

---

## 📋 2. Regras de Negócio Implementadas

As regras de negócio foram formalizadas no documento [`docs/regras_de_negocio_abate.md`](docs/regras_de_negocio_abate.md):

* **RN-01 (Janela de Abate):** Filtragem estrita de lotes com idade $42 \le \text{idade} \le 60\text{ dias}$ ($15.416$ registros de abate analisados).
* **RN-02 (Faixa Comercial de Peso):** Filtro biológico de peso no abate $1,80\text{ kg} \le \text{peso} \le 4,80\text{ kg}$ ($1.800\text{g}$ a $4.800\text{g}$).
* **RN-05 (Filtro IQR por Idade de Abate):** Remoção de anomalias extremas via $3,0 \times \text{IQR}$ especificamente para cada dia de abate.

---

## 🔬 3. Diagnóstico e Explicabilidade ELI5 no Abate

Com a neutralização do impacto da idade inicial, as variáveis operacionais e genéticas do lote passam a ditar o peso final ao abate.

### Ranking de Importância de Variáveis ELI5 (Peso de Abate):

| Rank | Variável | Descrição Zootécnica / Operacional | Importância Média ELI5 | Desvio Padrão |
|---|---|---|---|---|
| 1 | `c15` | Peso inicial do pintainho de 1 dia (g) | **23,20%** | $\pm 1,47\%$ |
| 2 | `mortalidade` | Desafio sanitário e mortes acumuladas | **14,75%** | $\pm 1,42\%$ |
| 3 | `cab_alojadas` | Densidade e quantidade alojada no lote | **13,03%** | $\pm 1,37\%$ |
| 4 | `x02` | Distância da propriedade ao abatedouro (km) | **11,73%** | $\pm 1,31\%$ |
| 5 | `idade` | Variação diária entre os dias de abate (42 a 54 dias) | **10,99%** | $\pm 1,76\%$ |
| 6 | `descartados` | Descartes sanitários no lote | **9,34%** | $\pm 1,25\%$ |
| 7 | `c12` | Fator multiplicador de peso aos 35 dias acima | **8,40%** | $\pm 0,85\%$ |
| 8 | `c11` | Fator multiplicador de peso aos 35 dias abaixo | **2,26%** | $\pm 0,58\%$ |

* 📄 Relatório em Markdown: [docs/explicabilidade_eli5.md](docs/explicabilidade_eli5.md)
* 🌐 Relatório HTML Interativo: [docs/explicabilidade_eli5.html](docs/explicabilidade_eli5.html)
* 📊 Gráfico de Importâncias ELI5: [plots/eli5_importancia_variaveis.png](plots/eli5_importancia_variaveis.png)

---

## 📈 4. Galeria de Gráficos Refatorados (Peso de Abate)

| Gráfico | Descrição do Diagnóstico de Abate |
|---|---|
| [distribuicao_peso_por_idade.png](plots/distribuicao_peso_por_idade.png) | Boxplots da distribuição do peso corporal de abate para cada dia (42 a 54 dias). |
| [curva_crescimento_gompertz.png](plots/curva_crescimento_gompertz.png) | Ajuste da curva de Gompertz na janela comercial de abate. |
| [predito_vs_observado_peso.png](plots/predito_vs_observado_peso.png) | Dispersão do modelo Fine-Tuned (MAE de $119\text{g}$ / RMSE de $160\text{g}$). |
| [boxplots_outliers_peso.png](plots/boxplots_outliers_peso.png) | Avaliação da variabilidade do peso por dia de abate após filtros biológicos. |
| [matriz_correlacao_features.png](plots/matriz_correlacao_features.png) | Correlação de Spearman entre variáveis operacionais e peso de abate. |
| [distribuicao_mortalidade.png](plots/distribuicao_mortalidade.png) | Distribuição da mortalidade acumulada no momento do abate. |
| [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) | Distribuição dos erros residuais de predição do peso de abate. |
| [analise_residuos_scatter.png](plots/analise_residuos_scatter.png) | Teste de homocedasticidade para a predição do peso de abate. |
| [matriz_confusao_peso.png](plots/matriz_confusao_peso.png) | Matriz de confusão para atingimento da meta de peso no abate. |
| [eli5_importancia_variaveis.png](plots/eli5_importancia_variaveis.png) | Barplot do ranking de importâncias ELI5 no momento do abate. |

---

## 🧠 5. Grafo de Conhecimento do Repositório (Graphify)

* 🕸️ **Visualização Interativa:** [graphify-out/graph.html](graphify-out/graph.html)
* 📄 **Relatório de Audit da Arquitetura:** [graphify-out/GRAPH_REPORT.md](graphify-out/GRAPH_REPORT.md)

---

## 📁 6. Estrutura do Repositório

```
.
├── .venv/                      # Ambiente virtual Python
├── config/
│   └── settings.py
├── data/processed/             # Datasets filtrados de abate e métricas
├── database/
│   └── prediction_data.db      # Banco de dados SQLite
├── docs/
│   ├── regras_de_negocio_abate.md # Regras de Negócio (idade >= 42 dias)
│   ├── modelo_entidade_relacionamento.md
│   ├── explicabilidade_eli5.md
│   ├── explicabilidade_eli5.html
│   ├── premissas.md
│   └── workflow.md
├── graphify-out/               # Artefatos Graphify
├── plots/                      # Galeria de gráficos de abate
├── src/
│   ├── etl/
│   ├── eda_outliers.py         # Filtro e EDA de abate (idade >= 42)
│   ├── models/
│   │   ├── train_predict_weight.py        # Treino do modelo de abate
│   │   ├── advanced_evaluation_eli5.py    # Resíduos, Confusão, CV e ELI5
│   │   ├── fine_tune_slaughter_model.py   # Fine-tuning & comparativo de modelos
│   │   └── saved/
│   └── utils/
├── requirements.txt
└── README.md
```

---

## 🛠️ 7. Guia de Execução Passo a Passo

```bash
# 1. Configurar Ambiente
source .venv/bin/activate

# 2. Executar EDA e Filtro de Abate (RN-01 a RN-05)
python3 -m src.eda_outliers

# 3. Treinar Modelo de Predição do Peso de Abate
python3 -m src.models.train_predict_weight

# 4. Executar Fine-Tuning de Hiperparâmetros
python3 -m src.models.fine_tune_slaughter_model

# 5. Executar Avaliações Avançadas de Abate e ELI5
python3 -m src.models.advanced_evaluation_eli5
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
