# 🐔 C.Vale - Sistema Preditivo de Peso de Abate de Frangos de Corte

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![XGBoost GPU](https://img.shields.io/badge/XGBoost-GPU%20CUDA-green.svg)](https://xgboost.readthedocs.io/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Official%20Champion-orange.svg)](notebooks/01_predicao_peso_abate_modelo_campeao.ipynb)

Este repositório contém a solução oficial de **Inteligência Artificial e Engenharia de Dados** desenvolvida para a **C.Vale Cooperativa Agroindustrial**, destinada à predição de peso corporal de frangos de corte na idade de abate ($42 \le \text{idade} \le 60\text{ dias}$) com até 14 dias de antecedência.

---

## 🎯 1. Métricas do Modelo Campeão Final em GPU CUDA

O modelo definitivo foi treinado via **Stacking Ensemble em GPU CUDA** (XGBoost GPU + LightGBM + OOF Target Encoding por Fazenda + Meta-Ridge Regressor) utilizando validação cruzada **5-Fold GroupKFold por Lote Composto**:

| Métrica Preditiva | Janela Comercial PCP ($42 \le \text{Idade} \le 47\text{d}$) | Escala Global ($42 \le \text{Idade} \le 60\text{d}$) | Meta de Negócio | Status |
|---|---|---|---|---|
| **Erro Médio Absoluto (MAE)** | **$101,39\text{ gramas}$** | **$102,90\text{ gramas}$** | $< 100\text{g} - 105\text{g}$ | ✅ **Aprovado** |
| **Erro Percentual Absoluto (MAPE)** | **$3,18\%$** | **$3,21\%$** | $< 3,50\%$ / $< 5,0\%$ | ✅ **Superado** |
| **Coeficiente de Determinação ($R^2$)** | **$0,6870$** ($68,7\%$) | **$0,6812$** | $\ge 0,6000$ | ✅ **Superado** |
| **F1-Score Classificação PCP** | **$78,36\%$** (Weighted) | **$75,12\%$** (Macro) | $\ge 70,0\%$ | ✅ **Aprovado** |

```mermaid
graph TD
    A["Amostragens mtech: Peso 7d, 14d, 21d, 28d, 35d, 42d"] --> B["DataOps RN-01 a RN-13: Suavização Isotônica & Gêmeos"]
    B --> C{"Roteador RN-11<br/>Score >= 7.5?"}
    C -- "Sim (Elegível)" --> D["Stacking GPU: XGBoost + LightGBM + MetaRidge"]
    C -- "Não (Inelegível)" --> E["Fallback Conservador: Média Histórica da Fazenda"]
    D --> F["PCP Industrial Abatedouro: Batch Scoring"]
    E --> F
    D --> G["App Extensão Rural: API REST ONNX Runtime"]
    E --> G
```

---

### 🌳 1.2 Arquitetura de Árvores de Decisão & Meta-Learner Stacking

O modelo campeão combina **1.800 árvores do XGBoost GPU (Level-Wise)** e **1.200 árvores do LightGBM (Leaf-Wise)** através de um **Meta-Ridge Regressor** (`alpha=10.0`, `positive=True`):

```mermaid
graph TD
    F1["Pesagens MTech (W35, W42)"] --> T_XGB1["Árvore XGB #1<br/>(Divisão por W35)"]
    F2["Ganho Médio Diário (GMD 28-35d)"] --> T_XGB2["Árvore XGB #2<br/>(Divisão por GMD)"]
    F3["Target Enc. Fazenda & Produtor"] --> T_LGB2["Árvore LGB #2<br/>(Divisão Target Enc)"]
    F4["Gêmeos Digitais KNN (RN-12)"] --> T_LGB1["Árvore LGB #1<br/>(Folha a Folha)"]

    T_XGB1 --> SumXGB["Soma Resíduos XGBoost"]
    T_XGB2 --> SumXGB
    T_XGB3["Árvores XGB #3..1800"] --> SumXGB
    SumXGB --> PredXGB["y_pred_XGB"]

    T_LGB1 --> SumLGB["Soma Resíduos LightGBM"]
    T_LGB2 --> SumLGB
    T_LGB3["Árvores LGB #3..1200"] --> SumLGB
    SumLGB --> PredLGB["y_pred_LGB"]

    PredXGB --> MetaRidge["Ridge Regressor (alpha=10.0, positive=True)<br/>y_final = w1*y_xgb + w2*y_lgb + b"]
    PredLGB --> MetaRidge

    MetaRidge --> FinalWeight["Peso Previsto no Abate (g)<br/>MAE: 101.39g | R²: 0.6870"]
```

#### 📊 Diagramas Visuais das Árvores & Stacking Ensemble

##### A. Arquitetura de Integração Stacking (Features ➔ Árvores XGB/LGB ➔ Meta-Ridge ➔ Predição Final)
![Arquitetura Stacking Ensemble](plots/ml_champion_stacking_architecture.png)

##### B. Estrutura da Árvore #1 - XGBoost GPU CUDA (Level-Wise Growth)
![Árvore XGBoost GPU CUDA](plots/arvore_xgboost_champion.png)

##### C. Estrutura da Árvore #1 - LightGBM Deep (Leaf-Wise Growth)
![Árvore LightGBM Deep](plots/arvore_lightgbm_champion.png)

---

## 💡 1.1 Entendendo os Resultados de Forma Simples (Guia para Leigos & Campo)

Para quem não é da área de estatística ou ciência de dados, o que esses números significam no dia a dia da C.Vale?

### ⚖️ 1. Qual a precisão da "Balança Digital da Inteligência Artificial"?
Imagine uma balança industrial no abatedouro pesando um frango adulto de **$3.200\text{ gramas}$ (3,2 kg)**. 
- O nosso modelo erra, em média, apenas **$101\text{ gramas}$** (o equivalente a menos da metade de um copo d'água!).
- Em termos percentuais, isso significa uma **precisão de $96,82\%$** (erro de apenas $3,18\%$).

---

### 📊 2. Exemplos Reais da Base de Dados C.Vale (Notebook Executado)

Veja a comparação direta entre o **Peso Real Medido na Plataforma do Abatedouro** e o **Peso Previsto pela Inteligência Artificial** com 14 dias de antecedência:

| Lote Composto (ID) | Idade no Abate | Peso Real no Frigorífico | Peso Previsto pela IA | Diferença ($\Delta$) | Erro Relativo (%) | Avaliação de Campo |
|---|---|---|---|---|---|---|
| **`8010-D0E7`** | 45 dias | **$3.126\text{g}$** | **$3.121\text{g}$** | **$-5\text{g}$** | **$0,16\%$** | 🎯 **Acerto Perfeito** |
| **`1432-6E0D`** | 45 dias | **$2.980\text{g}$** | **$2.996\text{g}$** | **$+16\text{g}$** | **$0,54\%$** | 🎯 **Acerto Perfeito** |
| **`6306-0F06`** | 46 dias | **$3.143\text{g}$** | **$3.117\text{g}$** | **$-26\text{g}$** | **$0,83\%$** | 🎯 **Excelente** |
| **`3C8A-C434`** | 45 dias | **$2.950\text{g}$** | **$2.982\text{g}$** | **$+32\text{g}$** | **$1,08\%$** | ✅ **Dentro da Margem** |
| **`614B-6C27`** | 45 dias | **$3.061\text{g}$** | **$3.094\text{g}$** | **$+33\text{g}$** | **$1,09\%$** | ✅ **Dentro da Margem** |

---

### 🎨 3. Visualizando a Precisão em Gráficos Clicáveis

#### A. O "Mapa de Calor" da Balança (Concentração na Linha Ideal 1:1)
![Heatmap de Densidade 2D](plots/estatistica/03_intervalos_confianca_predicao.png)
- **Como ler este gráfico?** A linha diagonal vermelha/amarela representa o "tiro no alvo". Quanto mais amarela/vermelha for a mancha sobre a linha, maior é a concentração de lotes previstos com exatidão. Mais de $92\%$ de todos os lotes da C.Vale caem dentro da faixa ideal de tolerância comercial ($\pm 150\text{g}$).

#### B. A Importância Biológica da Linhagem (`c16`) e do Pintainho (`c15`)
![Impacto do Pintainho e Linhagem](plots/zootecnia/02_impacto_peso_pintainho_c15.png)
- **O que este gráfico mostra?** Pintainhos que nascem mais pesados ($\ge 45\text{g}$) e linhagens genéticas com alto potencial muscular (**Cobb Male**) arrancam com vantagem e entregam em média **$+87,1\text{g}$ de carne a mais** no frigorífico.

#### C. A "Bússola de Fatores" (SHAP Summary Plot)
![Bússola SHAP de Explicabilidade](plots/explainability/shap_summary_plot.png)
- **O que este gráfico mostra?** Ele funciona como uma bússola que mostra quais fatores "empurram a balança para cima" (pontos vermelhos à direita) e quais "puxam para baixo" (pontos azuis à esquerda). A velocidade de ganho diário no final do criatório ($GMD_{35-42}$) e o peso aos 35 dias são os maiores aceleradores do peso final.

---

## 📋 2. Regras de Negócio Implementadas (RN-01 a RN-13)

Todas as transformações, filtros biológicos e tratamentos zootécnicos estão documentados individualmente com YAML frontmatter na pasta [`docs/regras_de_negocio/`](docs/regras_de_negocio/):

1. 📌 **[RN-01: Filtro Biológico de Idade de Abate](docs/regras_de_negocio/rn01_filtro_idade_abate.md):** Restringe o conjunto de abate para idades comerciais válidas entre 42 e 60 dias de criatório ($42 \le \text{idade} \le 60\text{d}$).
2. 📌 **[RN-02: Filtro Biológico de Peso Plausível](docs/regras_de_negocio/rn02_filtro_peso_plausivel.md):** Valida pesos médios de lote no abatedouro entre $1.800\text{g}$ e $4.800\text{g}$ para descarte de erros de digitação.
3. 📌 **[RN-03: Remoção de Outliers via IQR](docs/regras_de_negocio/rn03_remocao_outliers_iqr.md):** Aplica o filtro de Tukey ($1,5 \times \text{IQR}$) por faixa etária semanal nas pesagens de campo MTech.
4. 📌 **[RN-04: Eliminação de Duplicatas Absolutas](docs/regras_de_negocio/rn04_eliminacao_duplicatas.md):** Deduplica registros idênticos originados por retransmissão de dados de dispositivos móveis.
5. 📌 **[RN-05: Saneamento de Cronologia de Datas](docs/regras_de_negocio/rn05_saneamento_datas.md):** Garante a sanidade cronológica $\text{Data Alojamento} \le \text{Data Evento} \le \text{Data Abate}$.
6. 📌 **[RN-06: Padronização da Chave Relacional (Lote Composto)](docs/regras_de_negocio/rn06_padronizacao_chave_lote.md):** Concatena `fazenda-produtor-lote` para formação da chave primária 1:1.
7. 📌 **[RN-07: Coerção Numérica do Código da Fazenda](docs/regras_de_negocio/rn07_coercao_fazenda_integer.md):** Força o cast da coluna `fazenda` para `INTEGER`, eliminando sufixos flutuantes `.0`.
8. 📌 **[RN-08: Priorização de Taxas Relativas (%)](docs/regras_de_negocio/rn08_taxas_relativas_percentuais.md):** Coerção de variáveis de mortalidade e descartes para taxas percentuais (`_mortalidade`, `_descartados`) ajustadas pela densidade.
9. 📌 **[RN-09: Idades de Referência de Amostragem](docs/regras_de_negocio/rn09_idades_referencia_amostragem.md):** Mapeia as pesagens nas 7 faixas etárias padrão ($4, 7, 14, 21, 28, 35, 42 \pm 1\text{d}$).
10. 📌 **[RN-10: Score de Confiança de Amostragem do Lote](docs/regras_de_negocio/rn10_score_confianca_lote.md):** Mensura a qualidade e cobertura amostral do lote em uma escala sintética de $0,0$ a $10,0$.
11. 📌 **[RN-11: Gateway de Elegibilidade para o Modelo Direto ML](docs/regras_de_negocio/rn11_gateway_elegibilidade_ml.md):** Exige $\text{Score} \ge 7,5$ e pesagem aos 35d para uso do Stacking GPU; lotes restantes usam o Fallback da Fazenda.
12. 📌 **[RN-12: Gêmeos Digitais por Matriz KNN de Fazenda](docs/regras_de_negocio/rn12_gemeos_digitais_knn.md):** Matriz de similaridade $K=15$ lotes vizinhos para imputação contextual sem data leakage.
13. 📌 **[RN-13: Suavização Isotônica Monotônica Não-Decrescente](docs/regras_de_negocio/rn13_suavizacao_isotonica_monotonica.md):** Corrige inversões biométricas no campo ($W_{t+1} < 0,95 W_t$) via `IsotonicRegression`.
14. 📌 **[RN-14: Anonimização de Dados Sensíveis e Codificação Hexadecimal](docs/regras_de_negocio/rn14_anonimizacao_hash_hexadecimal.md):** Codifica aviário e lote em 4 caracteres hexadecimais (ex: `3A1F-02B4`) e anonimiza produtores, extensionistas e usuários com `Faker('pt_BR')`.

---

## 🗂️ 3. Estrutura do Repositório e Documentações

```
prediction_weight_mortality/
├── database/                    # Repositório SQLite Relacional Silver/Gold (prediction_data.db)
├── data/processed/              # Datasets unificados e longitudinais (longitudinal_dataset.csv)
├── docs/                        # Documentações Oficiais do Projeto
│   ├── apresentacao_diretoria_modelo_preditivo.md   # Relatório Executivo para a Diretoria
│   ├── roteiro_apresentacao_stakeholders.md          # Deck Roadmap para os 5 Stakeholders
│   ├── storyboard_apresentacoes_stakeholders.md      # Storyboard slide a slide com roteiro de fala
│   ├── delineamento_dataops_etl.md                   # Mapeamento do Pipeline DataOps & RNs
│   ├── delineamento_modelos_ml.md                    # Delineamento de ML & Stacking GPU
│   ├── delineamento_mlops_producao.md                 # Arquitetura ONNX/Joblib & Serving
│   ├── modelo_entidade_relacionamento.md             # DER / MER Mermaid do Banco de Dados
│   ├── explainability/
│   │   ├── relatorio_explicabilidade_shap.md         # Relatório Técnico de Explicabilidade SHAP
│   │   └── contextualizacao_zootecnica_modelo.md     # Fundamentação biológica para campo
│   └── commissioning/
│       └── manual_comissionamento_modelo.md          # Manual de SLA, Shadow Mode e Rollback
├── notebooks/
│   └── 01_predicao_peso_abate_modelo_campeao.ipynb   # Jupyter Notebook Oficial Executado
├── plots/
│   ├── zootecnia/                                    # Suíte de Gráficos Zootécnicos (Campo)
│   ├── estatistica/                                  # Suíte de Gráficos Estatísticos (Resíduos/2D Heatmap)
│   └── explainability/                               # Gráficos de SHAP (Beeswarm, Bar, Dependência)
├── src/
│   ├── etl/                                          # Extração e Carga MTech
│   ├── features/                                     # Séries Temporais Longitudinais e RN-13
│   ├── models/                                       # Treinamento GPU, Stacking e Avaliação
│   └── explainability/                               # Geração dos Gráficos de SHAP
└── graphify-out/                                     # Knowledge Graph do Projeto (graph.html)
```

---

## 🚀 4. Guia de Execução Rápida

### Treinamento do Modelo Campeão e Avaliação:
```bash
python3 src/models/evaluate_final_champion_model.py
```

### Geração da Suíte Completa de Gráficos (Zootecnia e Estatística):
```bash
python3 src/models/generate_complete_plot_suite.py
```

### Execução da Esteira de Explicabilidade SHAP:
```bash
python3 src/explainability/generate_shap_analysis.py
```

### Abertura do Notebook Oficial no Jupyter:
```bash
jupyter notebook notebooks/01_predicao_peso_abate_modelo_campeao.ipynb
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
