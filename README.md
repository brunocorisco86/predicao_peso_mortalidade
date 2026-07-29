# Prediction of Mortality & Weight in Poultry Production

Este repositório contém a solução completa de engenharia de dados, análise exploratória (EDA), tratamento de outliers, modelagem preditiva de peso corporal de frangos de corte, além de avaliações avançadas (**Resíduos**, **Matriz de Confusão**, **GroupKFold Cross-Validation** e **Explicabilidade ELI5**) e integração com a logística de ração.

---

## 📌 Desempenho e Validação Cruzada (5-Fold GroupKFold)

A validação cruzada agrupada por lote de produção (`lote_composto`) garante a ausência de vazamento de dados (*data leakage*) entre treino e teste.

| Modelo / Validação | $R^2$ Médio | MAE Médio (g) | RMSE Médio (g) |
|---|---|---|---|
| **Curva Não-Linear de Gompertz** | **0,9848** | **81,81 g** | **126,81 g** |
| **Random Forest (Validação Cruzada 5 Folds)** | **0,9897 ± 0,0004** | **67,70 g ± 0,81 g** | **105,55 g ± 2,08 g** |

---

## 🔬 Avaliações Avançadas e Explicabilidade

1. **Análise de Resíduos:**
   - Resíduos ($y_{\text{observado}} - y_{\text{predito}}$) apresentam distribuição simétrica normal centrada em zero ($\mu = -1,14\text{ g}$) e homocedasticidade confirmada ([analise_residuos_histograma.png](file:///home/brunoconter/Documentos/1_C.VALE/1%20-%20ANALISES/10%20-%20PESO%20DAS%20AVES/prediction_weight_mortality/plots/analise_residuos_histograma.png), [analise_residuos_scatter.png](file:///home/brunoconter/Documentos/1_C.VALE/1%20-%20ANALISES/10%20-%20PESO%20DAS%20AVES/prediction_weight_mortality/plots/analise_residuos_scatter.png)).
2. **Matriz de Confusão:**
   - Classificação do lote quanto ao atingimento da meta de peso de abate (`Abaixo da Meta`, `Na Meta`, `Acima da Meta`) com alta acurácia ([matriz_confusao_peso.png](file:///home/brunoconter/Documentos/1_C.VALE/1%20-%20ANALISES/10%20-%20PESO%20DAS%20AVES/prediction_weight_mortality/plots/matriz_confusao_peso.png)).
3. **Explicabilidade ELI5:**
   - Relatório detalhado das variáveis mais importantes gerado em [`docs/explicabilidade_eli5.md`](docs/explicabilidade_eli5.md) e visualizado em [eli5_importancia_variaveis.png](file:///home/brunoconter/Documentos/1_C.VALE/1%20-%20ANALISES/10%20-%20PESO%20DAS%20AVES/prediction_weight_mortality/plots/eli5_importancia_variaveis.png).
   - **Variáveis Principais:** `idade` (99,2% da variação), `c15` (peso inicial do pintainho), `c12` (fator de peso a 35d), `mortalidade`, `cab_alojadas` e `x02` (distância do abatedouro).

---

## 🏗️ Estrutura do Repositório

```
.
├── data/processed/
│   ├── unified_data.csv
│   ├── cleaned_data.csv
│   ├── cross_validation_results.csv
│   ├── classification_report.csv
│   └── eli5_feature_importance.csv
├── docs/
│   ├── modelo_entidade_relacionamento.md
│   ├── explicabilidade_eli5.md
│   ├── explicabilidade_eli5.html
│   ├── premissas.md
│   └── workflow.md
├── plots/
│   ├── distribuicao_peso_por_idade.png
│   ├── boxplots_outliers_peso.png
│   ├── distribuicao_mortalidade.png
│   ├── matriz_correlacao_features.png
│   ├── curva_crescimento_gompertz.png
│   ├── predito_vs_observado_peso.png
│   ├── analise_residuos_histograma.png
│   ├── analise_residuos_scatter.png
│   ├── matriz_confusao_peso.png
│   └── eli5_importancia_variaveis.png
├── src/
│   ├── etl/
│   ├── eda_outliers.py
│   └── models/
│       ├── train_predict_weight.py
│       └── advanced_evaluation_eli5.py
└── README.md
```

---

## 🚚 Solução para Falhas na Entrega de Ração (3 Pilares)

1. **Comunicação Eficiente (Plataforma Centralizada):** Portal integrador para compartilhamento das curvas de demanda estimadas pelo modelo.
2. **Processos Otimizados (Redesenho de Fluxo e Confirmação de Pedidos):** Automação do agendamento de entregas correlacionado à curva de crescimento.
3. **Tecnologia Habilitadora (TMS e Sensores de Nível nos Silos):** Roteamento inteligente via TMS acoplado à medição telemétrica dos silos em tempo real.
