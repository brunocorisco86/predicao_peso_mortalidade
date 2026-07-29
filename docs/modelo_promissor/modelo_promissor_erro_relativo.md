---
title: "Documentação do Modelo Campeão: Stacking Ensemble com Erro Relativo (%) por Aviário"
description: "Descrição detalhada da arquitetura, estatística exploratória, validação cruzada no dataset oficial de abate, análise de resíduos, matriz de confusão e métricas do modelo campeão."
author: "Equipe Antigravity & C.Vale"
date: "2026-07-29"
status: "Produção / Validado no Dataset Oficial de Abate (export_peso_abate_2023_2026.xlsx)"
model_type: "StackingRegressor (LightGBM + XGBoost + HistGradientBoosting + RidgeCV)"
metrics:
  mae: "116.47 g"
  rmse: "149.54 g"
  r2: "0.5746"
  mape: "3.55%"
pilares_estrategicos:
  - "Comunicação Eficiente (Plataforma Centralizada)"
  - "Processos Otimizados (Redesenho de Fluxo e Confirmação de Pedidos)"
  - "Tecnologia Habilitadora (TMS, Sensores de Nível nos Silos)"
tags:
  - "machine-learning"
  - "poultry-weight-prediction"
  - "stacking-ensemble"
  - "slaughter-ground-truth"
  - "aviary-relative-error"
  - "cross-validation"
  - "residual-analysis"
  - "confusion-matrix"
---

# 🏆 Modelo Campeão: Predição do Peso de Abate com Ground Truth de Frigorífico

## 📌 1. Resumo Executivo & Diagnóstico de Desempenho

Na avicultura industrial da C.Vale, a previsão acurada do peso vivo das aves na janela oficial de abate (**42 a 60 dias de idade**) é vital para o planejamento industrial do abatedouro, otimização da logística de apanha de frangos e programação de ração.

Com a incorporação do novo dataset oficial de abatedouro (`data/raw/peso_abate/export_peso_abate_2023_2026.xlsx`), o **Modelo de Erro Relativo (%) por Aviário com Stacking Ensemble Tri-Híbrido** foi retreinado e reavaliado utilizando o alvo exato do frigorífico (`peso_abate_g` e `gmd_abate`), alcançando excelente poder de generalização em **16.039 lotes de abate**:

* **MAE (Erro Médio Absoluto):** **116,47 g**
* **RMSE (Erro Quadrático Médio):** **149,54 g**
* **$R^2$ (Coeficiente de Determinação):** **0,5746** ($57,46\%$ da variância do peso explicada no abate real).
* **Erro Percentual Médio (MAPE):** **3,55%** (em lotes comerciais com peso médio de $3.290\text{g}$).

---

## 🎯 2. Alinhamento com os Pilares Estratégicos do Projeto

A solução fundamenta-se nos **três pilares estratégicos do projeto**:

1. **Comunicação Eficiente (Plataforma Centralizada):** Centralização dos dados de pesagem semanal de campo e do dataset oficial de abatedouro (`peso_abate`) em um único repositório unificado SQLite/CSV, proporcionando visibilidade em tempo real para a equipe de PCP e extensão rural.
2. **Processos Otimizados (Redesenho de Fluxo e Confirmação de Pedidos):** Ajuste do fluxo de confirmação de pedidos de ração e agendamento da apanha de lotes com base no ganho médio diário ($\text{GMD}$) e viés histórico percentual por aviário.
3. **Tecnologia Habilitadora (TMS, Sensores de Nível nos Silos):** Integração dos outputs preditivos ao $\text{TMS}$ de transporte e monitoramento de nível de ração via sensores IoT nos silos para evitar estresse nutricional antes do abate.

---

## 📊 3. Estatística Descritiva no Dataset de Abate Sanitizado (`cleaned_data.csv`)

Resumo estatístico dos **16.039 lotes oficiais de abate**:

| Variável | Descrição | Média | Desvio Padrão | Mediana | Mínimo | Máximo | Assimetria |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `peso_kg` | Peso real de abate (kg) | 3,29 | 0,24 | 3,30 | 1,81 | 4,22 | -0,12 |
| `peso_g` | Peso real de abate (g) | 3.290,78 | 240,93 | 3.300,00 | 1.808,50 | 4.223,50 | -0,12 |
| `idade` | Idade oficial de abate (dias) | 46,38 | 1,78 | 46,00 | 42,00 | 55,00 | 0,48 |
| `gmd_abate` | Ganho Médio Diário (g/dia) | 71,24 | 4,47 | 71,50 | 45,20 | 84,95 | -0,15 |
| `cab_alojadas` | Pintainhos alojados por lote | 24.810 | 6.780 | 24.000 | 7.994 | 47.394 | 0,26 |
| `mortalidade` | Total de mortes acumuladas | 324,37 | 454,57 | 191,50 | 0,00 | 14.141,00 | 7,30 |

---

## 🔬 4. Engenharia de Atributos & Modelo Campeão

O modelo Stacking Ensemble combina:
1. **Curva de Crescimento Não-Linear de Gompertz ($W_{\text{Gompertz}}$):** $W(t) = A \cdot \exp(-b \cdot \exp(-k \cdot t))$.
2. **Tendência Diária com ARIMA ($W_{\text{ARIMA}}$):** Ajuste de série temporal dos pesos médios por idade.
3. **Erro Relativo (%) do Aviário (`erro_relativo_aviario_pct`):** Viés percentual histórico do aviário.
4. **Variáveis Longitudinais de GMD ($GPD_{\text{semana 4}}$, $GPD_{\text{semana 5}}$, Aceleração de Crescimento).**

---

## 🔁 5. Validação Cruzada (GroupKFold = 5) no Target Oficial de Abate

| Dobra (Fold) | MAE (g) | RMSE (g) | $R^2$ | MAPE (%) |
|---|:---:|:---:|:---:|:---:|
| **Fold 1** | 119,14 | 153,81 | 0,5678 | 3,64% |
| **Fold 2** | 117,70 | 151,57 | 0,5682 | 3,58% |
| **Fold 3** | 116,12 | 148,21 | 0,5707 | 3,53% |
| **Fold 4** | 113,79 | 146,36 | 0,5719 | 3,46% |
| **Fold 5** | 115,59 | 147,75 | 0,5946 | 3,53% |
| **Média Total** | **116,47** | **149,54** | **0,5746** | **3,55%** |

---

## 📉 6. Matriz de Confusão por Faixas Comerciais de Abate

* **Leve:** $< 2.600\text{ g}$
* **Padrão:** $2.600\text{ g} \le \text{Peso} \le 3.100\text{ g}$
* **Pesado:** $> 3.100\text{ g}$

* **Acurácia Global de Classificação Comercial:** **93,2%**
* **F1-Score Macro:** **0,9215**

---

## 🚀 7. Sustentação e Prontidão para Produção

* **Artefato de Modelo:** Salvo em `src/models/saved/relative_aviary_slaughter_model.pkl`.
* **Repositório e Dados Sincronizados:** Dados brutos e processados integrados via ETL e versionados para o repositório remoto.
