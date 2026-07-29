---
title: "Documentação do Modelo Campeão: Stacking Ensemble com Erro Relativo (%) por Aviário"
description: "Descrição detalhada da arquitetura, engenharia de dados, métricas e implementação do modelo mais promissor para predição do peso de abate em frangos de corte."
author: "Equipe Antigravity & C.Vale"
date: "2026-07-29"
status: "Produção / Validado em CV GroupKFold"
model_type: "StackingRegressor (LightGBM + XGBoost + HistGradientBoosting + RidgeCV)"
metrics:
  mae: "92.11 g"
  rmse: "130.86 g"
  r2: "0.5814"
  mape: "~3.0%"
pilares_estrategicos:
  - "Comunicação Eficiente (Plataforma Centralizada)"
  - "Processos Otimizados (Redesenho de Fluxo e Confirmação de Pedidos)"
  - "Tecnologia Habilitadora (TMS, Sensores de Nível nos Silos)"
tags:
  - "machine-learning"
  - "poultry-weight-prediction"
  - "stacking-ensemble"
  - "gompertz"
  - "arima"
  - "aviary-relative-error"
---

# 🏆 Modelo Campeão: Predição do Peso de Abate de Frangos de Corte

## 📌 1. Resumo Executivo & Diagnóstico de Desempenho

No processo de avicultura industrial de corte, a previsão acurada do peso corporal das aves na janela de abate (**42 a 60 dias de idade**) é essencial para o planejamento do frigorífico, alocação de caminhões de apanha e otimização logística. 

Após a avaliação de 5 iterações metodológicas, o **Modelo de Erro Relativo (%) por Aviário com Stacking Ensemble Tri-Híbrido** consagrou-se como o modelo mais promissor do projeto, atingindo o **recorde histórico de desempenho**:

* **MAE (Erro Médio Absoluto):** **92,11 g** (uma redução expressiva em relação à baseline inicial de 118,80 g).
* **RMSE (Erro Quadrático Médio):** **130,86 g**
* **Métrica $R^2$ (Coeficiente de Determinação):** **0,5814** ($58,14\%$ da variância do peso explicada).
* **Erro Percentual Médio (MAPE):** **~3,0%** (em aves de $3.000\text{g}$).

---

## 🎯 2. Alinhamento com os Pilares Estratégicos do Projeto

A solução proposta para contornar gargalos e otimizar as entregas e previsões na cadeia avícola fundamenta-se em **três pilares estratégicos**:

1. **Comunicação Eficiente (Plataforma Centralizada):** Integração dos dados de pesagem amostral semanal de campo, relatórios de lote e dados do abatedouro em um único fluxo de dados centralizado, permitindo visibilidade em tempo real para zootecnistas e equipes do PCP (Planejamento e Controle da Produção).
2. **Processos Otimizados (Redesenho de Fluxo e Confirmação de Pedidos):** Redesenho da janela de confirmação de pedidos de ração e escalonamento de apanha de lote com base nas projeções de ganho de peso diário ($\text{GPD}$) e acurácia preditiva dos aviários.
3. **Tecnologia Habilitadora (TMS, Sensores de Nível nos Silos):** Uso de sistemas de gerenciamento de transporte ($\text{TMS}$) sincronizados às previsões de peso por lote e sensores IoT instalados nos silos de ração para evitar desabastecimento e estresse nutricional nas semanas finais de engorda ($35\text{ a }45\text{ dias}$).

---

## 🔬 3. Arquitetura do Modelo & Engenharia de Atributos

O modelo campeão combina modelos matemáticos não-lineares, análise de séries temporais, dimensões longitudinais de crescimento e codificação do viés histórico do aviário.

```
                                 ┌─────────────────────────────────┐
                                 │   Dados Históricos de Lotes    │
                                 └────────────────┬────────────────┘
                                                  │
         ┌────────────────────────┬───────────────┴───────────────┬────────────────────────┐
         │                        │                               │                        │
         ▼                        ▼                               ▼                        ▼
┌──────────────────┐    ┌──────────────────┐           ┌──────────────────┐     ┌─────────────────────┐
│  Curva Gompertz  │    │   Série ARIMA    │           │  Dimensão Aviário│     │Features Longitudinais│
│  (Crescimento   │    │  (Médias Diárias)│           │ (Erro Relat. %)  │     │(Pesagens 21,28,35d) │
│  Não-Linear)     │    │                  │           │                  │     │ & GPD / Aceleração  │
└────────┬─────────┘    └────────┬─────────┘           └────────┬─────────┘     └──────────┬──────────┘
         │                        │                               │                        │
         └────────────────────────┴───────────────┬───────────────┴────────────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │      Vetor de Atributos (X)     │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │   Stacking Ensemble Regressor   │
                                 │ (LightGBM + XGBoost + HistGBR)  │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │   Meta-Modelo RidgeCV (Linear)  │
                                 └────────────────┬────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │   Peso Estimado no Abate (g)    │
                                 └─────────────────────────────────┘
```

### Componentes Principais de Engenharia de Atributos:

1. **Curva de Crescimento Não-Linear de Gompertz ($W_{\text{Gompertz}}$):**
   $$W(t) = A \cdot \exp\left(-b \cdot \exp(-k \cdot t)\right)$$
   Onde os parâmetros calibrados com dados de campo foram $A = 6.260,16\text{g}$, $b = 4,7378$, $k = 0,0449$.

2. **Tendência Diária Temporal com ARIMA ($W_{\text{ARIMA}}$):**
   Ajuste de modelo $\text{ARIMA}(1,1,1)$ na série diária de pesos médios para capturar variações sazonais e de tendência de curto prazo por idade.

3. **Correção por Erro Relativo (%) por Aviário (`erro_relativo_aviario_pct`):**
   Em vez de apenas calcular a diferença em gramas ($\Delta g$), calculamos a variação percentual relativa histórica de cada aviário em relação ao comportamento teórico de Gompertz:
   $$\text{Erro Relativo}(\%) = \frac{W_{\text{observado}} - W_{\text{Gompertz}}}{W_{\text{Gompertz}}} \times 100$$
   Esta variável captura o perfil tecnológico do aviário (ex: climatizado vs tradicional, isolamento térmico, eficiência de comedouros) de forma invariante ao dia da medição.

4. **Features Longitudinais de Ganho de Peso Diário ($\text{GPD}$):**
   * $\text{GPD}_{\text{semana 4}} = \frac{W_{28d} - W_{21d}}{7}$
   * $\text{GPD}_{\text{semana 5}} = \frac{W_{35d} - W_{28d}}{7}$
   * $\text{Aceleração} = \frac{\text{GPD}_{\text{semana 5}}}{\text{GPD}_{\text{semana 4}}}$
   * $\text{Peso Projetado GPD} = W_{35d} + (\text{idade} - 35) \times \text{GPD}_{\text{semana 5}}$

---

## 📊 4. Tabela Comparativa da Evolução dos Modelos

| Fase / Modelo | MAE (g) | RMSE (g) | Métrica $R^2$ | Principal Inovação Metodológica |
|---|:---:|:---:|:---:|---|
| **1. Modelo Estático Inicial** | 118,80 | 160,74 | 0,3684 | Baseline com atributos estáticos do lote |
| **2. Modelo Longitudinal** | 96,92 | 137,79 | 0,5359 | Inclusão de pesagens intermediárias aos 21d, 28d e 35d |
| **3. Modelo Tri-Híbrido** | 96,15 | 137,16 | 0,5401 | Integração de Gompertz + ARIMA + Stacking ML |
| **4. Dimensão Delta (g) Aviário** | 92,57 | 131,54 | 0,5770 | Mapeamento do viés absoluto fixo em gramas por aviário |
| **5. Erro Relativo (%) Aviário** | **92,11** | **130,86** | **0,5814** | 🏆 **Recorde Histórico: Viés percentual invariante por aviário** |

---

## 🔍 5. Explicabilidade e Importância das Variáveis

Através do algoritmo LightGBM e análises de explicabilidade ELI5, identificou-se que as variáveis com maior peso na tomada de decisão do modelo são:

1. **`peso_dia_35` & `peso_projetado_gpd`:** Determinantes diretos do peso final ao abate.
2. **`erro_relativo_aviario_pct`:** Elevado ganho de informação ao caracterizar a capacidade produtiva única do aviário.
3. **`idade`:** Fator biológico de maturação.
4. **`w_gompertz` & `w_arima`:** Linha de base biológica e temporal.
5. **`c15` (Peso inicial do pintainho):** Forte correlação com a arrancada de ganho de peso na 1ª semana.

---

## 💻 6. Trecho de Código do Modelo Campeão

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Função Biológica Gompertz
def gompertz_func(t, A=6260.16, b=4.7378, k=0.0449):
    return A * np.exp(-b * np.exp(-k * t))

# 2. Definição do Stacking Ensemble
tuned_lgb = lgb.LGBMRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
tuned_xgb = xgb.XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.05, subsample=0.85, random_state=42, n_jobs=-1)
tuned_hgb = HistGradientBoostingRegressor(max_iter=400, max_depth=7, learning_rate=0.05, random_state=42)

estimators = [('lgb', tuned_lgb), ('xgb', tuned_xgb), ('hgb', tuned_hgb)]
champion_stacking = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)

# 3. Validação Cruzada por Grupo de Lotes (GroupKFold)
gkf = GroupKFold(n_splits=5)
# Realiza o treinamento garantindo que amostras do mesmo lote não estejam no treino e teste simultaneamente.
```

---

## 🚀 7. Próximos Passos & Metas Futuras

* **Meta Sub-80g (Erro $< 2,5\%$):** Incorporar dados de telemetria IoT de água/ração e índice de estresse térmico ($\text{ITU}$).
* **Integração Logística:** Disponibilizar os outputs preditivos na **plataforma centralizada** para sincronização com o $\text{TMS}$ de apanha de frango e programação do PCP.
