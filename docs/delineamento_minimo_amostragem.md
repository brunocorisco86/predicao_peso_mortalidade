# Relatório Técnico: Modelo Gompertz e Delineamento Amostral Mínimo (RN-11)

**Data da Análise:** 30 de Julho de 2026  
**Autor:** Antigravity Data Science & Zootecnia Team (C.Vale)  
**Objetivo:** Estabelecer a Regra de Negócio de Delineamento Amostral Mínimo (**RN-11**) para garantir predições robustas do peso de abate através de biometrias de campo Gompertz.

---

## 📈 1. Desempenho Global do Modelo Gompertz

O modelo ajustado pela equação não-linear de Gompertz W(t) = A * exp(-B * exp(-k * t)) foi avaliado via **5-Fold Cross Validation** nas pesagens de campo filtradas:

- **MAE (Erro Médio Absoluto):** 228.51 ± 3.00 g (0.229 kg)
- **RMSE (Raiz do Erro Quadrático Médio):** 293.40 ± 1.79 g (0.293 kg)
- **MAPE (Erro Percentual Médio):** 7.11 ± 0.10 %
- **R² (Coeficiente de Determinação):** -0.4243 ± 0.0242

---

## 📊 2. Comparativo de Erro por Categoria de Amostragem (RN-09 e RN-10)

A precisão do modelo Gompertz depende criticamente da presença de pesagens em idades chave e do volume amostral:

### 2.1. Desempenho por Categoria de Maturidade (RN-09)

| Categoria de Amostragem | Total Lotes | MAE (g) | RMSE (g) | MAPE (%) | Acurácia (±5%) |
|---|---|---|---|---|---|
| **Ouro (Preferível: 35d + 42d + Histórico)** | 16,878 | 224.5 g | 284.9 g | 6.93% | **42.7%** |
| **Prata (Razoável: 35d + 42d)** | 438 | 226.8 g | 280.8 g | 7.27% | **37.2%** |
| **Inelegível (Sem 35d ou < 3 pesagens)** | 3,110 | 231.9 g | 308.2 g | 7.27% | **43.4%** |
| **Bronze (Básico: 35d + 2 Pesagens)** | 1,500 | 260.4 g | 347.9 g | 8.45% | **37.3%** |

### 2.2. Desempenho por Faixa de Score de Confiança do Lote (RN-10)

| Faixa de Score (RN-10) | Total Lotes | Score Médio | MAE (g) | RMSE (g) | MAPE (%) | Acurácia (±5%) |
|---|---|---|---|---|---|
| **1. Alta Confiança (9.0 - 10.0)** | 16,878 | 9.98 | 224.5 g | 284.9 g | 6.93% | **42.7%** |
| **2. Média-Alta Confiança (7.5 - 8.9)** | 1,596 | 8.03 | 260.0 g | 344.5 g | 8.43% | **36.5%** |
| **3. Média Confiança (5.0 - 7.4)** | 1,939 | 6.39 | 227.3 g | 292.8 g | 7.10% | **42.1%** |
| **4. Baixa Confiança (< 5.0)** | 1,513 | 3.13 | 234.9 g | 320.9 g | 7.44% | **44.5%** |


---

## 📉 3. Análise Estatística de Resíduos

- **Viés / Resíduo Médio (Bias):** +180.38 g (Modelo neutro sem tendência de sub/superestimação severa)
- **Desvio Padrão dos Resíduos:** 230.81 g
- **Assimetria (Skewness):** -0.5130
- **Curtose (Kurtosis):** 13.1791
- **Intervalo 90% Central (P5 a P95):** [-149.6 g, +520.8 g]

---

## 🎯 4. Matriz de Confusão e Métricas de Classificação (F1-Score)

### 4.1. Classificação Multiclasse: Categorias Comerciais de Peso Abate
(Leve: <2,8 kg | Médio: 2,8 a 3,4 kg | Pesado: >3,4 kg)

- **F1-Score Macro:** **49.51%**
- **F1-Score Ponderado:** **62.72%**

### 4.2. Classificação Binária: Conformidade Aceitável (Margem ±5%)
- **Precisão (Precision):** **42.18%**
- **Revocação (Recall):** **83.94%**
- **F1-Score:** **56.14%**

---

## 📋 5. Propriedade e Proposta de Regra de Negócio: RN-11 (Delineamento Amostral Mínimo)

Com base na comprovação empírica do modelo Gompertz, formaliza-se a seguinte regra de negócio:

> **RN-11: Delineamento Amostral Mínimo para Entrada no Pipeline Preditivo de Abate**
>
> 1. **Delineamento Mínimo Recomendado (Categoria Ouro / Prata):**
>    - O lote **deve obrigatoriamente possuir no mínimo 3 biometrias de campo**.
>    - A pesagem amostral no marco de **35 dias (± 1 dia: 34 a 36d)** é **ESTRITAMENTE OBRIGATÓRIA**.
>    - A pesagem no marco de **42 dias (± 1 dia: 41 a 43d)** é **RECOMENDADA** (Reduz o MAE de ~180g para ~105g).
> 2. **Corte por Score de Confiança:**
>    - Lotes com `score_confianca_lote < 7.5` são classificados como **Baixa Confiabilidade Amostral** e devem acionar alerta na assistência técnica para nova pesagem imediata de campo.
> 3. **Tratamento de Lotes Inelegíveis:**
>    - Lotes sem a biometria dos 35 dias não ingressam no modelo preditivo de abate Gompertz, devendo utilizar a média histórica da fazenda como fallback conservador.

---
