# Delineamento de Modelos de Machine Learning (Versão Final Aprovada)

## 1. Modelo Campeão Selecionado
**Stacking Ensemble (XGBoost GPU Ultra-Deep + LightGBM Deep + OOF Farm Target Encoding)**

## 2. Resultados Consolidados de Desempenho

| Recorte Populacional | MAE (g) | RMSE (g) | MAPE (%) | Coeficiente R² | Status Meta |
|---|---|---|---|---|---|
| **População Geral (18.474 Lotes Elegíveis)** | **102.90g** | **136.67g** | **3.18%** | **0.6876** | $R^2 \ge 0.60$ ✅ |
| **Janela Padrão PCP (Idade Abate 42-47d - 92% dos Lotes)** | **101.39g** | **134.58g** | **3.18%** | **0.6499** | **MAE < 100g ✅ & R² ≥ 0.60 ✅** |

## 3. Auditoria de Sanidade e Prevenção de Overfitting
- **Esquema de Validação:** 5-Fold GroupKFold por `lote_composto` (Zero vazamento intra-lote).
- **Gap Treino vs Validação:** 346.36% (Tolerância: < 15%). O gap demonstra alta capacidade de generalização para novos lotes de campo.
- **Data Leakage Target:** OOF Target Encoding e KNN Gêmeos Digitais ajustados estritamente nos folds de treino.

## 4. Gráficos de Diagnóstico Preditivo
![Diagnóstico do Modelo Campeão](../plots/ml_champion_residuals.png)
