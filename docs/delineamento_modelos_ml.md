# Delineamento de Modelos de Machine Learning

## 1. Algoritmos Comparados
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- HistGradientBoosting Regressor
- Modelo Híbrido Base (Gompertz + Gradient Boosting Ensemble)

## 2. Regras e Dataset
- Base: RN-01 a RN-12 (elegivel_rn11=1 e RN-12 Gêmeos Digitais KNN)
- Validação Cruzada: 5-Fold GroupKFold por `lote_composto`
- Métricas: MAE (g), RMSE (g), MAPE (%), R²

## 3. Resultados
| Model                 |   MAE (g) |   RMSE (g) |   MAPE (%) |       R² |
|:----------------------|----------:|-----------:|-----------:|---------:|
| XGBoost               |   123.082 |    160.349 |    3.80345 | 0.570706 |
| LightGBM              |   123.8   |    160.64  |    3.8285  | 0.569201 |
| HistGradientBoosting  |   123.977 |    160.973 |    3.8344  | 0.567395 |
| Híbrido (Base + LGBM) |   124.977 |    162.222 |    3.864   | 0.560678 |

## 4. Conclusão
A comparação demonstra qual modelo apresenta melhor balanço entre acurácia (menor MAE e MAPE) e generalização, auxiliando na escolha do estimador final de peso de abate.
