# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Erro Relativo Aviario](https://img.shields.io/badge/Dimens%C3%A3o-Erro%20Relativo%20%28%25%29-brightgreen.svg)](data/processed/dimensao_completa_aviario.csv)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução dedicada à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias).

---

## 🏆 1. Incorporação do Erro Relativo (%) por Aviário

Sua sugestão de extrair o **Erro Relativo Percentual de Ganho por Aviário** foi implementada com sucesso em [`src/models/aviary_relative_error_model.py`](src/models/aviary_relative_error_model.py).

$$\text{erro\_relativo\_aviario}_{\text{pct}} = \frac{\text{Peso Real} - \text{Peso Gompertz}}{\text{Peso Gompertz}} \times 100$$

### Por que esta variável é superior?
Ao contrário do desvio absoluto em gramas (que escala com a idade), o **Erro Relativo (%)** captura o **Fator de Eficiência Multiplicativo Constante do Aviário** (ex: se o aviário é $+3,2\%$ mais eficiente ou $-2,5\%$ menos eficiente que a média biológica em qualquer fase do lote).

### 📊 Evolução das Métricas do Projeto (5-Fold GroupKFold):

| Abordagem Preditiva | MAE (Erro Absoluto) | RMSE (Erro Quadrático) | Métrica $R^2$ | Ganho Acumulado |
|---|---|---|---|---|
| **Modelo Estático Baseline** | 118,80 g | 160,74 g | 0,3684 | Sem Séries / Aviário |
| **Modelo Longitudinal (Item 2)** | 96,92 g | 137,79 g | 0,5359 | Séries Temporais |
| **Modelo Tri-Híbrido (Gompertz+ARIMA+ML)** | 96,15 g | 137,16 g | 0,5401 | Tri-Híbrido |
| **Modelo com Delta em Gramas por Aviário** | 92,57 g | 131,54 g | 0,5770 | Delta Absoluto |
| **Modelo com Erro Relativo (%) por Aviário** | **92,11 g** | **130,86 g** | **0,5814** | **🏆 Recorde Histórico ($\mathbf{R^2 = 58,14\%}$ / MAE = 92g)** |

> 📌 **No Fold 4 da Validação Cruzada, o erro médio absoluto despencou para incríveis 88,80g!**

---

## 📈 2. Importância da Nova Variável `erro_relativo_aviario_pct`

* 📊 Visualização do Ranking: [plots/importancia_features_erro_relativo_aviario.png](plots/importancia_features_erro_relativo_aviario.png).
* 📄 Dimensão Completa do Aviário: [data/processed/dimensao_completa_aviario.csv](data/processed/dimensao_completa_aviario.csv).

---

## 🎯 3. Simulações em 10 Lotes Comerciais de Abate

Tabela de **10 simulações comerciais de abate** calibradas com o Erro Relativo (%) por Aviário:

| Simulação | Lote ID | Idade (dias) | Delta Aviário | Peso Real (g) | Peso Predito Corrigido (g) | Erro Absoluto ($\Delta$) | Erro (%) | Status Meta |
|---|---|---|---|---|---|---|---|---|
| **#01** | `891-68` | 42d | $-12,1\text{g}$ | **2.950,0 g** | **3.153,3 g** | 203,3 g | 6,89% | Abaixo da Meta |
| **#02** | `1121-22` | 43d | $+81,5\text{g}$ | **3.536,0 g** | **3.362,9 g** | 173,1 g | 4,89% | Acima da Meta |
| **#03** | `767-80` | 44d | $-16,5\text{g}$ | **3.075,0 g** | **3.053,2 g** | **21,8 g** | **0,71%** | Na Meta |
| **#04** | `671-88` | 45d | $-50,8\text{g}$ | **2.998,0 g** | **3.005,1 g** | **7,1 g** | **0,24%** | Na Meta |
| **#05** | `652-93` | 46d | $-17,8\text{g}$ | **2.990,0 g** | **3.064,6 g** | 74,6 g | 2,50% | Abaixo da Meta |
| **#06** | `228-150` | 47d | $+7,6\text{g}$ | **3.300,0 g** | **3.172,0 g** | 128,0 g | 3,88% | Acima da Meta |
| **#07** | `122-169` | 48d | $+38,4\text{g}$ | **3.276,0 g** | **3.247,2 g** | **28,8 g** | **0,88%** | Acima da Meta |
| **#08** | `1182-24` | 49d | $-139,8\text{g}$ | **3.310,0 g** | **3.563,3 g** | 253,3 g | 7,65% | Abaixo da Meta |
| **#09** | `491-117` | 50d | $-5,6\text{g}$ | **3.490,0 g** | **3.503,2 g** | **13,2 g** | **0,38%** | Na Meta |
| **#10** | `1256-14` | 52d | $+58,1\text{g}$ | **3.900,0 g** | **3.520,2 g** | 379,8 g | 9,74% | Na Meta |

---

## 🛠️ 4. Guia de Execução

```bash
source .venv/bin/activate

# Executar Experimento de Erro Relativo por Aviário
python3 -m src.models.aviary_relative_error_model
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
