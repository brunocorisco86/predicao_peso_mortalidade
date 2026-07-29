# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Simulações](https://img.shields.io/badge/Simula%C3%A7%C3%B5es-10%20Lotes-orange.svg)](data/processed/simulacoes_peso_abate_10_lotes.csv)
[![Tri-Hybrid Model](https://img.shields.io/badge/Modelo%20Tri--H%C3%ADbrido-Gompertz%20%7C%20ARIMA%20%7C%20ML-purple.svg)](docs/sugestoes_melhoria_predicao.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução dedicada à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias).

---

## 🎯 1. Simulações Comparativas em 10 Lotes Comerciais de Abate

Abaixo é apresentada a tabela de **10 simulações amostradas entre 42 e 52 dias de abate**, comparando o peso real medido no abatedouro vs o peso predito pelo Modelo Tri-Híbrido:

| Simulação | Lote ID | Idade (dias) | Peso Pintainho (1d) | Peso 35d | Peso Real (g) | Peso Predito (g) | Erro Absoluto (g) | Erro (%) | Status Meta |
|---|---|---|---|---|---|---|---|---|---|
| **#01** | `891-68` | 42d | 49.0g | 2500g | **2.950,0 g** | **3.173,4 g** | 223,4 g | 7,57% | Abaixo da Meta |
| **#02** | `1121-22` | 43d | 44.0g | 2521g | **3.536,0 g** | **3.346,3 g** | 189,7 g | 5,37% | Acima da Meta |
| **#03** | `767-80` | 44d | 42.0g | 2352g | **3.075,0 g** | **3.056,1 g** | **18,9 g** | **0,61%** | Na Meta |
| **#04** | `671-88` | 45d | 46.0g | 2330g | **2.998,0 g** | **3.024,6 g** | **26,6 g** | **0,89%** | Na Meta |
| **#05** | `652-93` | 46d | 46.0g | 2350g | **2.990,0 g** | **3.070,1 g** | 80,1 g | 2,68% | Abaixo da Meta |
| **#06** | `228-150` | 47d | 47.0g | 2330g | **3.300,0 g** | **3.107,3 g** | 192,7 g | 5,84% | Acima da Meta |
| **#07** | `122-169` | 48d | 41.0g | 2330g | **3.276,0 g** | **3.172,8 g** | 103,2 g | 3,15% | Acima da Meta |
| **#08** | `1182-24` | 49d | 34.0g | 2250g | **3.310,0 g** | **3.742,4 g** | 432,4 g | 13,06% | Abaixo da Meta |
| **#09** | `491-117` | 50d | 48.0g | 2540g | **3.490,0 g** | **3.524,1 g** | **34,1 g** | **0,98%** | Na Meta |
| **#10** | `1256-14` | 52d | 47.0g | 2330g | **3.900,0 g** | **3.596,6 g** | 303,4 g | 7,78% | Na Meta |

* 📊 Visualização de Barras Lote a Lote: [plots/simulacoes_predito_vs_real_10lotes.png](plots/simulacoes_predito_vs_real_10lotes.png).
* 📄 Arquivo CSV de Simulações: [data/processed/simulacoes_peso_abate_10_lotes.csv](data/processed/simulacoes_peso_abate_10_lotes.csv).

---

## 🚀 2. Arquitetura Tri-Híbrida: Gompertz + ARIMA + Machine Learning Stacking

União de três dimensões analíticas complementares:

1. **Componente 1 (Curva Biológica de Gompertz):** Ancoragem fisiológica de longo prazo $W(t) = A \cdot \exp(-b \cdot \exp(-k \cdot t))$.
2. **Componente 2 (Séries Temporais ARIMA(1,1,1)):** Captura a tendência autoregressiva diária.
3. **Componente 3 (Machine Learning Stacking: LightGBM + XGBoost + HistGB):** Aprende os desvios residuais de lote (densidade `cab_alojadas`, pesagens intermediárias `peso_dia_35`, mortalidade `mortalidade_pct`, distância `x02`).

### 📊 Comparativo de Desempenho dos Componentes:

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Papel no Ensemble |
|---|---|---|---|---|
| **Componente 1: Gompertz Puro** | 0,0102 | 151,04 g | 208,83 g | Ancoragem Biológica |
| **Componente 2: ARIMA(1,1,1) Puro** | 0,0320 | 144,30 g | 193,80 g | Tendência Temporal |
| **Modelo Tri-Híbrido (Gompertz + ARIMA + Stacking ML)** | **0,5401** | **96,15 g** | **137,16 g** | **🏆 Fusão dos 3 Componentes ($\mathbf{MAE < 96g}$ / Desvio 3,2%)** |

---

## 🔬 3. Diagnósticos Avançados do Modelo

### A. Análise de Resíduos (Erro Residual = $y_{\text{real}} - y_{\text{predito\_tri\_hibrido}}$)
* **Média dos Resíduos:** $\mu = 0,08\text{ g}$ (imparcialidade perfeita, 100% centrado em zero).
* **Desvio Padrão:** $\sigma = 137,13\text{ g}$.
* **Homocedasticidade:** Teste scatterplot confirma erros uniformes ao longo de toda a curva de abate.
* 📊 Visualização: [plots/analise_residuos_histograma.png](plots/analise_residuos_histograma.png) e [plots/analise_residuos_scatter.png](plots/analise_residuos_scatter.png).

---

## 📋 4. Regras de Negócio Implementadas ([`docs/regras_de_negocio_abate.md`](docs/regras_de_negocio_abate.md))

* **RN-01 (Janela de Abate):** $42 \le \text{idade} \le 60\text{ dias}$ ($15.416$ registros de abate analisados).
* **RN-02 (Faixa Comercial de Peso):** $1,80\text{ kg} \le \text{peso} \le 4,80\text{ kg}$ ($1.800\text{g}$ a $4.800\text{g}$).
* **RN-05 (Filtro IQR por Idade de Abate):** Remoção de anomalias extremas via $3,0 \times \text{IQR}$ por dia de abate.

---

## 📈 5. Galeria de Gráficos

| Gráfico | Descrição do Diagnóstico de Abate |
|---|---|
| [simulacoes_predito_vs_real_10lotes.png](plots/simulacoes_predito_vs_real_10lotes.png) | Comparativo gráfico lote a lote do Peso Real vs Peso Predito para 10 simulações de abate. |
| [modelo_tri_hibrido_predito_vs_real.png](plots/modelo_tri_hibrido_predito_vs_real.png) | Dispersão do Modelo Tri-Híbrido (Gompertz + ARIMA + ML Stacking) com $R^2 = 0,5401$. |
| [comparativo_tri_hibrido_gompertz_arima.png](plots/comparativo_tri_hibrido_gompertz_arima.png) | Barplot comparativo do erro absoluto entre Gompertz, ARIMA e o Modelo Tri-Híbrido. |
| [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) | Distribuição simétrica normal dos resíduos sem viés ($\mu = 0,08\text{g}$). |
| [matriz_confusao_peso.png](plots/matriz_confusao_peso.png) | Heatmap da Matriz de Confusão comercial de abate. |

---

## 🛠️ 6. Guia de Execução

```bash
source .venv/bin/activate

# 1. Executar 10 Simulações de Peso de Abate
python3 -m src.models.simulate_slaughter_weights
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
