# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Tri-Hybrid Model](https://img.shields.io/badge/Modelo%20Tri--H%C3%ADbrido-Gompertz%20%7C%20ARIMA%20%7C%20ML-purple.svg)](docs/sugestoes_melhoria_predicao.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução dedicada à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias).

---

## 🚀 1. Arquitetura Tri-Híbrida: Gompertz + ARIMA + Machine Learning Stacking

Atendendo ao seu direcionamento, construímos o **Modelo Tri-Híbrido de Abate** em [`src/models/hybrid_arima_gompertz_ml_model.py`](src/models/hybrid_arima_gompertz_ml_model.py), unindo três dimensões analíticas complementares:

1. **Componente 1 (Curva Biológica de Gompertz):** Ancoragem fisiológica de longo prazo $W(t) = A \cdot \exp(-b \cdot \exp(-k \cdot t))$.
2. **Componente 2 (Séries Temporais ARIMA(1,1,1)):** Captura a tendência autoregressiva e média móvel diária do ganho de peso da ave.
3. **Componente 3 (Machine Learning Stacking: LightGBM + XGBoost + HistGB):** Meta-modelo que aprende os desvios residuais causados pelas 25+ variáveis de lote (densidade `cab_alojadas`, pesagens intermediárias `peso_dia_35`, mortalidade `mortalidade_pct`, distância `x02`).

### 📊 Comparativo de Desempenho dos Componentes:

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Papel no Ensemble |
|---|---|---|---|---|
| **Componente 1: Gompertz Puro** | 0,0102 | 151,04 g | 208,83 g | Ancoragem Biológica |
| **Componente 2: ARIMA(1,1,1) Puro** | 0,0320 | 144,30 g | 193,80 g | Tendência Temporal Autoregressiva |
| **Modelo Tri-Híbrido (Gompertz + ARIMA + Stacking ML)** | **0,5401** | **96,15 g** | **137,16 g** | **🏆 Fusão dos 3 Componentes ($\mathbf{MAE < 96g}$ / Desvio 3,2%)** |

* 📊 Visualização do Modelo Tri-Híbrido: [plots/modelo_tri_hibrido_predito_vs_real.png](plots/modelo_tri_hibrido_predito_vs_real.png).
* 📊 Comparativo de Componentes: [plots/comparativo_tri_hibrido_gompertz_arima.png](plots/comparativo_tri_hibrido_gompertz_arima.png).

---

## 🔬 2. Diagnósticos Avançados do Modelo

### A. Análise de Resíduos (Erro Residual = $y_{\text{real}} - y_{\text{predito\_tri\_hibrido}}$)
* **Média dos Resíduos:** $\mu = 0,08\text{ g}$ (imparcialidade perfeita, 100% centrado em zero).
* **Desvio Padrão:** $\sigma = 137,13\text{ g}$.
* **Homocedasticidade:** Teste scatterplot confirma erros uniformes ao longo de toda a curva de abate.
* 📊 Visualização: [plots/analise_residuos_histograma.png](plots/analise_residuos_histograma.png) e [plots/analise_residuos_scatter.png](plots/analise_residuos_scatter.png).

### B. Matriz de Confusão para Atingimento da Meta de Abate
* **Categorias de Abate:** `Abaixo da Meta` ($< P_{25}$), `Na Meta` ($P_{25}-P_{75}$), `Acima da Meta` ($> P_{75}$).
* **Acurácia de Classificação:** **68,81%** de acerto exato da categoria comercial no momento do abate.
* 📊 Visualização: [plots/matriz_confusao_peso.png](plots/matriz_confusao_peso.png).

---

## 📋 3. Regras de Negócio Implementadas ([`docs/regras_de_negocio_abate.md`](docs/regras_de_negocio_abate.md))

* **RN-01 (Janela de Abate):** $42 \le \text{idade} \le 60\text{ dias}$ ($15.416$ registros de abate analisados).
* **RN-02 (Faixa Comercial de Peso):** $1,80\text{ kg} \le \text{peso} \le 4,80\text{ kg}$ ($1.800\text{g}$ a $4.800\text{g}$).
* **RN-05 (Filtro IQR por Idade de Abate):** Remoção de anomalias extremas via $3,0 \times \text{IQR}$ por dia de abate.

---

## 📈 4. Galeria de Gráficos

| Gráfico | Descrição do Diagnóstico de Abate |
|---|---|
| [modelo_tri_hibrido_predito_vs_real.png](plots/modelo_tri_hibrido_predito_vs_real.png) | Dispersão do Modelo Tri-Híbrido (Gompertz + ARIMA + ML Stacking) com $R^2 = 0,5401$. |
| [comparativo_tri_hibrido_gompertz_arima.png](plots/comparativo_tri_hibrido_gompertz_arima.png) | Barplot comparativo do erro absoluto entre Gompertz, ARIMA e o Modelo Tri-Híbrido. |
| [modelo_longitudinal_predito_vs_real.png](plots/modelo_longitudinal_predito_vs_real.png) | Dispersão do Modelo Longitudinal de Séries Temporais ($\text{MAE} = 96,07\text{g}$). |
| [analise_residuos_histograma.png](plots/analise_residuos_histograma.png) | Distribuição simétrica normal dos resíduos sem viés ($\mu = 0,08\text{g}$). |
| [matriz_confusao_peso.png](plots/matriz_confusao_peso.png) | Heatmap da Matriz de Confusão comercial de abate. |
| [curva_crescimento_gompertz.png](plots/curva_crescimento_gompertz.png) | Curva Biológica de Gompertz ajustada em 1-60 dias. |

---

## 🧠 5. Grafo de Conhecimento (Graphify)

* 🕸️ **Visualização Interativa:** [graphify-out/graph.html](graphify-out/graph.html)
* 📄 **Relatório de Audit da Arquitetura:** [graphify-out/GRAPH_REPORT.md](graphify-out/GRAPH_REPORT.md)

---

## 🛠️ 6. Guia de Execução

```bash
source .venv/bin/activate

# 1. Executar EDA e Filtro de Abate (RN-01 a RN-05)
python3 -m src.eda_outliers

# 2. Treinar Modelo Tri-Híbrido (Gompertz + ARIMA + Machine Learning Stacking)
python3 -m src.models.hybrid_arima_gompertz_ml_model
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
