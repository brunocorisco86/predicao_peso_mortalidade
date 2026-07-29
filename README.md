# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Dimensão Aviário](https://img.shields.io/badge/Dimens%C3%A3o-Delta%20por%20Avi%C3%A1rio-success.svg)](data/processed/dimensao_delta_por_aviario.csv)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução dedicada à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias).

---

## 🏆 1. Incorporação da Dimensão do Delta por Aviário (Correção por Propriedade)

A pedido da análise de negócio, criamos a **Dimensão do Delta por Aviário** ([`data/processed/dimensao_delta_por_aviario.csv`](data/processed/dimensao_delta_por_aviario.csv)), capturando o viés fixo histórico de desempenho de cada aviário ($1.132$ aviários mapeados).

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Redução de Erro / Ganho |
|---|---|---|---|---|
| **Modelo Estático Baseline** | 0,3684 | 118,80 g | 160,74 g | Sem Séries / Aviário |
| **Modelo Longitudinal (Item 2)** | 0,5359 | 96,92 g | 137,79 g | Séries Temporais |
| **Modelo Tri-Híbrido (Gompertz + ARIMA + ML)** | 0,5401 | 96,15 g | 137,16 g | Tri-Híbrido |
| **Modelo Calibrado com Dimensão do Aviário** | **0,5770** | **92,57 g** | **131,54 g** | **🏆 Menor Erro Absoluto ($\mathbf{MAE < 93g}$ / $R^2 = 57,7\%$)** |

> 📌 **Impacto no Negócio:** Ao adicionar o deslocamento histórico do aviário (`delta_aviario_g`), o erro médio absoluto caiu para **$92,57\text{g}$** e o poder explicativo ($R^2$) subiu para **$57,70\%$**.

---

## 🎯 2. Simulações em 10 Lotes Comerciais com Correção por Aviário

Tabela de **10 simulações comerciais de abate** com a calibração da Dimensão de Delta por Aviário:

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

* 📊 Visualização Comparativa com Aviário: [plots/simulacoes_predito_vs_real_correcao_aviario.png](plots/simulacoes_predito_vs_real_correcao_aviario.png).
* 📄 Dimensão de Delta Exportada: [data/processed/dimensao_delta_por_aviario.csv](data/processed/dimensao_delta_por_aviario.csv).

---

## 🔬 3. Diagnósticos Avançados

### A. Análise de Resíduos (Erro Residual = $y_{\text{real}} - y_{\text{predito\_aviario}}$)
* **Média dos Resíduos:** $\mu = 0,08\text{ g}$ (100% livre de viés).
* **Desvio Padrão:** $\sigma = 131,54\text{ g}$.
* 📊 Visualização: [plots/analise_residuos_histograma.png](plots/analise_residuos_histograma.png) e [plots/analise_residuos_scatter.png](plots/analise_residuos_scatter.png).

---

## 🛠️ 4. Guia de Execução

```bash
source .venv/bin/activate

# Executar Otimização com a Dimensão de Delta por Aviário
python3 -m src.models.aviary_delta_correction_model
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
