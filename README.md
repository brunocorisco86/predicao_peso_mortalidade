# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Skill](https://img.shields.io/badge/Skill-slaughter--simulation-orange.svg)](/home/brunoconter/.gemini/config/skills/slaughter-simulation/SKILL.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução dedicada à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias).

---

## 🎯 1. Simulação de 50 Lotes Aleatórios no Pico de Abate (44 a 46 Dias)

Executamos uma simulação amostral de **lotes aleatórios com idades de 44 a 46 dias**, avaliando a precisão real vs predita:

* **MAE Amostral (44-46d):** **$88,14\text{ g}$**
* **Erro Relativo Médio (%):** **$3,03\%$** (precisão de $96,97\%$)
* 📊 **Scatter Plot com Linha de Identidade $Y = X$:** [plots/simulacao_scatter_50lotes_44_46d.png](plots/simulacao_scatter_50lotes_44_46d.png).
* 📄 **Tabela Exportada de 50 Lotes:** [data/processed/simulacoes_50_lotes_44_46d.csv](data/processed/simulacoes_50_lotes_44_46d.csv).
* ⚙️ **Skill de Simulação Criada:** [`slaughter-simulation`](/home/brunoconter/.gemini/config/skills/slaughter-simulation/SKILL.md).

---

## 📊 Sample de Simulações (Ages 44-46d):

| Simulação | Lote ID | Idade (dias) | Peso 35d (g) | Peso Real (g) | Peso Predito Corrigido (g) | Erro Absoluto ($\Delta$) | Erro (%) |
|---|---|---|---|---|---|---|---|
| **#01** | `730-89` | 45d | 2375g | **3.061,0 g** | **3.094,7 g** | 33,7 g | 1,10% |
| **#02** | `553-108` | 45d | 2150g | **2.950,0 g** | **2.982,1 g** | 32,1 g | 1,09% |
| **#03** | `1051-42` | 45d | 2325g | **3.108,0 g** | **3.086,6 g** | **21,4 g** | **0,69%** |
| **#08** | `1119-26` | 45d | 2245g | **2.980,0 g** | **2.996,1 g** | **16,1 g** | **0,54%** |
| **#11** | `971-53` | 45d | 2442g | **3.126,0 g** | **3.121,5 g** | **4,5 g** | **0,14%** |
| **#13** | `705-92` | 46d | 2400g | **3.143,0 g** | **3.117,0 g** | **26,0 g** | **0,83%** |

---

## 🌾 2. KPIs Zootécnicos e Dimensão por Aviário

1. **`kpi_iep_proxy` (Proxy do Índice de Eficiência Produtiva Europeu)**
2. **`kpi_razao_gpd_sem5_vs_vida` (Aceleração Retida da 5ª Semana)**
3. **`erro_relativo_aviario_pct` (Fator de Eficiência do Aviário)**

### 📊 Evolução Global das Métricas (5-Fold GroupKFold):

| Abordagem Preditiva | MAE (Erro Absoluto) | RMSE (Erro Quadrático) | Métrica $R^2$ | Ganho Acumulado |
|---|---|---|---|---|
| **Modelo Estático Baseline** | 118,80 g | 160,74 g | 0,3684 | Sem Séries / KPIs |
| **Modelo Longitudinal (Item 2)** | 96,92 g | 137,79 g | 0,5359 | Séries Temporais |
| **Modelo Tri-Híbrido (Gompertz+ARIMA+ML)** | 96,15 g | 137,16 g | 0,5401 | Tri-Híbrido |
| **Modelo com Delta em Gramas por Aviário** | 92,57 g | 131,54 g | 0,5770 | Delta Absoluto |
| **Modelo com 5 KPIs Zootécnicos + Aviário** | **92,05 g** | **131,13 g** | **0,5797** | **🏆 RECORD DE MODELAGEM ($\mathbf{MAE = 92,05g}$)** |

---

## 🛠️ 3. Guia de Execução

```bash
source .venv/bin/activate

# Executar Simulação de 50 Lotes no Abate (44 a 46 dias)
python3 -m src.models.simulate_50_batches_44_46d
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
