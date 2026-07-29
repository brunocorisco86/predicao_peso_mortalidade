# 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Item 2 - Longitudinal](https://img.shields.io/badge/Modelagem%20Longitudinal-S%C3%A9ries%20Temporais-brightgreen.svg)](docs/sugestoes_melhoria_predicao.md)
[![Graphify](https://img.shields.io/badge/Knowledge%20Graph-Graphify-orange.svg)](graphify-out/GRAPH_REPORT.md)

Este repositório contém a solução dedicada à **predição do peso corporal de frangos de corte na idade de abate** (aves com idade entre 42 e 60 dias).

---

## 🚀 1. Avanço da Modelagem Longitudinal de Séries Temporais (Item 2)

Implementamos a **Modelagem Longitudinal de Séries Temporais** (Item 2 do Plano Estratégico), extraindo as pesagens intermediárias do lote (dias 21, 28 e 35), o **Ganho de Peso Diário (GPD)** da 4ª e 5ª semanas e a **Aceleração de Crescimento**:

| Abordagem Preditiva | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático Médio) | Redução de Erro / Ganho |
|---|---|---|---|---|
| **Modelo Estático Baseline** | 0,3684 | 118,80 g | 160,74 g | Sem Séries Temporais |
| **HistGB Longitudinal (Item 2)** | 0,5262 | 97,95 g | 139,20 g | MAE < 98g |
| **LightGBM Longitudinal (Item 2)** | 0,5296 | 97,49 g | 138,72 g | MAE < 98g |
| **XGBoost Longitudinal (Item 2)** | 0,5331 | 97,40 g | 138,20 g | MAE < 98g |
| **Stacking Ensemble Longitudinal (Best Model)** | **0,5359** | **96,92 g** | **137,79 g** | **🏆 Menor Erro Absoluto (MAE < 97g / Desvio 3,2%)** |

> 📌 **Impacto no Negócio:** Com a inclusão das séries temporais de ganho de peso, o desvio médio na previsão de abate de um frango de $3,0\text{kg}$ caiu de $118,8\text{g}$ para **$96,92\text{g}$** (apenas **$3,2\%$** de desvio), e a capacidade explicativa ($R^2$) subiu de **$0,36$ para $0,53$** ($+45\%$ de ganho).

---

## 📈 2. Importância das Variáveis no Modelo Longitudinal

Ranking de relevância das variáveis com a introdução das séries temporais:

1. **`peso_dia_35`**: Peso médio observado no lote aos 35 dias (última pesagem da 5ª semana).
2. **`gpd_semana5`**: Ganho de peso diário médio ($\text{g/dia}$) observado entre os dias 28 e 35.
3. **`peso_projetado_gpd`**: Trajetória projetada a partir da velocidade de ganho da 5ª semana.
4. **`c15`**: Peso inicial do pintainho de 1 dia.
5. **`cab_alojadas`**: Densidade / Quantidade de cabeças alojadas.
6. **`mortalidade`**: Desafio sanitário acumulado.
7. **`x02`**: Distância entre a propriedade e o abatedouro (km).

* 📊 Visualização de Importância: [plots/importancia_features_longitudinal.png](plots/importancia_features_longitudinal.png).
* 📊 Visualização Predito vs Real: [plots/modelo_longitudinal_predito_vs_real.png](plots/modelo_longitudinal_predito_vs_real.png).

---

## 💡 3. Metodologia de Ajuste da Curva de Gompertz vs Alvo de Abate

A curva de Gompertz é ajustada utilizando o histórico completo de crescimento (dias 1 a 60), enquanto a avaliação de erro é filtrada na janela de abate ($\ge 42$ dias):

* **Ajuste da Curva (Fitting):** Feito sobre todo o histórico etário (dias 1 a 60).
  $$\mathbf{W(t) = 6260,16 \cdot \exp\left(-4,7378 \cdot \exp(-0,0449 \cdot t)\right)}$$

---

## 📋 4. Regras de Negócio Implementadas ([`docs/regras_de_negocio_abate.md`](docs/regras_de_negocio_abate.md))

* **RN-01 (Janela de Abate):** $42 \le \text{idade} \le 60\text{ dias}$ ($15.416$ registros de abate analisados).
* **RN-02 (Faixa Comercial de Peso):** $1,80\text{ kg} \le \text{peso} \le 4,80\text{ kg}$ ($1.800\text{g}$ a $4.800\text{g}$).
* **RN-05 (Filtro IQR por Idade de Abate):** Remoção de anomalias extremas via $3,0 \times \text{IQR}$ por dia de abate.

---

## 🖼️ 5. Galeria de Gráficos Atualizados

| Gráfico | Descrição do Diagnóstico de Abate |
|---|---|
| [modelo_longitudinal_predito_vs_real.png](plots/modelo_longitudinal_predito_vs_real.png) | Dispersão do Modelo Longitudinal de Séries Temporais ($\text{MAE} = 96,9\text{g}$ / $R^2 = 0,5359$). |
| [importancia_features_longitudinal.png](plots/importancia_features_longitudinal.png) | Ranking de importância de variáveis com pesagens intermediárias e GPD. |
| [comparativo_modelos_avancados.png](plots/comparativo_modelos_avancados.png) | Comparativo de RMSE e $R^2$ entre algoritmos. |
| [curva_crescimento_gompertz.png](plots/curva_crescimento_gompertz.png) | Curva Biológica de Gompertz ajustada em 1-60 dias. |

---

## 🛠️ 6. Guia de Execução

```bash
source .venv/bin/activate

# 1. Executar EDA e Filtro de Abate (RN-01 a RN-05)
python3 -m src.eda_outliers

# 2. Treinar Modelo de Predição do Peso de Abate
python3 -m src.models.train_predict_weight

# 3. Executar Modelo Longitudinal de Séries Temporais (Item 2)
python3 -m src.models.longitudinal_time_series_model
```

---

## 📄 Licença

Este projeto é protegido sob a **GNU General Public License v3.0 (GPLv3)**. Veja o arquivo [`LICENSE`](LICENSE) para maiores detalhes.
