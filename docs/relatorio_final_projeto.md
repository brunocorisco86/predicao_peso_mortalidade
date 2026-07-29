# Relatório Final de Consolidação do Projeto: Predição do Peso de Abate em Aves de Corte

**Data da Consolidação:** 29 de Julho de 2026  
**Status do Repositório:** 100% Sincronizado com o GitHub Remote (`origin/main`)  
**Métrica Final Alcançada:** $\text{MAE} = 92,11\text{ g}$ | $\text{RMSE} = 130,86\text{ g}$ | $R^2 = 0,5814$

---

## 🏆 Resumo Executivo da Evolução do Projeto

Ao longo do desenvolvimento, refatoramos o escopo para focar **exclusivamente na predição do peso final ao abate ($42 \le \text{idade} \le 60\text{ dias}$)**. A tabela abaixo resume o avanço contínuo obtido através de engenharia de dados, séries temporais e dimensões de vizinhança:

| Fase / Iteração | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático) | Métrica $R^2$ | Inovação Aplicada |
|---|---|---|---|---|
| **1. Modelo Estático Inicial** | 118,80 g | 160,74 g | 0,3684 | Baseline com atributos fixos do lote |
| **2. Modelo Longitudinal (Item 2)** | 96,92 g | 137,79 g | 0,5359 | Pesagens aos 21d, 28d, 35d e GPD |
| **3. Modelo Tri-Híbrido** | 96,15 g | 137,16 g | 0,5401 | Fusão: Gompertz + ARIMA + Stacking ML |
| **4. Dimensão Delta (g) por Aviário** | 92,57 g | 131,54 g | 0,5770 | Mapeamento do viés fixo do aviário em gramas |
| **5. Dimensão Erro Relativo (%) por Aviário** | **92,11 g** | **130,86 g** | **0,5814** | **🏆 Recorde Histórico ($\mathbf{R^2 = 58,14\%}$ / MAE = 92,1g)** |

---

## 🔬 Subagentes Especializados Criados

Para garantir a continuação e suporte ao projeto nas suas próximas sessões, criamos 2 subagentes dedicados no sistema:

### 1. Subagente `explicador_avanco`
* **Função:** Fornecer explicações técnicas, zootécnicas e executivas para diretores, extensionistas e partes interessadas sobre a evolução do projeto, justificando a escolha das variáveis, curva de Gompertz, ARIMA e o ganho do Erro Relativo (%).

### 2. Subagente `facilitador_implementacao`
* **Função:** Auxiliar desenvolvedores e equipe de TI na execução das rotinas Python, inferência preditiva para novos lotes de abate, manutenção do ambiente virtual `.venv` e integração dos modelos salvos em `src/models/saved/`.

---

## 🛠️ Guia de Execução Rápida dos Scripts

```bash
# Ativar o ambiente virtual
source .venv/bin/activate

# 1. Pipeline de ETL e Carga
python3 -m src.etl.export_unified_data

# 2. Limpeza e Filtro de Abate (RN-01 a RN-05)
python3 -m src.eda_outliers

# 3. Execução do Modelo Campeão (Erro Relativo por Aviário)
python3 -m src.models.aviary_relative_error_model

# 4. Simulações Comparativas de 10 Lotes
python3 -m src.models.simulate_slaughter_weights
```

---

## 📄 Licença
Este repositório está protegido sob a **GNU General Public License v3.0 (GPLv3)**.
