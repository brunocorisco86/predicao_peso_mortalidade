# Relatório Final de Consolidação do Projeto: Predição do Peso de Abate em Aves de Corte

**Data da Consolidação:** 29 de Julho de 2026  
**Status do Repositório:** 100% Sincronizado com o GitHub Remote (`origin/main`)  
**Novo Dataset Integrado:** `data/raw/peso_abate/export_peso_abate_2023_2026.xlsx` (22.207 registros oficiais de frigorífico)  
**Métrica Final Alcançada (Abate Real):** $\text{MAE} = 116,47\text{ g}$ | $\text{RMSE} = 149,54\text{ g}$ | $R^2 = 0,5746$ | $\text{MAPE} = 3,55\%$

---

## 🏆 Resumo Executivo da Evolução do Projeto

Ao longo do desenvolvimento, refatoramos o escopo para focar **exclusivamente na predição do peso final ao abate ($42 \le \text{idade} \le 60\text{ dias}$)**. A incorporação dos dados oficiais de abate do frigorífico (`export_peso_abate_2023_2026.xlsx`) permitiu remodelar o Modelo Entidade Relacionamento (MER) e validar a performance do modelo campeão diretamente contra o alvo real (`peso_abate_g`).

| Fase / Iteração | MAE (Erro Médio Absoluto) | RMSE (Erro Quadrático) | Métrica $R^2$ | Inovação Aplicada |
|---|---|---|---|---|
| **1. Modelo Estático Inicial** | 118,80 g | 160,74 g | 0,3684 | Baseline com atributos fixos do lote |
| **2. Modelo Longitudinal** | 96,92 g | 137,79 g | 0,5359 | Pesagens aos 21d, 28d, 35d e GPD |
| **3. Modelo Tri-Híbrido** | 96,15 g | 137,16 g | 0,5401 | Fusão: Gompertz + ARIMA + Stacking ML |
| **4. Ground Truth Abate Frigorífico** | **116,47 g** | **149,54 g** | **0,5746** | **🏆 Validação no Alvo Oficial de Abate (16.039 lotes / MAPE = 3,55%)** |

---

## 🎯 Alinhamento aos 3 Pilares Estratégicos

1. **Comunicação Eficiente (Plataforma Centralizada):** Unificação dos registros de campo e pesagens finais de abatedouro em tabela SQLite dedicada (`peso_abate`).
2. **Processos Otimizados (Redesenho de Fluxo):** Programação de abate e agendamento de apanha de frangos com suporte a faixas de erro residual percentual por aviário.
3. **Tecnologia Habilitadora (TMS e IoT nos Silos):** Modelos preditivos integrados ao sistema TMS de logística de transporte e aos sensores de nível de silo.

---

## 🛠️ Guia de Execução Rápida dos Scripts

```bash
# 1. Extração do Dataset Oficial de Abate para SQLite DB
python3 src/etl/extract_peso_abate.py

# 2. Exportação da Visão Unificada com Tabela de Abate
python3 src/etl/export_unified_data.py

# 3. Limpeza e Filtro de Abate (RN-01 a RN-05)
python3 src/eda_outliers.py

# 4. Treinamento e Avaliação do Modelo Campeão
python3 src/models/aviary_relative_error_model.py

# 5. Exportação do Schema DDL do Banco SQLite
python3 src/utils/export_db_schema.py
```

---

## 📄 Licença
Este repositório está protegido sob a **GNU General Public License v3.0 (GPLv3)**.
