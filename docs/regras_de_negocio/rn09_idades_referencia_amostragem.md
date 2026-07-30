---
rule_id: RN-09
title: Mapeamento de Idades de Referência de Amostragem
category: Séries Temporais & Discretização
target_table: extracao_mtech_data
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-09: Mapeamento de Idades de Referência de Amostragem

## 📋 Descrição Geral
A **RN-09** discretiza as idades brutas de amostragem do MTech em 7 marcos etários zootécnicos padrão, permitindo o pivotamento longitudinal dos pesos por lote.

---

## ⚙️ Regra de Negócio Técnica
| Idade de Referência (`idade_ref`) | Intervalo de Idade Real Permitido |
|---|---|
| **4d** | [3, 5] dias |
| **7d** | [6, 8] dias |
| **14d** | [13, 15] dias |
| **21d** | [20, 22] dias |
| **28d** | [27, 29] dias |
| **35d** | [34, 36] dias |
| **42d** | [41, 43] dias |

Pesagens fora dessas janelas de $\pm 1$ dia são descartadas da matriz pivô.

---

## 🎯 Impacto no Modelo e DataOps
- **Estandardização Temporal:** Habilita o cálculo exato de velocidades $GMD$ entre semanas consecutivas.
