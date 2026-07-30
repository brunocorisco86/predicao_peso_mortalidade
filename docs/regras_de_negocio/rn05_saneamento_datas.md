---
rule_id: RN-05
title: Saneamento de Cronologia e Datas Incompatíveis
category: Governança Temporária & Sanidade
target_table: variables
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-05: Saneamento de Cronologia e Datas Incompatíveis

## 📋 Descrição Geral
A **RN-05** verifica a sanidade cronológica de todos os eventos da vida do lote. Registros onde a data de pesagem precede a data de alojamento do pintainho ou sucede a data de abate no frigorífico são rejeitados.

---

## ⚙️ Regra de Negócio Técnica
$$	ext{Data Alojamento} \le 	ext{Data Evento Amostragem} \le 	ext{Data Abate Frigorífico}$$

---

## 🎯 Impacto no Modelo e DataOps
- **Prevenção de Anacronismos:** Evita que pesagens pós-abate contaminem o histórico preditivo do lote.
