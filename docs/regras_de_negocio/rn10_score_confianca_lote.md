---
rule_id: RN-10
title: Cálculo do Score de Confiança de Amostragem do Lote
category: Data Quality & Governança
target_table: lote_sampling_confidence
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-10: Cálculo do Score de Confiança de Amostragem do Lote

## 📋 Descrição Geral
A **RN-10** calcula um indicador sintético de qualidade de dados (`score_confianca_lote`) para medir a confiabilidade biométrica de cada lote de frangos.

---

## ⚙️ Regra de Negócio Técnica
$$	ext{Score} = (	ext{Qtd Marcos Presentes}) 	imes 1,25 + (	ext{Presença de 35d}) 	imes 1,25$$

O score varia de $0,0$ (sem dados) a $10,0$ (amostragem perfeita em todos os marcos e 35d presente).

---

## 🎯 Impacto no Modelo e DataOps
- **Filtro de Governança:** Alimenta diretamente o Gateway de Elegibilidade RN-11.
