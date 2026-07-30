---
rule_id: RN-11
title: Gateway de Elegibilidade para o Modelo Direto de ML
category: Arquitetura Preditiva & Resiliência
target_table: lote_sampling_confidence
status: HOMOLOGADO
author: C.Vale DataOps & MLOps Team
last_updated: 2026-07-30
---

# 📌 RN-11: Gateway de Elegibilidade para o Modelo Direto de ML

## 📋 Descrição Geral
A **RN-11** é o roteador de resiliência em produção. Ela impede que lotes com dados insuficientes sofram extrapolações errôneas no modelo preditivo direto.

---

## ⚙️ Regra de Negócio Técnica
```python
if score_confianca_lote >= 7.5 and possui_pesagem_35d == True:
    elegivel_rn11 = 1  # Roteia para Stacking GPU (Modelo Direto ML)
else:
    elegivel_rn11 = 0  # Roteia para Fallback Conservador (Média da Fazenda)
```

---

## 🎯 Impacto no Modelo e DataOps
- **Zero Fail em Produção:** Garante $100\%$ de entrega no PCP, mesmo se o campo falhar na coleta dos dados.
