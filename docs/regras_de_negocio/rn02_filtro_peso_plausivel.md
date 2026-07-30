---
rule_id: RN-02
title: Filtro Biológico de Peso de Abate Plausível
category: Filtro Biológico & Qualidade de Dados
target_table: peso_abate
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-02: Filtro Biológico de Peso de Abate Plausível

## 📋 Descrição Geral
A **RN-02** valida o peso médio corporativo dos frangos entregues na plataforma do abatedouro. Pesos abaixo de $1.800	ext{g}$ ou acima de $4.800	ext{g}$ representam inconsistência de digitação (ex: peso em kg gravado como g) ou falha na integração do sistema MTech.

---

## ⚙️ Regra de Negócio Técnica
```sql
WHERE peso_abate_g BETWEEN 1800 AND 4800
```

---

## 🎯 Impacto no Modelo e DataOps
- **Integridade da Escala:** Evita valores fisiologicamente impossíveis no target de regressão.
- **Estabilidade do Loss:** Impede explosão de gradiente ($RMSE$) durante o treinamento em GPU CUDA.
