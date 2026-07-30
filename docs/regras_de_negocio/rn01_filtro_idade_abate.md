---
rule_id: RN-01
title: Filtro Biológico de Idade de Abate Frigorífico
category: Filtro Biológico & Qualidade de Dados
target_table: peso_abate
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-01: Filtro Biológico de Idade de Abate Frigorífico

## 📋 Descrição Geral
A **RN-01** define o intervalo etário comercialmente aceito para o abate de frangos de corte nas plantas industriais da C.Vale. Registros fora da janela etária de $42 \le 	ext{idade\_abate} \le 60	ext{ dias}$ são descartados por caracterizarem erros de digitação ou lotes experimentais/descarte extraordinário.

---

## ⚙️ Regra de Negócio Técnica
```sql
WHERE idade_abate >= 42 AND idade_abate <= 60
```

---

## 🎯 Impacto no Modelo e DataOps
- **Eliminação de Ruído:** Remove lotes abatidos precocemente (ex: 20 dias por emergência sanitária) ou aves matrizes velhas (>60 dias).
- **Consistência do Target:** Garante que a variável dependente `peso_abate_g` reflita a janela operacional de expedição do PCP.
