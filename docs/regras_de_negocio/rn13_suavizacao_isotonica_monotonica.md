---
rule_id: RN-13
title: Detecção de Inversão Biométrica & Suavização Isotônica Monotônica
category: DataOps & Filtro Fisiológico
target_table: longitudinal_dataset
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-13: Detecção de Inversão Biométrica & Suavização Isotônica Monotônica

## 📋 Descrição Geral
A **RN-13** trata da correção de erros de amostragem de balança no campo. Biologicamente, um lote de frangos saudável em crescimento não perde peso corporal significativo entre duas semanas consecutivas ($W_{t+1} < 0,95 W_t$). Quando essa inversão ocorre por erro humano ou amostragem viesada, a RN-13 ajusta a curva.

---

## ⚙️ Regra de Negócio Técnica
```python
from sklearn.isotonic import IsotonicRegression

# Se detectada queda > 5% entre pesagens válidas
if valid_weights[i+1] < valid_weights[i] * 0.95:
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    smoothed_weights = iso.fit_transform(valid_ages, valid_weights)
```

---

## 🎯 Impacto no Modelo e DataOps
- **Fisiologia Garantida:** Restaura a monotonicidade não-decrescente da curva de crescimento do lote.
- **Melhoria no $R^2$:** Eleva a estabilidade das acelerações e velocidades $GMD$ utilizadas pelas árvores de decisão.
