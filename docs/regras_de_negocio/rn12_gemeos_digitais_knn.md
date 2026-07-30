---
rule_id: RN-12
title: Gêmeos Digitais por Matriz KNN de Fazenda
category: Feature Engineering & Imputação Contextual
target_table: lote_rn12_digital_twins
status: HOMOLOGADO
author: C.Vale Data Science Team
last_updated: 2026-07-30
---

# 📌 RN-12: Gêmeos Digitais por Matriz KNN de Fazenda

## 📋 Descrição Geral
A **RN-12** implementa a estratégia de Gêmeos Digitais. O sistema identifica os $K=15$ lotes históricos mais parecidos da mesma fazenda e aviário para fornecer um baseline biológico robusto.

---

## ⚙️ Regra de Negócio Técnica
- Matriz de Distância Euclidiana Padronizada via `StandardScaler`.
- Imputação e projeção via `sklearn.neighbors.NearestNeighbors(n_neighbors=15)`.
- Treinamento estritamente dentro dos folds de validação cruzada `GroupKFold` por `lote_composto` (Zero Leakage).

---

## 🎯 Impacto no Modelo e DataOps
- **Recuperação de Dados Faltantes:** Permite estimar a tendência da linhagem mesmo quando a sigla `c16` não foi cadastrada no incubatório.
