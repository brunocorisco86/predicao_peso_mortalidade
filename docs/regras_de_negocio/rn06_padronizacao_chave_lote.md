---
rule_id: RN-06
title: Padronização da Chave Relacional 1:1 (Lote Composto)
category: Modelagem Dimensional & Arquitetura
target_table: todas
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-06: Padronização da Chave Relacional 1:1 (Lote Composto)

## 📋 Descrição Geral
A **RN-06** estabelece a chave relacional primária universal em todas as tabelas do banco SQLite `database/prediction_data.db`, concatenando o código da fazenda, o código do produtor e o número do lote.

---

## ⚙️ Regra de Negócio Técnica
```sql
lote_composto = CAST(fazenda AS TEXT) || '-' || CAST(produtor AS TEXT) || '-' || CAST(lote AS TEXT)
```

---

## 🎯 Impacto no Modelo e DataOps
- **Integridade de Join:** Permite fusão exata de 1:1 entre biometrias, variáveis zootécnicas, constantes e abate.
