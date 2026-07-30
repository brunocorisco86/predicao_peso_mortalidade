---
rule_id: RN-04
title: Eliminação de Duplicatas Absolutas de Registro
category: Integridade de Dados & Governança
target_table: extracao_mtech_data
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-04: Eliminação de Duplicatas Absolutas de Registro

## 📋 Descrição Geral
A **RN-04** elimina registros idênticos sincronizados repetidamente pelo sistema de integração do MTech durante falhas de rede ou sincronizações duplas de dispositivos móveis da extensão rural.

---

## ⚙️ Regra de Negócio Técnica
```sql
DELETE FROM extracao_mtech_data 
WHERE rowid NOT IN (
    SELECT MIN(rowid) 
    FROM extracao_mtech_data 
    GROUP BY lote_composto, idade, peso, data_hora_transacao
);
```

---

## 🎯 Impacto no Modelo e DataOps
- **Integridade Relacional:** Evita ponderação dupla de pesagens na média diária do lote.
