---
rule_id: RN-07
title: Coerção de Formato do Código da Fazenda
category: Qualidade de Dados & Tipagem
target_table: constantes
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-07: Coerção de Formato do Código da Fazenda

## 📋 Descrição Geral
A **RN-07** força a tipagem do código da fazenda para valor inteiro, eliminando formatações incorretas de ponto flutuante (ex: `105.0`) originadas da importação de arquivos Excel/CSV.

---

## ⚙️ Regra de Negócio Técnica
```python
df['fazenda'] = df['fazenda'].fillna(0).astype(int)
```

---

## 🎯 Impacto no Modelo e DataOps
- **Sanidade de Chave Extrangeira:** Impede falhas de junção entre a tabela `constantes` e as demais tabelas Fato.
