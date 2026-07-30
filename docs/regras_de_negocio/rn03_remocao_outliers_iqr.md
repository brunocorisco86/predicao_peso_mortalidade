---
rule_id: RN-03
title: Eliminação de Outliers de Peso por Idade via IQR
category: Tratamento Estatístico & Sanitização
target_table: extracao_mtech_data
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-03: Eliminação de Outliers de Peso por Idade via IQR

## 📋 Descrição Geral
A **RN-03** realiza a limpeza estatística dos pesos amostrados pelos técnicos de campo ao longo do criatório (MTech). Pesagens discrepantes são filtradas dentro de cada faixa etária de referência utilizando a regra de Tukey ($1,5 	imes 	ext{IQR}$).

---

## ⚙️ Regra de Negócio Técnica
Para cada idade $t \in \{4, 7, 14, 21, 28, 35, 42\}$:
$$	ext{Limite Inferior} = Q_1 - 1,5 	imes 	ext{IQR}$$
$$	ext{Limite Superior} = Q_3 + 1,5 	imes 	ext{IQR}$$

Registros fora dessa faixa são removidos da tabela `extracao_mtech_data`.

---

## 🎯 Impacto no Modelo e DataOps
- **Proteção contra balança descalibrada:** Descarta pesagens pontuais errôneas registradas por erro humano no campo.
- **Suavização da Curva:** Garante que a velocidade de ganho diário ($GMD$) reflita a biometria real do lote.
