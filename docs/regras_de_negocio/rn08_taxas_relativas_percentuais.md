---
rule_id: RN-08
title: Priorização de Taxas Relativas (%) sobre Contagens Absolutas
category: Engenharia de Features Zootécnicas
target_table: variables
status: HOMOLOGADO
author: C.Vale DataOps Team (zootecnia-data-rules)
last_updated: 2026-07-30
---

# 📌 RN-08: Priorização de Taxas Relativas (%) sobre Contagens Absolutas

## 📋 Descrição Geral
A **RN-08** estabelece que variáveis de mortalidade e descartes devem ser expressas como **taxas percentuais relativas ao total alojado** (`cab_alojadas`), eliminando o viés de tamanho do aviário.

---

## ⚙️ Regra de Negócio Técnica
$$	ext{taxa\_mortalidade\_pct} = rac{	ext{cabeças\_mortas}}{	ext{cab\_alojadas}} 	imes 100$$
$$	ext{taxa\_descarte\_pct} = rac{	ext{cabeças\_descartadas}}{	ext{cab\_alojadas}} 	imes 100$$

---

## 🎯 Impacto no Modelo e DataOps
- **Isonomia entre Granjas:** Permite comparar aviários de 15.000 aves com aviários de 40.000 aves sem distorção.
- **Normalização para ML:** Melhora o condicionamento das matrizes em modelos baseados em gradiente.
