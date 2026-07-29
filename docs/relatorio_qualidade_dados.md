# Relatório de Auditoria de Qualidade de Dados (Data Quality Audit)

**Data da Auditoria:** 29 de Julho de 2026
**Banco de Dados Avaliado:** `prediction_data.db`

## 1. Completude (Completeness & Missing Values)

Análise do percentual de nulos em cada tabela:

- **peso_abate** (22.207 linhas): 
  - Nenhuma coluna apresentou valores nulos.
- **variables** (11.283 linhas):
  - Nenhuma coluna apresentou valores nulos.
- **constantes** (1.110 linhas):
  - Colunas `i10` e `i11` apresentam cerca de 0,09% de valores nulos (1 registro cada). Demais completas.
- **extracao_mtech_data** (147.816 linhas):
  - `extensionista`: 19,55% de valores nulos.
  - `nome_fazenda`: 0,40% de valores nulos.
  - `lote_composto` e `fazenda`: 0,13% de nulos.
  - `id_usurio_criao` e `id_usurio`: 0,06% de nulos.

**Conclusão de Completude:** Não foram identificadas colunas com excesso de vazios (acima do limite de 20%). A coluna `extensionista` (19,5%) requer atenção, mas está dentro do limite.

## 2. Unicidade e Duplicatas (Uniqueness & Deduplication)

Análise de linhas duplicadas absolutas e por chaves (`lote_composto` e `fazenda`):

- **peso_abate**: 0 duplicatas absolutas e 0 duplicatas de chave. (Chave Primária 100% única)
- **variables**: 0 duplicatas absolutas e 0 duplicatas de chave. (Chave Primária 100% única)
- **constantes**: 0 duplicatas absolutas.
- **extracao_mtech_data**: 186 duplicatas absolutas. Como esta tabela é um histórico (série temporal), observamos 125.398 repetições para a chave (`lote_composto`, `fazenda`). Isso é esperado, visto que a chave real (granularidade) deveria incluir a `data_evento` ou `idade`. As 186 duplicatas absolutas devem ser removidas.

## 3. Validade e Limites Biológicos Zootécnicos (Validity & Range Checks)

Verificação das regras e limites biológicos:

- **peso_abate**:
  - `peso_abate_g` fora do limite (1.800g a 4.800g): **7 registros**.
  - `peso_abate_g` com valores negativos: **0 registros**.
- **extracao_mtech_data**:
  - `cab_alojadas` inválidas ($\le 0$): **11 registros**.
  - `_mortalidade` fora do limite ($0\% \le taxa \le 100\%$): **678 registros**.
  - `_descartados` fora do limite ($0\% \le taxa \le 100\%$): **2 registros**.

## 4. Integridade Referencial e Órfãos (Referential Integrity)

Verificação de registros órfãos considerando a tabela `variables` como dimensão principal de lotes e `constantes` como dimensão de fazendas:

- Lotes em `extracao_mtech_data` não encontrados em `variables`: **11.135 registros órfãos**.
- Lotes em `peso_abate` não encontrados em `variables`: **10.948 registros órfãos**.
- Fazendas em `extracao_mtech_data` não encontradas em `constantes`: **68 registros órfãos**.
- Fazendas em `variables` não encontradas em `constantes`: **15 registros órfãos**.
- Fazendas em `peso_abate` não encontradas em `constantes`: **64 registros órfãos**.

## 5. Consistência de Tipos e Esquema (Schema Consistency)

O esquema dos dados confere com as expectativas (tipagem do SQLite):
- Os campos de datas estão corretamente tipados como `TEXT` (SQLite não tem tipo Date nativo).
- Valores numéricos fracionários estão como `REAL` e inteiros como `INTEGER`.
- Textos e Chaves como `TEXT`.
- Exceção para o campo `fazenda` em `extracao_mtech_data`, que foi ingerido como `REAL` onde o esperado seria `INTEGER` (ou `TEXT`), enquanto nas outras tabelas está como `INTEGER`.

## 6. Recomendações e Próximos Passos (Cleaning Recommendations)

1. **Limpeza de Duplicadas:**
   - Realizar `DROP DUPLICATES` absoluto na tabela `extracao_mtech_data` para remover as 186 linhas 100% idênticas.
2. **Correção de Anomalias Biológicas (Outliers):**
   - Investigar e tratar (imputação ou exclusão) os 7 pesos de abate anômalos.
   - Corrigir ou remover as 11 medições com cabeças alojadas nulas/negativas.
   - Avaliar as 678 taxas de mortalidade fora de padrão em `extracao_mtech_data`.
3. **Resolução de Órfãos:**
   - Realizar cruzamentos (INNER JOINs) nas modelagens preditivas, descartando temporariamente os dados órfãos, ou atualizar as tabelas dimensão (`variables` e `constantes`) para contemplar todos os lotes e fazendas históricas.
4. **Cast de Tipos:**
   - Converter `fazenda` na tabela `extracao_mtech_data` de `REAL` para `INTEGER` para perfeita correspondência (JOIN) com as demais tabelas.
