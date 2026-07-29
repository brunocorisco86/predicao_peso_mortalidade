# Modelo Entidade Relacionamento (MER) - Prediction Weight Mortality Database

Este documento descreve o esquema do banco de dados `database/prediction_data.db` e as relações entre as tabelas de fatos e dimensões do projeto.

```mermaid
erDiagram
    extracao_mtech_data ||--o{ variables : "lote_composto"
    extracao_mtech_data ||--o{ constantes : "fazenda"

    extracao_mtech_data {
        TEXT data_alojamento
        TEXT nome_fazenda
        TEXT lote_composto
        TEXT data_evento
        REAL idade
        REAL cab_alojadas
        REAL estoque_aves
        INTEGER mortalidade
        REAL _mortalidade
        INTEGER descartados
        REAL _descartados
        REAL peso
        REAL fazenda
    }

    variables {
        TEXT data_alojamento
        TEXT lote_composto
        TEXT produtor
        INTEGER f01 "Vazio curto"
        INTEGER f02 "Vazio médio"
        INTEGER f03 "Vazio longo"
        INTEGER f04 "Número de camas 1-4"
        INTEGER c05 "Idade matriz baixa"
        INTEGER c06 "Idade matriz alta"
        INTEGER c15 "Peso pintainho (g)"
        TEXT c16 "Linhagem"
        TEXT c17 "Fornecedor"
    }

    constantes {
        INTEGER fazenda
        INTEGER nucleo
        INTEGER a01 "Aviário Convencional"
        INTEGER a02 "Aviário Climatizado"
        INTEGER a03 "Aviário Dark House"
        INTEGER a08 "Aquecimento Fornalha"
        INTEGER a09 "Aquecimento Campânulas"
        INTEGER x02 "Distância Abatedouro (km)"
    }
```

## Resumo das Tabelas no Banco SQLite (`prediction_data.db`)

1. **`extracao_mtech_data` (Tabela Fato - 104.601 registros):**
   - Armazena as pesagens diárias, mortes, descartes e dados operacionais de campo.
   - Chaves de Junção: `lote_composto` (para a tabela `variables`) e `fazenda` (para a tabela `constantes`).

2. **`variables` (Tabela de Atributos do Lote - 11.283 registros):**
   - Contém informações específicas do lote alojado, como linhagem do pintainho (`c16`), peso inicial do pintainho (`c15`), tempo de vazio sanitário (`f01`, `f02`, `f03`) e idade da matriz (`c05`, `c06`).

3. **`constantes` (Tabela de Infraestrutura da Fazenda - 1.110 registros):**
   - Armazena a caracterização técnica do aviário, como tipo de tecnologia (`a01` convencional, `a02` climatizado, `a03` dark house), sistema de aquecimento (`a08`, `a09`) e distância do abatedouro (`x02`).

4. **Visão Unificada (`data/processed/unified_data.csv` e `cleaned_data.csv`):**
   - Resultado do `LEFT JOIN` entre a tabela fato e as dimensões via `lote_composto` e `fazenda`, contendo 68 colunas e 88.021 observações limpas prontas para modelagem.
