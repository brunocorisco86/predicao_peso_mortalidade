# Modelo Entidade Relacionamento (MER) - Prediction Weight Mortality Database

Este documento descreve o esquema do banco de dados `prediction_data.db`, que armazena dados relacionados à predição de peso e mortalidade na avicultura.

## Tabelas

### Tabela: extracao_mtech_data

| Coluna | Tipo de Dado | Descrição |
|---|---|---|
| `data_alojamento` | TEXT | Data de alojamento do lote. |
| `nome_fazenda` | TEXT | Nome da fazenda. |
| `data_hora_transao` | TEXT | Data e hora da transação. |
| `lote_composto` | TEXT | Identificador composto do lote. |
| `data_evento` | TEXT | Data do evento. |
| `idade` | REAL | Idade (provavelmente em dias ou semanas). |
| `cab_alojadas` | REAL | Cabeças alojadas. |
| `estoque_aves` | REAL | Estoque de aves. |
| `mortalidade` | INTEGER | Número de aves mortas. |
| `_mortalidade` | REAL | Taxa de mortalidade (%). |
| `descartados` | INTEGER | Número de aves descartadas. |
| `_descartados` | REAL | Taxa de aves descartadas (%). |
| `_mortalidade__descartados` | REAL | Taxa combinada de mortalidade e descartes (%). |
| `data_criao` | TEXT | Data de criação. |
| `peso` | REAL | Peso das aves. |
| `id_usurio_criao` | TEXT | ID do usuário que criou o registro. |
| `extensionista` | TEXT | Nome do extensionista. |
| `id_usurio` | TEXT | ID do usuário. |
| `fazenda` | REAL | Número do aviário. |

### Tabela: variables

| Coluna | Tipo de Dado | Descrição |
|---|---|---|
| `data_alojamento` | TEXT | Data de alojamento do lote. |
| `lote_composto` | TEXT | Identificador composto do lote. |
| `produtor` | TEXT | Nome do produtor. |
| `f01` | INTEGER | Vazio curto (14 a 18 dias). |
| `f02` | INTEGER | Vazio médio (19 a 24 dias). |
| `f03` | INTEGER | Vazio longo (mais de 25 dias). |
| `f04` | INTEGER | Número de camas (1 a 4). |
| `f05` | INTEGER | Número de camas (5 a 9). |
| `f06` | INTEGER | Número de camas (mais de 10). |
| `c05` | INTEGER | Idade da matriz baixa. |
| `c06` | INTEGER | Idade da matriz alta. |
| `c11` | INTEGER | Fator multiplicador peso 35 dias abaixo. |
| `c12` | INTEGER | Fator multiplicador peso 35 dias acima. |
| `y03` | INTEGER | Lote abatido no primeiro turno. |
| `y04` | INTEGER | Lote abatido no segundo turno. |
| `c15` | INTEGER | Peso do pintainho em gramas. |
| `c16` | TEXT | Linhagem do pintainho. |
| `c17` | TEXT | Fornecedor do pintainho. |
| `f07` | INTEGER | Lotes medicados até os 14 dias. |

### Tabela: constantes

| Coluna | Tipo de Dado | Descrição |
|---|---|---|
| `fazenda` | INTEGER | Identificador da fazenda. |
| `nucleo` | INTEGER | Identificador do núcleo. |
| `a01` | INTEGER | Aviário Convencional. |
| `a02` | INTEGER | Aviário Climatizado. |
| `a03` | INTEGER | Aviário Dark House. |
| `a08` | INTEGER | Aquecimento principal por Fornalha. |
| `a09` | INTEGER | Aquecimento principal por Campânulas. |
| `a10` | INTEGER | Climatização adequada. |
| `a11` | INTEGER | Climatização inadequada. |
| `c07` | INTEGER | IEP seco alto (ranking). |
| `c08` | INTEGER | IEP seco baixo (ranking). |
| `d01` | INTEGER | Microrregiões com desafio de sanidade (aerossaculite). |
| `d02` | INTEGER | Microrregiões sem desafio de sanidade (aerossaculite). |
| `d03` | INTEGER | Produtores com alto desafio de aerossaculite. |
| `d04` | INTEGER | Produtores com baixo desafio de aerossaculite. |
| `d05` | INTEGER | Produtores com desafio de aerossaculite dentro da média. |
| `f15` | INTEGER | Tempo de jejum em horas. |
| `h01` | INTEGER | Produtores com alto rendimento financeiro (10% top). |
| `h02` | INTEGER | Produtores com baixo rendimento financeiro (10% bottom). |
| `i02` | INTEGER | CLASSIFICAÇÃO AVIARIO A. |
| `i03` | INTEGER | CLASSIFICAÇÃO AVIARIO B. |
| `i04` | INTEGER | CLASSIFICAÇÃO AVIARIO C. |
| `i05` | INTEGER | CLASSIFICAÇÃO AVIARIO D. |
| `i06` | INTEGER | CLASSIFICAÇÃO AVIARIO E. |
| `i07` | INTEGER | Aquecimento nota zero. |
| `i08` | INTEGER | Aviário Global Gap. |
| `i09` | INTEGER | Participa do projeto PICS. |
| `i10` | REAL | Assertividade - peso abaixo. |
| `i11` | REAL | Assertividade - peso acima. |
| `i12` | REAL | Densidade Nominal. |
| `x01` | INTEGER | Distância abatedouro longe/perto. |
| `x02` | INTEGER | Distância abatedouro em km. |
| `i01` | INTEGER | Nota Promob Acima de 90%. |
