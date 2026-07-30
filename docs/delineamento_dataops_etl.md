# Delineamento de DataOps e ETL - C.Vale

Este documento consolida o mapeamento arquitetural do fluxo de dados, a auditoria dos scripts da esteira de ETL e o rastreamento rigoroso de todas as Regras de Negócio (RN-01 a RN-14) aplicadas na predição de peso de abate e mortalidade.

---

## 1. Auditoria da Esteira de ETL e Fluxo de Dados

A esteira de dados foi auditada e apresenta um fluxo sequencial estruturado em três camadas principais:

### 1.1 Extração e Transformação Primária (Bronze $\rightarrow$ Silver)
- **Script:** `src/etl/extract_mtech_data.py`
- **Função:** Lê as extrações brutas do MTech (`.xlsx`), aplica racionalização de cabeçalhos, coerção de tipos, e garante a integridade de colunas nulas (`data_alojamento`). Elimina duplicidades absolutas e filtra as idades válidas (`idade_ref`). 
- **Carga:** Insere os dados limpos na tabela `extracao_mtech_data` no banco SQLite (`database/prediction_data.db`).
- **Garantia de Não Redundância:** Aplica `drop_duplicates` e recria a tabela (`if_exists='replace'`), garantindo processamento idempotente.

### 1.2 Consolidação de Features Longitudinais (Silver $\rightarrow$ Gold)
- **Script:** `src/features/build_longitudinal_features.py`
- **Função:** Constrói o dataset avançado de atributos longitudinais de série temporal por lote. Pivotamento dos pesos biométricos (`peso_d04` até `peso_d42`), cálculo de velocidades de ganho de peso (GMD) e ajuste de curvas individuais de Gompertz.
- **Integração:** Realiza joins com as tabelas de `variables`, `constantes`, `peso_abate`, e tabelas de métricas computadas por outras features.

### 1.3 Exportação da Visão Unificada (Gold $\rightarrow$ Analytics)
- **Script:** `src/etl/export_unified_data.py`
- **Função:** Exporta a visão analítica completa integrando dados da extração MTech, metadados do lote (`peso_abate`, `variables`, `constantes`) e features avançadas criadas por outros módulos (como índices de confiança e features KNN).
- **Carga:** Gera o CSV `unified_data.csv` e `longitudinal_dataset.csv` para posterior modelagem.

---

## 2. Mapeamento das Regras de Negócio (RN-01 a RN-14)

Abaixo é apresentado exatamente onde e como cada regra de negócio está instanciada no pipeline de DataOps e Engenharia de Features.

| Regra | Título / Descrição | Onde é Aplicada no Pipeline de DataOps / ETL |
|---|---|---|
| **RN-01** | **Janela de Idade de Abate** ($42 \le \text{idade} \le 60$) | Aplicada em `src/eda_outliers.py` e queries de `build_longitudinal_features.py`. Filtra apenas registros válidos para abate. |
| **RN-02** | **Limites de Peso de Abate** ($1,80 \text{ kg} \le \text{peso} \le 4,80 \text{ kg}$) | Aplicada em `src/eda_outliers.py` e consultas de abate, eliminando anomalias extremas ou erros de digitação. |
| **RN-03** | **Integridade do Lote** (`cab_alojadas > 0`) | Aplicada nas etapas de `eda_outliers.py`. |
| **RN-04** | **Sanidade Plausível** (Mortalidade acumulada $\le$ Alojadas) | Filtro de coerência de dados de mortalidade nas rotinas de EDA e modelagem. |
| **RN-05** | **Filtro Estatístico de Outliers** | Implementado no `src/eda_outliers.py` utilizando Limites IQR e descartando distorções por idade no momento do abate. |
| **RN-06** | **Identificação da Fazenda** | Implementado em `src/etl/extract_mtech_data.py` pela função `extract_fazenda()` e `process_lote_composto()`. |
| **RN-07** | **Composição do Lote Composto** | Estruturado via concatenações em queries de relacionamento (chaves primárias de joins no ETL). |
| **RN-08** | **Preferência por Taxas Relativas (%)** | Modelos (ex: `aviary_relative_error_model.py`) e visualizações forçam o uso de taxas relativas (`_mortalidade`) sobre contagens brutas. |
| **RN-09** | **Elegibilidade Mínima de Pesagens** | Computado em `src/features/calculate_sampling_confidence.py` (Mínimo de 3 pesagens, exigência da pesagem de 35 dias). |
| **RN-10** | **Índice de Confiança de Amostragem** | Calculado em `calculate_sampling_confidence.py` gerando a tabela `lote_sampling_confidence` no SQLite. |
| **RN-11** | **Delineamento Amostral Mínimo** | Define a elegibilidade final para modelagem ML (`score >= 7.5`). Agregado junto com RN-09/10 e exportado via `export_unified_data.py`. |
| **RN-12** | **Gêmeos Digitais & Imputação Contextual (KNN)** | Implementada por `apply_rn12_knn_digital_twins.py` que cria a tabela `lote_rn12_digital_twins`, cujas features são consolidadas em `build_longitudinal_features.py` e `export_unified_data.py`. |
| **RN-13** | **Inversão Biométrica & Suavização Isotônica** | Implementada diretamente em `src/features/build_longitudinal_features.py` através da regressão isotônica (`IsotonicRegression`), identificando e corrigindo quedas irreais de peso entre semanas ($W_{t+1} < 0.95 W_t$). |
| **RN-14** | **Anonimização LGPD & Codificação Hexadecimal** | Implementada em `src/utils/anonymize_dataset.py`, convertendo aviários e lotes em 4 caracteres hexadecimais (ex: `3A1F-02B4`) e nomes sensíveis via `Faker('pt_BR')`. |

---

## 3. Arquitetura e Integridade Relacional do Banco de Dados

O banco SQLite (`database/prediction_data.db`) atua como um repositório *Silver/Gold* do pipeline de ETL, conectando metadados por meio das chaves relacionais:

- **Chaves Primárias de Conexão:** `lote_composto` e `fazenda` (derivada da RN-06).
- **Modelo Entidade-Relacionamento:** 
  - `extracao_mtech_data`: Tabela fato temporal.
  - `peso_abate`: Variáveis de Target do abate.
  - `variables` / `constantes`: Atributos dimensionais estáticos por lote e fazenda.
  - `lote_sampling_confidence` / `lote_rn12_digital_twins`: Tabelas derivadas das engenharias de features (RN-09 a RN-12).

O modelo garante **ausência de redundâncias** porque os relacionamentos se baseiam em tabelas temporais pivotadas (features longitudinais) agregadas no nível de 1 registro por `lote_composto` nas visões analíticas geradas pelos scripts de exportação.
