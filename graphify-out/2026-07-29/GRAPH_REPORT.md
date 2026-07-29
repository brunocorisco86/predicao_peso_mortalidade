# Graph Report - prediction_weight_mortality  (2026-07-29)

## Corpus Check
- 34 files · ~215,274 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 117 nodes · 133 edges · 20 communities (13 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `aa49b335`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- settings.py
- extract_mtech_data.py
- logger.py
- 🏆 Modelo Campeão: Predição do Peso de Abate de Frangos de Corte
- train_predict_weight.py
- Plano Estratégico de Melhoria Contínua da Predição do Peso de Abate
- Relatório Final de Consolidação do Projeto: Predição do Peso de Abate em Aves de Corte
- 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)
- Regras de Negócio: Predição do Peso de Abate em Frangos de Corte
- Premissas do Projeto: Predição de Peso e Mortalidade de Aves
- Modelo Entidade Relacionamento (MER) - Prediction Weight Mortality Database
- Roteiro do Projeto (Workflow): Predição de Peso e Mortalidade
- aviary_relative_error_model.py
- simulate_50_batches_44_46d.py
- simulate_slaughter_weights.py
- test_zootecnic_kpis_model.py
- explicabilidade_eli5.md

## God Nodes (most connected - your core abstractions)
1. `setup_logging()` - 17 edges
2. `🏆 Modelo Campeão: Predição do Peso de Abate de Frangos de Corte` - 10 edges
3. `🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)` - 6 edges
4. `Plano Estratégico de Melhoria Contínua da Predição do Peso de Abate` - 6 edges
5. `Regras de Negócio: Predição do Peso de Abate em Frangos de Corte` - 5 edges
6. `Relatório Final de Consolidação do Projeto: Predição do Peso de Abate em Aves de Corte` - 5 edges
7. `run_etl()` - 4 edges
8. `🚀 1. Adição de Novas Variáveis de Campo (Engenharia de Dados)` - 4 edges
9. `rationalize_header()` - 3 edges
10. `process_lote_composto()` - 3 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (20 total, 7 thin omitted)

### Community 0 - "settings.py"
Cohesion: 0.25
Nodes (3): Settings, extract_and_load_sheet(), Extracts data from a specified Excel sheet, renames columns based on a map,

### Community 1 - "extract_mtech_data.py"
Cohesion: 0.36
Nodes (7): extract_fazenda(), process_lote_composto(), Racionaliza o nome do cabeçalho: minúsculas, substitui espaços por underscores,, Processa a coluna 'Lote Composto':     - Traz o dado anterior ao segundo hifen ", Extrai o código da fazenda a partir do lote_composto., rationalize_header(), run_etl()

### Community 2 - "logger.py"
Cohesion: 0.12
Nodes (5): gompertz_func(), run_aviary_delta_correction(), gompertz_func(), run_hybrid_tri_model(), setup_logging()

### Community 3 - "🏆 Modelo Campeão: Predição do Peso de Abate de Frangos de Corte"
Cohesion: 0.15
Nodes (12): 📌 1. Resumo Executivo & Diagnóstico de Desempenho, 🎯 2. Alinhamento com os Pilares Estratégicos do Projeto, 📊 3. Estatística Descritiva & Análise Exploratória (EDA), 🔬 4. Arquitetura do Modelo & Engenharia de Atributos, 🔁 5. Resultados de Validação Cruzada (GroupKFold = 5), 📉 6. Análise de Resíduos (Out-of-Fold), 🎯 7. Matriz de Confusão por Faixa Comercial de Peso, 📊 8. Tabela Comparativa da Evolução dos Modelos (+4 more)

### Community 4 - "train_predict_weight.py"
Cohesion: 0.67
Nodes (3): gompertz_model(), Gompertz non-linear biological growth curve:     W(t) = A * exp(-b * exp(-k * t), train_and_evaluate()

### Community 7 - "Plano Estratégico de Melhoria Contínua da Predição do Peso de Abate"
Cohesion: 0.20
Nodes (9): 🚀 1. Adição de Novas Variáveis de Campo (Engenharia de Dados), 📈 2. Modelagem Longitudinal e Séries Temporais, 🎯 3. Segmentação Clustered & Regressão Quantílica, A. Variáveis Nutricionais e Alimentos (Impacto Esperado: 🔥🔥🔥 Muito Alto), B. Microclima e Monitoramento IoT (Impacto Esperado: 🔥🔥🔥 Muito Alto), C. Zootecnia e Manejo de Campo (Impacto Esperado: 🔥🔥 Alto), 📌 Diagnóstico Atual do Desempenho, Plano Estratégico de Melhoria Contínua da Predição do Peso de Abate (+1 more)

### Community 8 - "Relatório Final de Consolidação do Projeto: Predição do Peso de Abate em Aves de Corte"
Cohesion: 0.25
Nodes (7): 1. Subagente `explicador_avanco`, 2. Subagente `facilitador_implementacao`, 🛠️ Guia de Execução Rápida dos Scripts, 📄 Licença, Relatório Final de Consolidação do Projeto: Predição do Peso de Abate em Aves de Corte, 🏆 Resumo Executivo da Evolução do Projeto, 🔬 Subagentes Especializados Criados

### Community 9 - "🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)"
Cohesion: 0.25
Nodes (7): 🎯 1. Simulação de 50 Lotes Aleatórios no Pico de Abate (44 a 46 Dias), 🌾 2. KPIs Zootécnicos e Dimensão por Aviário, 🛠️ 3. Guia de Execução, 📊 Evolução Global das Métricas (5-Fold GroupKFold):, 📄 Licença, 🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias), 📊 Sample de Simulações (Ages 44-46d):

### Community 10 - "Regras de Negócio: Predição do Peso de Abate em Frangos de Corte"
Cohesion: 0.33
Nodes (5): 📋 1. Escopo e Objetivo de Negócio, ⚙️ 2. Regras de Negócio e Filtros Biológicos, 🎯 3. Classificação de Desempenho do Lote ao Abate (Target Categorizado), 🧠 4. Variáveis Explicativas do Lote, Regras de Negócio: Predição do Peso de Abate em Frangos de Corte

### Community 11 - "Premissas do Projeto: Predição de Peso e Mortalidade de Aves"
Cohesion: 0.50
Nodes (3): 1. Estrutura e Qualidade dos Dados, 2. Modelagem Preditiva de Crescimento, Premissas do Projeto: Predição de Peso e Mortalidade de Aves

## Knowledge Gaps
- **36 isolated node(s):** `Settings`, `🎯 1. Simulação de 50 Lotes Aleatórios no Pico de Abate (44 a 46 Dias)`, `📊 Sample de Simulações (Ages 44-46d):`, `📊 Evolução Global das Métricas (5-Fold GroupKFold):`, `🛠️ 3. Guia de Execução` (+31 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `setup_logging()` connect `logger.py` to `settings.py`, `extract_mtech_data.py`, `train_predict_weight.py`, `aviary_relative_error_model.py`, `simulate_50_batches_44_46d.py`, `simulate_slaughter_weights.py`, `test_zootecnic_kpis_model.py`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **What connects `Settings`, `🎯 1. Simulação de 50 Lotes Aleatórios no Pico de Abate (44 a 46 Dias)`, `📊 Sample de Simulações (Ages 44-46d):` to the rest of the system?**
  _36 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `logger.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11666666666666667 - nodes in this community are weakly interconnected._