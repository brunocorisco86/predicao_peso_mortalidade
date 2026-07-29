# Graph Report - prediction_weight_mortality  (2026-07-29)

## Corpus Check
- 35 files · ~253,496 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 114 nodes · 137 edges · 23 communities (12 shown, 11 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.57)
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
- aviary_delta_correction_model.py
- hybrid_arima_gompertz_ml_model.py

## God Nodes (most connected - your core abstractions)
1. `setup_logging()` - 19 edges
2. `🏆 Modelo Campeão: Predição do Peso de Abate com Ground Truth de Frigorífico` - 8 edges
3. `🐥 Predição Exclusiva do Peso de Abate em Aves de Corte (Idade $\ge$ 42 Dias)` - 6 edges
4. `Plano Estratégico de Melhoria Contínua da Predição do Peso de Abate` - 6 edges
5. `Regras de Negócio: Predição do Peso de Abate em Frangos de Corte` - 5 edges
6. `Relatório Final de Consolidação do Projeto: Predição do Peso de Abate em Aves de Corte` - 5 edges
7. `extract_fazenda()` - 4 edges
8. `run_etl()` - 4 edges
9. `🚀 1. Adição de Novas Variáveis de Campo (Engenharia de Dados)` - 4 edges
10. `rationalize_header()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `extract_and_load_peso_abate()` --indirect_call--> `extract_fazenda()`  [INFERRED]
  src/etl/extract_peso_abate.py → src/etl/extract_mtech_data.py

## Import Cycles
- None detected.

## Communities (23 total, 11 thin omitted)

### Community 1 - "extract_mtech_data.py"
Cohesion: 0.31
Nodes (8): extract_fazenda(), process_lote_composto(), Racionaliza o nome do cabeçalho: minúsculas, substitui espaços por underscores,, Processa a coluna 'Lote Composto':     - Traz o dado anterior ao segundo hifen ", Extrai o código da fazenda a partir do lote_composto., rationalize_header(), run_etl(), extract_and_load_peso_abate()

### Community 3 - "🏆 Modelo Campeão: Predição do Peso de Abate de Frangos de Corte"
Cohesion: 0.22
Nodes (8): 📌 1. Resumo Executivo & Diagnóstico de Desempenho, 🎯 2. Alinhamento com os Pilares Estratégicos do Projeto, 📊 3. Estatística Descritiva no Dataset de Abate Sanitizado (`cleaned_data.csv`), 🔬 4. Engenharia de Atributos & Modelo Campeão, 🔁 5. Validação Cruzada (GroupKFold = 5) no Target Oficial de Abate, 📉 6. Matriz de Confusão por Faixas Comerciais de Abate, 🚀 7. Sustentação e Prontidão para Produção, 🏆 Modelo Campeão: Predição do Peso de Abate com Ground Truth de Frigorífico

### Community 4 - "train_predict_weight.py"
Cohesion: 0.67
Nodes (3): gompertz_model(), Gompertz non-linear biological growth curve:     W(t) = A * exp(-b * exp(-k * t), train_and_evaluate()

### Community 7 - "Plano Estratégico de Melhoria Contínua da Predição do Peso de Abate"
Cohesion: 0.20
Nodes (9): 🚀 1. Adição de Novas Variáveis de Campo (Engenharia de Dados), 📈 2. Modelagem Longitudinal e Séries Temporais, 🎯 3. Segmentação Clustered & Regressão Quantílica, A. Variáveis Nutricionais e Alimentos (Impacto Esperado: 🔥🔥🔥 Muito Alto), B. Microclima e Monitoramento IoT (Impacto Esperado: 🔥🔥🔥 Muito Alto), C. Zootecnia e Manejo de Campo (Impacto Esperado: 🔥🔥 Alto), 📌 Diagnóstico Atual do Desempenho, Plano Estratégico de Melhoria Contínua da Predição do Peso de Abate (+1 more)

### Community 8 - "Relatório Final de Consolidação do Projeto: Predição do Peso de Abate em Aves de Corte"
Cohesion: 0.33
Nodes (5): 🎯 Alinhamento aos 3 Pilares Estratégicos, 🛠️ Guia de Execução Rápida dos Scripts, 📄 Licença, Relatório Final de Consolidação do Projeto: Predição do Peso de Abate em Aves de Corte, 🏆 Resumo Executivo da Evolução do Projeto

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
- **33 isolated node(s):** `Settings`, `🎯 1. Simulação de 50 Lotes Aleatórios no Pico de Abate (44 a 46 Dias)`, `📊 Sample de Simulações (Ages 44-46d):`, `📊 Evolução Global das Métricas (5-Fold GroupKFold):`, `🛠️ 3. Guia de Execução` (+28 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `setup_logging()` connect `logger.py` to `extract_mtech_data.py`, `train_predict_weight.py`, `aviary_relative_error_model.py`, `simulate_50_batches_44_46d.py`, `simulate_slaughter_weights.py`, `test_zootecnic_kpis_model.py`, `aviary_delta_correction_model.py`, `hybrid_arima_gompertz_ml_model.py`, `longitudinal_time_series_model.py`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **What connects `Settings`, `🎯 1. Simulação de 50 Lotes Aleatórios no Pico de Abate (44 a 46 Dias)`, `📊 Sample de Simulações (Ages 44-46d):` to the rest of the system?**
  _33 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `logger.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1422924901185771 - nodes in this community are weakly interconnected._