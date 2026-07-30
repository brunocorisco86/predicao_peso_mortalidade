# Graph Report - /home/brunoconter/Documentos/1_C.VALE/1 - ANALISES/10 - PESO DAS AVES/prediction_weight_mortality  (2026-07-30)

## Corpus Check
- 43 files · ~45,000 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 153 nodes · 194 edges · 30 communities (15 shown, 15 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Comunidade 0
- Comunidade 1
- Comunidade 2
- Comunidade 3
- Comunidade 4
- Comunidade 5
- Comunidade 6
- Comunidade 7
- Comunidade 8
- Comunidade 9
- Comunidade 10
- Comunidade 11
- Comunidade 12
- Comunidade 13
- Comunidade 14
- Comunidade 15
- Comunidade 16
- Comunidade 17
- Comunidade 18
- Comunidade 19
- Comunidade 20
- Comunidade 21
- Comunidade 22
- Comunidade 23
- Comunidade 24
- Comunidade 25
- Comunidade 26

## God Nodes (most connected - your core abstractions)
1. `setup_logging()` - 17 edges
2. `main()` - 10 edges
3. `extract_fazenda()` - 5 edges
4. `gompertz_func()` - 5 edges
5. `fit_global_gompertz()` - 5 edges
6. `predict_batch_gompertz()` - 5 edges
7. `run_etl()` - 4 edges
8. `TabularResNet` - 4 edges
9. `load_data()` - 4 edges
10. `main()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `extract_and_load_peso_abate()` --indirect_call--> `extract_fazenda()`  [INFERRED]
  src/etl/extract_peso_abate.py → src/etl/extract_mtech_data.py
- `extract_and_load_sheet()` --indirect_call--> `extract_fazenda()`  [INFERRED]
  src/etl/extract_excel_to_db.py → src/etl/extract_mtech_data.py

## Import Cycles
- None detected.

## Communities (30 total, 15 thin omitted)

### Community 1 - "Comunidade 1"
Cohesion: 0.21
Nodes (17): analyze_residuals(), evaluate_by_confidence(), evaluate_classification_metrics(), fit_global_gompertz(), generate_markdown_report(), generate_plots(), gompertz_func(), load_and_prepare_data() (+9 more)

### Community 2 - "Comunidade 2"
Cohesion: 0.21
Nodes (10): extract_and_load_sheet(), Extracts data from a specified Excel sheet, renames columns based on a map,, extract_fazenda(), process_lote_composto(), Racionaliza o nome do cabeçalho: minúsculas, substitui espaços por underscores,, Processa a coluna 'Lote Composto':     - Traz o dado anterior ao segundo hifen ", Extrai o código da fazenda a partir do lote_composto., rationalize_header() (+2 more)

### Community 3 - "Comunidade 3"
Cohesion: 0.25
Nodes (10): evaluate_model(), extract_field_features(), load_data(), main(), normalize_peso_kg(), field_weighing_weight_model.py -------------------------------- Modelo de Prediç, Calcula e exibe as métricas globais e por idade de amostragem., Uniformiza a escala de peso para Quilogramas (kg), convertendo gramas (> 10.0) p (+2 more)

### Community 4 - "Comunidade 4"
Cohesion: 0.40
Nodes (4): apply_rn13_isotonic_smoothing(), build_longitudinal_features(), build_longitudinal_features.py -------------------------------- Constrói o datas, RN-13: Identifica Inversão Biométrica (queda de peso > 5% entre pesagens)      e

### Community 5 - "Comunidade 5"
Cohesion: 0.40
Nodes (3): pytorch_tabular_model.py ------------------------- Rede Neural Profunda Tabular, run_pytorch_model(), TabularResNet

### Community 6 - "Comunidade 6"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), evaluate_final_champion_model.py ---------------------------------- Avaliação Co, run_champion_evaluation()

### Community 7 - "Comunidade 7"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), gpu_optimization_loop.py ------------------------- Loop de Otimização Preditiva, run_gpu_optimization_loop()

### Community 8 - "Comunidade 8"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), optimize_residual_blend.py ---------------------------- Otimização Bayesiana de, run_residual_optimization()

### Community 9 - "Comunidade 9"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), push_mae_below_100.py ----------------------- Ajuste fino final para romper a ba, run_fine_tuning()

### Community 10 - "Comunidade 10"
Cohesion: 0.67
Nodes (3): gompertz_model(), Gompertz non-linear biological growth curve:     W(t) = A * exp(-b * exp(-k * t), train_and_evaluate()

### Community 11 - "Comunidade 11"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), train_xgb_oof_target_encoding.py --------------------------------- XGBoost GPU C, run_oof_target_encoding_experiment()

### Community 12 - "Comunidade 12"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), train_xgb_residual_target.py ------------------------------ Treinamento do XGBoo, run_residual_target_experiment()

## Knowledge Gaps
- **1 isolated node(s):** `Settings`
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `setup_logging()` connect `Comunidade 0` to `Comunidade 2`, `Comunidade 10`, `Comunidade 19`, `Comunidade 22`, `Comunidade 23`, `Comunidade 24`, `Comunidade 25`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `extract_fazenda()` (e.g. with `extract_and_load_sheet()` and `run_etl()`) actually correct?**
  _`extract_fazenda()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Settings` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Comunidade 0` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._