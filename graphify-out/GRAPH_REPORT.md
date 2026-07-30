# Graph Report - /home/brunoconter/Documentos/1_C.VALE/1 - ANALISES/10 - PESO DAS AVES/prediction_weight_mortality  (2026-07-30)

## Corpus Check
- 45 files · ~55,000 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 159 nodes · 198 edges · 31 communities (16 shown, 15 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Infraestrutura & ETL SQLite
- Engenharia de Features Longitudinais & RN-13
- Modelagem Preditiva Stacking & OOF Target Encoding
- Explicabilidade SHAP & Diagnóstico de Resíduos
- Suíte de Gráficos Zootécnicos & Estatísticos
- Serviços de Utilidades & Formatação Zootécnica
- Módulo do Projeto 6
- Módulo do Projeto 7
- Módulo do Projeto 8
- Módulo do Projeto 9
- Módulo do Projeto 10
- Módulo do Projeto 11
- Módulo do Projeto 12
- Módulo do Projeto 13
- Módulo do Projeto 14
- Módulo do Projeto 15
- Módulo do Projeto 16
- Módulo do Projeto 17
- Módulo do Projeto 18
- Módulo do Projeto 19
- Módulo do Projeto 20
- Módulo do Projeto 21
- Módulo do Projeto 22
- Módulo do Projeto 23
- Módulo do Projeto 24
- Módulo do Projeto 25
- Módulo do Projeto 26
- Módulo do Projeto 27

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

## Communities (31 total, 15 thin omitted)

### Community 0 - "Infraestrutura & ETL SQLite"
Cohesion: 0.11
Nodes (4): Settings, gompertz_func(), run_aviary_delta_correction(), setup_logging()

### Community 1 - "Engenharia de Features Longitudinais & RN-13"
Cohesion: 0.21
Nodes (17): analyze_residuals(), evaluate_by_confidence(), evaluate_classification_metrics(), fit_global_gompertz(), generate_markdown_report(), generate_plots(), gompertz_func(), load_and_prepare_data() (+9 more)

### Community 2 - "Modelagem Preditiva Stacking & OOF Target Encoding"
Cohesion: 0.21
Nodes (10): extract_and_load_sheet(), Extracts data from a specified Excel sheet, renames columns based on a map,, extract_fazenda(), process_lote_composto(), Racionaliza o nome do cabeçalho: minúsculas, substitui espaços por underscores,, Processa a coluna 'Lote Composto':     - Traz o dado anterior ao segundo hifen ", Extrai o código da fazenda a partir do lote_composto., rationalize_header() (+2 more)

### Community 3 - "Explicabilidade SHAP & Diagnóstico de Resíduos"
Cohesion: 0.25
Nodes (10): evaluate_model(), extract_field_features(), load_data(), main(), normalize_peso_kg(), field_weighing_weight_model.py -------------------------------- Modelo de Prediç, Calcula e exibe as métricas globais e por idade de amostragem., Uniformiza a escala de peso para Quilogramas (kg), convertendo gramas (> 10.0) p (+2 more)

### Community 4 - "Suíte de Gráficos Zootécnicos & Estatísticos"
Cohesion: 0.40
Nodes (4): apply_rn13_isotonic_smoothing(), build_longitudinal_features(), build_longitudinal_features.py -------------------------------- Constrói o datas, RN-13: Identifica Inversão Biométrica (queda de peso > 5% entre pesagens)      e

### Community 5 - "Serviços de Utilidades & Formatação Zootécnica"
Cohesion: 0.40
Nodes (3): pytorch_tabular_model.py ------------------------- Rede Neural Profunda Tabular, run_pytorch_model(), TabularResNet

### Community 6 - "Módulo do Projeto 6"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), evaluate_final_champion_model.py ---------------------------------- Avaliação Co, run_champion_evaluation()

### Community 7 - "Módulo do Projeto 7"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), gpu_optimization_loop.py ------------------------- Loop de Otimização Preditiva, run_gpu_optimization_loop()

### Community 8 - "Módulo do Projeto 8"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), optimize_residual_blend.py ---------------------------- Otimização Bayesiana de, run_residual_optimization()

### Community 9 - "Módulo do Projeto 9"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), push_mae_below_100.py ----------------------- Ajuste fino final para romper a ba, run_fine_tuning()

### Community 10 - "Módulo do Projeto 10"
Cohesion: 0.67
Nodes (3): gompertz_model(), Gompertz non-linear biological growth curve:     W(t) = A * exp(-b * exp(-k * t), train_and_evaluate()

### Community 11 - "Módulo do Projeto 11"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), train_xgb_oof_target_encoding.py --------------------------------- XGBoost GPU C, run_oof_target_encoding_experiment()

### Community 12 - "Módulo do Projeto 12"
Cohesion: 0.67
Nodes (3): mean_absolute_percentage_error(), train_xgb_residual_target.py ------------------------------ Treinamento do XGBoo, run_residual_target_experiment()

## Knowledge Gaps
- **1 isolated node(s):** `Settings`
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `setup_logging()` connect `Infraestrutura & ETL SQLite` to `Modelagem Preditiva Stacking & OOF Target Encoding`, `Módulo do Projeto 10`, `Módulo do Projeto 23`, `Módulo do Projeto 24`, `Módulo do Projeto 25`, `Módulo do Projeto 26`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `extract_fazenda()` (e.g. with `extract_and_load_sheet()` and `run_etl()`) actually correct?**
  _`extract_fazenda()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Settings` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Infraestrutura & ETL SQLite` be split into smaller, more focused modules?**
  _Cohesion score 0.10837438423645321 - nodes in this community are weakly interconnected._