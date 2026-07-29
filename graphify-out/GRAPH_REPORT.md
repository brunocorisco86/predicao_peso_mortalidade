# Graph Report - .  (2026-07-29)

## Corpus Check
- 37 files · ~15,000 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 62 nodes · 96 edges · 16 communities (9 shown, 7 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Modulo 0
- Modulo 1
- Modulo 2
- Modulo 3
- Modulo 5
- Modulo 6
- Modulo 8
- Modulo 9
- Modulo 10
- Modulo 12

## God Nodes (most connected - your core abstractions)
1. `setup_logging()` - 19 edges
2. `extract_fazenda()` - 5 edges
3. `run_etl()` - 4 edges
4. `extract_and_load_sheet()` - 3 edges
5. `rationalize_header()` - 3 edges
6. `process_lote_composto()` - 3 edges
7. `gompertz_model()` - 3 edges
8. `extract_and_load_peso_abate()` - 2 edges
9. `gompertz_func()` - 2 edges
10. `run_slaughter_simulations()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `extract_and_load_peso_abate()` --indirect_call--> `extract_fazenda()`  [INFERRED]
  src/etl/extract_peso_abate.py → src/etl/extract_mtech_data.py
- `extract_and_load_sheet()` --indirect_call--> `extract_fazenda()`  [INFERRED]
  src/etl/extract_excel_to_db.py → src/etl/extract_mtech_data.py

## Import Cycles
- None detected.

## Communities (16 total, 7 thin omitted)

### Community 0 - "Modulo 0"
Cohesion: 0.15
Nodes (3): gompertz_func(), run_aviary_delta_correction(), setup_logging()

### Community 1 - "Modulo 1"
Cohesion: 0.24
Nodes (9): extract_and_load_sheet(), Extracts data from a specified Excel sheet, renames columns based on a map,, extract_fazenda(), process_lote_composto(), Racionaliza o nome do cabeçalho: minúsculas, substitui espaços por underscores,, Processa a coluna 'Lote Composto':     - Traz o dado anterior ao segundo hifen ", Extrai o código da fazenda a partir do lote_composto., rationalize_header() (+1 more)

### Community 3 - "Modulo 3"
Cohesion: 0.67
Nodes (3): gompertz_model(), Gompertz non-linear biological growth curve:     W(t) = A * exp(-b * exp(-k * t), train_and_evaluate()

## Knowledge Gaps
- **1 isolated node(s):** `Settings`
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `setup_logging()` connect `Modulo 0` to `Modulo 1`, `Modulo 2`, `Modulo 3`, `Modulo 4`, `Modulo 5`, `Modulo 6`, `Modulo 7`, `Modulo 8`, `Modulo 9`, `Modulo 10`, `Modulo 11`, `Modulo 12`?**
  _High betweenness centrality (0.328) - this node is a cross-community bridge._
- **Why does `extract_fazenda()` connect `Modulo 1` to `Modulo 12`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `extract_fazenda()` (e.g. with `extract_and_load_sheet()` and `run_etl()`) actually correct?**
  _`extract_fazenda()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `run_etl()` (e.g. with `extract_fazenda()` and `process_lote_composto()`) actually correct?**
  _`run_etl()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Settings` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._