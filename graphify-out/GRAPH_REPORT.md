# Graph Report - .  (2026-07-29)

## Corpus Check
- 37 files · ~173,226 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 32 nodes · 42 edges · 7 communities (6 shown, 1 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 3 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Modulo 0
- Modulo 1
- Modulo 2
- Modulo 3
- Modulo 4

## God Nodes (most connected - your core abstractions)
1. `setup_logging()` - 5 edges
2. `run_etl()` - 4 edges
3. `calculate_descriptive_stats()` - 3 edges
4. `rationalize_header()` - 3 edges
5. `process_lote_composto()` - 3 edges
6. `extract_fazenda()` - 3 edges
7. `gompertz_model()` - 3 edges
8. `main()` - 2 edges
9. `extract_and_load_sheet()` - 2 edges
10. `train_and_evaluate()` - 2 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (7 total, 1 thin omitted)

### Community 0 - "Modulo 0"
Cohesion: 0.25
Nodes (3): Settings, extract_and_load_sheet(), Extracts data from a specified Excel sheet, renames columns based on a map,…

### Community 1 - "Modulo 1"
Cohesion: 0.36
Nodes (7): extract_fazenda(), process_lote_composto(), Racionaliza o nome do cabeçalho: minúsculas, substitui espaços por underscores,…, Processa a coluna 'Lote Composto': - Traz o dado anterior ao segundo hifen "-"…, Extrai o código da fazenda a partir do lote_composto., rationalize_header(), run_etl()

### Community 3 - "Modulo 3"
Cohesion: 0.67
Nodes (3): calculate_descriptive_stats(), main(), Calcule estatística descritiva completa para colunas especificadas.

### Community 4 - "Modulo 4"
Cohesion: 0.67
Nodes (3): gompertz_model(), Gompertz non-linear growth curve: W(t) = A * exp(-b * exp(-k * t)) t: age in…, train_and_evaluate()

## Knowledge Gaps
- **1 isolated node(s):** `Settings`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `setup_logging()` connect `Modulo 2` to `Modulo 1`, `Modulo 4`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `run_etl()` (e.g. with `extract_fazenda()` and `process_lote_composto()`) actually correct?**
  _`run_etl()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Settings` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._