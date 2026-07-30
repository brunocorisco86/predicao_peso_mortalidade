"""
run_traditional_graphify.py
----------------------------
Executa o Graphify pelo método tradicional determinístico (AST Python + Comunidades NetworkX),
gerando um grafo limpo, preciso e direto do projeto C.Vale.
"""

import json
from pathlib import Path
from graphify.extract import extract
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json
from graphify.exporters.html import to_html

def run_traditional():
    scan_dir = Path('.').resolve()
    print("=======================================================")
    print(" 🛠️ RUNNING GRAPHIFY TRADICIONAL (AST DETERMINÍSTICO)")
    print("=======================================================")
    
    project_files = [
        p for p in scan_dir.rglob('*.py')
        if not any(part.startswith('.') or part in ['venv', 'graphify-out', 'site-packages', 'dist-packages'] for part in p.parts)
    ]
    print(f"Mapeando e extraindo AST de {len(project_files)} arquivos Python do projeto...")
    
    ast_result = extract(project_files, cache_root=scan_dir)
    G = build_from_json(ast_result, root=str(scan_dir), directed=False)
    print(f"Grafo construído: {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")
    
    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    
    # Nomes descritivos traduzidos para as comunidades do projeto
    labels = {
        0: "Infraestrutura & ETL SQLite",
        1: "Engenharia de Features Longitudinais & RN-13",
        2: "Modelagem Preditiva Stacking & OOF Target Encoding",
        3: "Explicabilidade SHAP & Diagnóstico de Resíduos",
        4: "Suíte de Gráficos Zootécnicos & Estatísticos",
        5: "Serviços de Utilidades & Formatação Zootécnica"
    }
    # Preencher outras comunidades se houver
    for cid in communities:
        if cid not in labels:
            labels[cid] = f"Módulo do Projeto {cid}"
            
    questions = suggest_questions(G, communities, labels)
    
    out_dir = Path('graphify-out')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Remover graph.json antigo do Ollama se existir para permitir sobrescrita limpa
    if (out_dir / 'graph.json').exists():
        (out_dir / 'graph.json').unlink()
    
    # Salvar graph.json
    to_json(G, communities, str(out_dir / 'graph.json'), force=True)
    
    # Salvar GRAPH_REPORT.md
    detection = {
        'total_files': len(project_files),
        'total_words': 55000,
        'files': {'code': [str(f.relative_to(scan_dir)) for f in project_files]}
    }
    tokens = {'input': 0, 'output': 0}
    report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, str(scan_dir), suggested_questions=questions)
    Path(out_dir / 'GRAPH_REPORT.md').write_text(report, encoding='utf-8')
    
    # Salvar labels
    Path(out_dir / '.graphify_labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False), encoding='utf-8')
    
    # Gerar graph.html tradicional
    to_html(G, communities, str(out_dir / 'graph.html'))
    
    print("\n=======================================================")
    print(" ✅ GRAPHIFY TRADICIONAL GERADO COM SUCESSO!")
    print("=======================================================")
    print(f" Nós: {G.number_of_nodes()} | Arestas: {G.number_of_edges()} | Comunidades: {len(communities)}")
    print(f" Visualizador HTML: {out_dir / 'graph.html'}")
    print(f" Relatório: {out_dir / 'GRAPH_REPORT.md'}")
    print("=======================================================\n")

if __name__ == '__main__':
    run_traditional()
