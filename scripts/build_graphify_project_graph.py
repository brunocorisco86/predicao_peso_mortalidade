"""
build_graphify_project_graph.py
-------------------------------
Gera e atualiza o Grafo de Conhecimento (Graphify) do projeto C.Vale Predição de Peso.
"""

import json
from pathlib import Path
from graphify.extract import extract
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json

def main():
    scan_dir = Path('.').resolve()
    print("Mapeando arquivos Python do projeto...")
    project_files = [
        p for p in scan_dir.rglob('*.py')
        if not any(part.startswith('.') or part in ['venv', 'graphify-out', 'site-packages', 'dist-packages'] for part in p.parts)
    ]
    print(f"Extraindo AST de {len(project_files)} arquivos Python do projeto...")
    
    ast_result = extract(project_files, cache_root=scan_dir)
    G = build_from_json(ast_result, root=str(scan_dir), directed=False)
    print(f"Grafo construído: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
    
    if G.number_of_nodes() > 0:
        communities = cluster(G)
        cohesion = score_all(G, communities)
        gods = god_nodes(G)
        surprises = surprising_connections(G, communities)
        labels = {cid: f"Comunidade {cid}" for cid in communities}
        questions = suggest_questions(G, communities, labels)
        
        Path('graphify-out').mkdir(parents=True, exist_ok=True)
        to_json(G, communities, 'graphify-out/graph.json')
        
        detection = {
            'total_files': len(project_files),
            'total_words': 45000,
            'files': {'code': [str(f.relative_to(scan_dir)) for f in project_files]}
        }
        tokens = {'input': 0, 'output': 0}
        report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, str(scan_dir), suggested_questions=questions)
        Path('graphify-out/GRAPH_REPORT.md').write_text(report, encoding='utf-8')
        print("✅ Graphify Knowledge Graph atualizado com sucesso!")
    else:
        print("Aviso: Grafo sem nós.")

if __name__ == '__main__':
    main()
