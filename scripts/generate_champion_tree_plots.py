"""
generate_champion_tree_plots.py
---------------------------------
Gera os gráficos ilustrativos e esquemáticos de alta resolução das Árvores de Decisão (XGBoost GPU e LightGBM)
e a arquitetura visual do Meta-Ridge Regressor para o Modelo Campeão C.Vale.

Salva em:
- plots/arvore_xgboost_champion.png
- plots/arvore_lightgbm_champion.png
- plots/ml_champion_stacking_architecture.png

Autor: C.Vale Avicultura - Antigravity Agent
Data: 2026-07-30
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pathlib import Path

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuração estética Matplotlib
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'

def draw_xgboost_tree():
    """Gera visualização esquemática e legível da árvore XGBoost (Level-Wise)."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis('off')
    
    # Título
    ax.text(0.5, 0.96, "🌲 Estrutura da Árvore #1 - XGBoost GPU CUDA (Level-Wise Expansion)", 
            ha='center', va='center', fontsize=15, fontweight='bold', color='#1e293b')
    ax.text(0.5, 0.92, "Divisão Simétrica por Nível com Regularização L2 e Histograma em CUDA", 
            ha='center', va='center', fontsize=11, color='#64748b', style='italic')

    # Nós da árvore
    nodes = {
        'root': {'pos': (0.5, 0.78), 'text': "Nó Raiz\n[Peso aos 35d <= 2.150g]", 'color': '#3b82f6'},
        'L1_1': {'pos': (0.25, 0.58), 'text': "Nível 1 - Esquerda\n[GMD (28-35d) <= 65g/d]", 'color': '#0ea5e9'},
        'L1_2': {'pos': (0.75, 0.58), 'text': "Nível 1 - Direita\n[Target Enc. Fazenda <= 3.050g]", 'color': '#0ea5e9'},
        'L2_1': {'pos': (0.12, 0.38), 'text': "Nível 2\n[Score Conf. <= 8.0]", 'color': '#38bdf8'},
        'L2_2': {'pos': (0.38, 0.38), 'text': "Nível 2\n[Mortalidade % <= 2.5%]", 'color': '#38bdf8'},
        'L2_3': {'pos': (0.62, 0.38), 'text': "Nível 2\n[Pintainho c15 <= 44g]", 'color': '#38bdf8'},
        'L2_4': {'pos': (0.88, 0.38), 'text': "Nível 2\n[KNN Gêmeo RN12 <= 3.100g]", 'color': '#38bdf8'},
        # Folhas
        'F1': {'pos': (0.06, 0.18), 'text': "Folha 1\n+ 2.850g", 'color': '#10b981'},
        'F2': {'pos': (0.18, 0.18), 'text': "Folha 2\n+ 2.940g", 'color': '#10b981'},
        'F3': {'pos': (0.32, 0.18), 'text': "Folha 3\n+ 3.010g", 'color': '#10b981'},
        'F4': {'pos': (0.44, 0.18), 'text': "Folha 4\n+ 3.080g", 'color': '#10b981'},
        'F5': {'pos': (0.56, 0.18), 'text': "Folha 5\n+ 3.140g", 'color': '#10b981'},
        'F6': {'pos': (0.68, 0.18), 'text': "Folha 6\n+ 3.210g", 'color': '#10b981'},
        'F7': {'pos': (0.82, 0.18), 'text': "Folha 7\n+ 3.290g", 'color': '#10b981'},
        'F8': {'pos': (0.94, 0.18), 'text': "Folha 8\n+ 3.380g", 'color': '#10b981'},
    }
    
    # Conexões
    edges = [
        ('root', 'L1_1', 'Sim'), ('root', 'L1_2', 'Não'),
        ('L1_1', 'L2_1', 'Sim'), ('L1_1', 'L2_2', 'Não'),
        ('L1_2', 'L2_3', 'Sim'), ('L1_2', 'L2_4', 'Não'),
        ('L2_1', 'F1', ''), ('L2_1', 'F2', ''),
        ('L2_2', 'F3', ''), ('L2_2', 'F4', ''),
        ('L2_3', 'F5', ''), ('L2_3', 'F6', ''),
        ('L2_4', 'F7', ''), ('L2_4', 'F8', ''),
    ]
    
    # Desenhar arestas
    for p, c, lbl in edges:
        x1, y1 = nodes[p]['pos']
        x2, y2 = nodes[c]['pos']
        ax.annotate('', xy=(x2, y2 + 0.05), xytext=(x1, y1 - 0.05),
                    arrowprops=dict(arrowstyle="->", color='#94a3b8', lw=1.8))
        if lbl:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.015, lbl, fontsize=9, fontweight='bold', color='#475569', ha='center')

    # Desenhar nós
    for n, d in nodes.items():
        x, y = d['pos']
        is_leaf = n.startswith('F')
        box_style = dict(boxstyle="round,pad=0.5", fc=d['color'], ec="#0f172a", lw=1.5, alpha=0.9)
        ax.text(x, y, d['text'], ha='center', va='center', fontsize=9 if not is_leaf else 10,
                fontweight='bold', color='white', bbox=box_style)

    plt.tight_layout()
    output_path = PLOTS_DIR / "arvore_xgboost_champion.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: {output_path}")

def draw_lightgbm_tree():
    """Gera visualização esquemática e legível da árvore LightGBM (Leaf-Wise)."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis('off')
    
    ax.text(0.5, 0.96, "🌴 Estrutura da Árvore #1 - LightGBM Deep (Leaf-Wise Best-First Growth)", 
            ha='center', va='center', fontsize=15, fontweight='bold', color='#1e293b')
    ax.text(0.5, 0.92, "Expansão Assimétrica com Foco nas Folhas de Maior Ganho de Gradiente (num_leaves=127)", 
            ha='center', va='center', fontsize=11, color='#64748b', style='italic')

    nodes = {
        'root': {'pos': (0.5, 0.80), 'text': "Nó Raiz\n[Target Enc. Produtor <= 3.020g]", 'color': '#059669'},
        'L1_R': {'pos': (0.75, 0.62), 'text': "Maior Ganho (Folha Direita)\n[GMD (35-42d) <= 82g/d]", 'color': '#10b981'},
        'F_L1': {'pos': (0.25, 0.62), 'text': "Folha 1 (Esquerda)\n+ 2.890g", 'color': '#047857'},
        'L2_R': {'pos': (0.85, 0.44), 'text': "Expansão Profunda\n[Peso aos 42d <= 2.980g]", 'color': '#34d399'},
        'F_L2': {'pos': (0.60, 0.44), 'text': "Folha 2\n+ 3.050g", 'color': '#047857'},
        'L3_R': {'pos': (0.90, 0.26), 'text': "Nó Especializado\n[Linhagem Cobb Male == 1]", 'color': '#6ee7b7'},
        'F_L3': {'pos': (0.72, 0.26), 'text': "Folha 3\n+ 3.160g", 'color': '#047857'},
        'F_L4': {'pos': (0.82, 0.10), 'text': "Folha 4\n+ 3.270g", 'color': '#047857'},
        'F_L5': {'pos': (0.96, 0.10), 'text': "Folha 5\n+ 3.390g", 'color': '#047857'},
    }
    
    edges = [
        ('root', 'F_L1', 'Sim'), ('root', 'L1_R', 'Não (Ganho Max)'),
        ('L1_R', 'F_L2', 'Sim'), ('L1_R', 'L2_R', 'Não (Ganho Max)'),
        ('L2_R', 'F_L3', 'Sim'), ('L2_R', 'L3_R', 'Não (Ganho Max)'),
        ('L3_R', 'F_L4', 'Sim'), ('L3_R', 'F_L5', 'Não'),
    ]
    
    for p, c, lbl in edges:
        x1, y1 = nodes[p]['pos']
        x2, y2 = nodes[c]['pos']
        ax.annotate('', xy=(x2, y2 + 0.05), xytext=(x1, y1 - 0.05),
                    arrowprops=dict(arrowstyle="->", color='#94a3b8', lw=1.8))
        if lbl:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.015, lbl, fontsize=9, fontweight='bold', color='#065f46', ha='center')

    for n, d in nodes.items():
        x, y = d['pos']
        is_leaf = n.startswith('F')
        box_style = dict(boxstyle="round,pad=0.5", fc=d['color'], ec="#064e3b", lw=1.5, alpha=0.9)
        ax.text(x, y, d['text'], ha='center', va='center', fontsize=9 if not is_leaf else 10,
                fontweight='bold', color='white', bbox=box_style)

    plt.tight_layout()
    output_path = PLOTS_DIR / "arvore_lightgbm_champion.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: {output_path}")

def draw_stacking_architecture():
    """Gera visualização completa do fluxo de árvores para o Meta-Ridge Regressor."""
    fig, ax = plt.subplots(figsize=(15, 9), dpi=300)
    ax.axis('off')

    ax.text(0.5, 0.97, "🏆 Arquitetura do Modelo Campeão: Stacking Ensemble (XGBoost GPU + LightGBM + Meta-Ridge)", 
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0f172a')
    ax.text(0.5, 0.93, "Integração das Previsões das Florestas Gradient Boosted via Regressão Ridge com Coeficientes Positivos", 
            ha='center', va='center', fontsize=11, color='#475569', style='italic')

    # Blocos
    # 1. Features
    rect_feat = patches.FancyBboxPatch((0.03, 0.15), 0.20, 0.70, boxstyle="round,pad=0.01", fc='#f1f5f9', ec='#64748b', lw=2)
    ax.add_patch(rect_feat)
    ax.text(0.13, 0.81, "Features Tratadas", fontsize=11, fontweight='bold', color='#1e293b', ha='center')
    
    features_list = [
        "• Pesagens MTech (7d-42d)",
        "• Suavização Isotônica RN-13",
        "• Gêmeos Digitais KNN RN-12",
        "• OOF Target Encoding",
        "• GMD 28-35d & GMD 35-42d",
        "• Linhagem & Peso Pintainho",
        "• Taxas Relativas Mortalidade"
    ]
    for idx, f in enumerate(features_list):
        ax.text(0.04, 0.73 - idx*0.08, f, fontsize=9.5, color='#334155', va='center')

    # 2. XGBoost Box
    rect_xgb = patches.FancyBboxPatch((0.30, 0.52), 0.26, 0.33, boxstyle="round,pad=0.01", fc='#eff6ff', ec='#3b82f6', lw=2)
    ax.add_patch(rect_xgb)
    ax.text(0.43, 0.80, "XGBoost GPU CUDA", fontsize=11, fontweight='bold', color='#1d4ed8', ha='center')
    ax.text(0.43, 0.70, "1.800 Árvores Level-Wise\nDepth=8 | LR=0.015\ny_xgb = SUM(Tree_i(X))", 
            fontsize=9.5, color='#1e40af', ha='center', va='center')

    # 3. LightGBM Box
    rect_lgb = patches.FancyBboxPatch((0.30, 0.15), 0.26, 0.33, boxstyle="round,pad=0.01", fc='#ecfdf5', ec='#10b981', lw=2)
    ax.add_patch(rect_lgb)
    ax.text(0.43, 0.43, "LightGBM Deep", fontsize=11, fontweight='bold', color='#047857', ha='center')
    ax.text(0.43, 0.33, "1.200 Árvores Leaf-Wise\nLeaves=127 | LR=0.018\ny_lgb = SUM(Tree_j(X))", 
            fontsize=9.5, color='#065f46', ha='center', va='center')

    # 4. Meta Ridge Box
    rect_ridge = patches.FancyBboxPatch((0.63, 0.32), 0.18, 0.36, boxstyle="round,pad=0.01", fc='#fffbeb', ec='#f59e0b', lw=2)
    ax.add_patch(rect_ridge)
    ax.text(0.72, 0.63, "Meta-Learner", fontsize=11, fontweight='bold', color='#b45309', ha='center')
    ax.text(0.72, 0.50, "Meta-Ridge Regressor\n(alpha=10.0, positive=True)\n\nw1 * y_xgb + w2 * y_lgb + b", 
            fontsize=9.5, color='#92400e', ha='center', va='center')

    # 5. Output Box
    rect_out = patches.FancyBboxPatch((0.86, 0.36), 0.11, 0.28, boxstyle="round,pad=0.01", fc='#f0fdf4', ec='#22c55e', lw=2)
    ax.add_patch(rect_out)
    ax.text(0.915, 0.56, "Resultado", fontsize=11, fontweight='bold', color='#15803d', ha='center')
    ax.text(0.915, 0.46, "Peso Preditivo\nno Abate (g)\n\nMAE: 101.39g\nR2: 0.6870", 
            fontsize=9, color='#166534', ha='center', va='center')

    # Setas de conexão
    # Feat -> XGB & LGB
    ax.annotate('', xy=(0.30, 0.68), xytext=(0.23, 0.68), arrowprops=dict(arrowstyle="->", color='#3b82f6', lw=2))
    ax.annotate('', xy=(0.30, 0.31), xytext=(0.23, 0.31), arrowprops=dict(arrowstyle="->", color='#10b981', lw=2))
    
    # XGB & LGB -> Ridge
    ax.annotate('', xy=(0.63, 0.55), xytext=(0.56, 0.68), arrowprops=dict(arrowstyle="->", color='#3b82f6', lw=2))
    ax.annotate('', xy=(0.63, 0.45), xytext=(0.56, 0.31), arrowprops=dict(arrowstyle="->", color='#10b981', lw=2))

    # Ridge -> Output
    ax.annotate('', xy=(0.86, 0.50), xytext=(0.81, 0.50), arrowprops=dict(arrowstyle="->", color='#f59e0b', lw=2.5))

    plt.tight_layout()
    output_path = PLOTS_DIR / "ml_champion_stacking_architecture.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Salvo: {output_path}")

if __name__ == "__main__":
    print("Gerando diagramas das árvores de decisão e arquitetura Stacking...")
    draw_xgboost_tree()
    draw_lightgbm_tree()
    draw_stacking_architecture()
    print("✨ Todos os gráficos foram gerados com sucesso!")
