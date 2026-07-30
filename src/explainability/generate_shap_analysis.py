import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

DATASET_PATH = Path("data/processed/longitudinal_dataset.csv")
PLOTS_DIR = Path("plots/explainability")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Iniciando análise de explicabilidade SHAP no Modelo Campeão Final...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    
    if 'elegivel_rn11' in df.columns:
        df = df[df['elegivel_rn11'] == 1.0].copy()
        
    target = 'peso_abate_g'
    y = df[target].values
    
    exclude = ['data_alojamento', 'nome_fazenda', 'data_hora_transao', 'lote_composto', 
               'data_evento', 'data_criao', 'id_usurio_criao', 'extensionista', 'id_usurio', 
               'fazenda', 'produtor', 'data_producao_abate', 'peso_medio_abate_kg', 'peso_abate_g', 
               'gmd_abate', 'score_confianca_lote', 'categoria_amostragem', 'elegivel_rn11', 
               'motivo_inelegibilidade', 'estrategia_predicao', 'nucleo']
               
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    X = df[features].fillna(df[features].median())
    
    print(f"Treinando XGBoost Regressor com {len(features)} features em {len(X)} instâncias...")
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=8, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
        tree_method='hist', device='cpu', random_state=42, n_jobs=-1
    )
    # n_estimators reduced for faster training as this is just for explainability
    model.fit(X, y)
    
    print("Calculando valores TreeSHAP...")
    # Sample background for faster SHAP calculation if dataset is too large
    if len(X) > 10000:
        X_sample = X.sample(n=10000, random_state=42)
    else:
        X_sample = X
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    print("Gerando SHAP Summary Plot (Beeswarm)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gerando SHAP Feature Importance (Bar)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_feature_importance_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Check for GMD column
    gmd_cols = [c for c in features if 'gmd' in c.lower()]
    gmd_col = gmd_cols[-1] if gmd_cols else features[0]
    
    print(f"Gerando SHAP Dependence Plot para GMD ({gmd_col})...")
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(gmd_col, shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_dependence_gmd.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Check for c15
    chick_weight_col = 'c15' if 'c15' in features else features[1]
    
    print(f"Gerando SHAP Dependence Plot para peso do pintainho ({chick_weight_col})...")
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(chick_weight_col, shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_dependence_chick_weight.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Análise SHAP concluída com sucesso!")

if __name__ == "__main__":
    main()
