import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
import warnings
warnings.filterwarnings('ignore')

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_experiment():
    print("Loading data...")
    df = pd.read_csv('data/processed/unified_data.csv', low_memory=False)
    
    # Filter by elegivel_rn11
    if 'elegivel_rn11' in df.columns:
        df = df[df['elegivel_rn11'] == 1.0].copy()
    
    # Drop rows without target or essential features
    df = df.dropna(subset=['peso_abate_g', 'peso', 'idade', 'idade_abate'])
    
    target = 'peso_abate_g'
    group_col = 'lote_composto'
    
    exclude = ['data_alojamento', 'nome_fazenda', 'data_hora_transao', 'lote_composto', 
               'data_evento', 'data_criao', 'id_usurio_criao', 'extensionista', 'id_usurio', 
               'fazenda', 'produtor', 'data_producao_abate', 'peso_medio_abate_kg', 'peso_abate_g', 
               'gmd_abate', 'score_confianca_lote', 'categoria_amostragem', 'elegivel_rn11', 
               'motivo_inelegibilidade', 'estrategia_predicao', 'nucleo']
    
    features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64]]
    
    print(f"Number of features: {len(features)}")
    
    X = df[features]
    y = df[target]
    groups = df[group_col]
    
    models = {
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
    
    results = []
    gkf = GroupKFold(n_splits=5)
    feature_importances = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        maes, rmses, mapes, r2s = [], [], [], []
        importances = np.zeros(len(features))
        
        for train_idx, val_idx in gkf.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            maes.append(mean_absolute_error(y_val, preds))
            rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
            mapes.append(mean_absolute_percentage_error(y_val, preds))
            r2s.append(r2_score(y_val, preds))
            
            if name == 'CatBoost':
                importances += model.feature_importances_ / 5
            elif hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_ / 5
                
        results.append({
            'Model': name,
            'MAE (g)': np.mean(maes),
            'RMSE (g)': np.mean(rmses),
            'MAPE (%)': np.mean(mapes),
            'R²': np.mean(r2s)
        })
        
        if name in ['XGBoost', 'LightGBM', 'CatBoost']:
            feature_importances[name] = importances
            
    print("Training Híbrido (Base + LGBM)...")
    maes, rmses, mapes, r2s = [], [], [], []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if 'knn_pred_weight_k15' in X_train.columns:
            base_train = X_train['knn_pred_weight_k15']
            base_val = X_val['knn_pred_weight_k15']
        else:
            base_train = X_train['peso'] * (X_train['idade_abate'] / X_train['idade'])
            base_val = X_val['peso'] * (X_val['idade_abate'] / X_val['idade'])
            
        base_train = base_train.fillna(y_train.mean())
        base_val = base_val.fillna(y_train.mean())
            
        residuals_train = y_train - base_train
        
        meta_model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        meta_model.fit(X_train, residuals_train)
        
        res_preds = meta_model.predict(X_val)
        final_preds = base_val + res_preds
        
        maes.append(mean_absolute_error(y_val, final_preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, final_preds)))
        mapes.append(mean_absolute_percentage_error(y_val, final_preds))
        r2s.append(r2_score(y_val, final_preds))
        
    results.append({
        'Model': 'Híbrido (Base + LGBM)',
        'MAE (g)': np.mean(maes),
        'RMSE (g)': np.mean(rmses),
        'MAPE (%)': np.mean(mapes),
        'R²': np.mean(r2s)
    })
    
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    
    # Plot models comparison
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=res_df, x='Model', y='MAE (g)', palette='viridis')
    plt.title('Comparação de Modelos - MAE (g)')
    plt.xticks(rotation=45)
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('plots/ml_models_comparison.png')
    
    # Plot Feature Importance (LightGBM)
    if 'LightGBM' in feature_importances:
        fi_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances['LightGBM']
        }).sort_values(by='Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=fi_df, x='Importance', y='Feature', palette='magma')
        plt.title('Top 15 Atributos mais Importantes (LightGBM)')
        plt.tight_layout()
        plt.savefig('plots/ml_feature_importance.png')
        
    # Write to MD
    with open('docs/delineamento_modelos_ml.md', 'w') as f:
        f.write("# Delineamento de Modelos de Machine Learning\n\n")
        f.write("## 1. Algoritmos Comparados\n")
        f.write("- XGBoost Regressor\n- LightGBM Regressor\n- CatBoost Regressor\n- HistGradientBoosting Regressor\n- Modelo Híbrido Base (Gompertz + Gradient Boosting Ensemble)\n\n")
        f.write("## 2. Regras e Dataset\n")
        f.write("- Base: RN-01 a RN-12 (elegivel_rn11=1 e RN-12 Gêmeos Digitais KNN)\n")
        f.write("- Validação Cruzada: 5-Fold GroupKFold por `lote_composto`\n")
        f.write("- Métricas: MAE (g), RMSE (g), MAPE (%), R²\n\n")
        f.write("## 3. Resultados\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## 4. Conclusão\n")
        f.write("A comparação demonstra qual modelo apresenta melhor balanço entre acurácia (menor MAE e MAPE) e generalização, auxiliando na escolha do estimador final de peso de abate.\n")

if __name__ == '__main__':
    run_experiment()
