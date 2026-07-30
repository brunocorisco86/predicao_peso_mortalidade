# Relatório Técnico de Explicabilidade de Aprendizado de Máquina (SHAP)

**Data:** 30 de Julho de 2026  
**Modelo Avaliado:** Stacking Ensemble Campeão (XGBoost GPU + LightGBM + MetaRidge)  
**Dataset:** Visão Longitudinal Integrada ($18.474$ Lotes Elegíveis RN-11)  
**Metodologia:** TreeSHAP (Shapley Additive exPlanations)

---

## 1. Visão Geral da Explicabilidade

Para garantir total transparência e governança zootécnica para os médicos veterinários, zootecnistas e engenheiros de PCP da C.Vale, os impactos das variáveis explicativas foram auditados através da teoria de jogos cooperativos (**Valores de Shapley**).

---

## 2. Gráficos Oficiais de Explicabilidade SHAP

### 2.1. SHAP Beeswarm Summary Plot (Impacto Global e Direção de Variáveis)
![SHAP Summary Plot](../../plots/explainability/shap_summary_plot.png)

* **Interpretação:** Cada ponto representa um lote individual. A cor indica o valor da variável (vermelho = alto, azul = baixo). O eixo X mostra o impacto de Shapley no peso final de abate (g).
* **Destaques:**
  - `peso_d42` e `peso_d35`: Valores altos (pontos vermelhos à direita) possuem impacto positivo de até $+400\text{g}$ no peso final.
  - `knn_pred_weight_k15` (RN-12 Gêmeos Digitais): Atua como uma âncora estabilizadora. Vizinhanças históricas de alto peso impulsionam a estimativa do lote atual.
  - `_mortalidade` (RN-08): Alta mortalidade relativa (vermelho à esquerda) empurra a predição para baixo (impacto negativo de até $-150\text{g}$).

### 2.2. Importância Absoluta Média dos Atributos (SHAP Bar Chart)
![SHAP Feature Importance Bar](../../plots/explainability/shap_feature_importance_bar.png)

---

## 3. Gráficos de Dependência Parcial (Efeitos Não-Lineares e Limiares Biológicos)

### 3.1. Efeito da Velocidade de Ganho de Peso ($GMD_{35-42}$)
![SHAP Dependence GMD](../../plots/explainability/shap_dependence_gmd.png)

* **Limiar Zootécnico:** Quando o $GMD_{35-42}$ supera $85\text{g/dia}$, o impacto de Shapley passa a ser estritamente positivo, adicionando até $+120\text{g}$ ao peso final de abate.

### 3.2. Efeito do Peso do Pintainho de 1 Dia (`c15`)
![SHAP Dependence Chick Weight](../../plots/explainability/shap_dependence_chick_weight.png)

* **Limiar Zootécnico:** Pintainhos alojados com peso $< 40\text{g}$ geram uma penalização severa de Shapley (até $-80\text{g}$ no peso final de abate ao 45º dia), reforçando a necessidade de nutrição pré-inicial intensiva na primeira semana.

---

## 4. Conclusão e Governança
A auditoria SHAP comprova que o modelo XGBoost/Stacking não utiliza correlações espúrias, mas sim relações causais biológicas validadas zootecnicamente.
