# Relatório de Explicabilidade SHAP - Modelo Campeão Final

## 1. Visão Geral
Este documento detalha os resultados da análise de explicabilidade baseada em valores SHAP (Shapley Additive exPlanations) aplicada ao Modelo Campeão Final (XGBoost + LightGBM + MetaRidge) no projeto de predição de peso das aves e mortalidade.

O método TreeSHAP foi empregado para decodificar as decisões não-lineares da modelagem baseada em árvores, garantindo total transparência do impacto de cada feature no modelo.

## 2. Gráficos de Explicabilidade

### 2.1. SHAP Summary Plot (Global Beeswarm)
O gráfico de resumo global ilustra a direção e a magnitude do impacto das variáveis no peso de abate.
![SHAP Summary Plot](../../plots/explainability/shap_summary_plot.png)
**Interpretação:**
- Cores quentes (vermelho) indicam valores altos da variável.
- Cores frias (azul) indicam valores baixos da variável.
- A posição no eixo X mostra o impacto no modelo. 

### 2.2. Importância SHAP (Bar Plot)
![SHAP Feature Importance](../../plots/explainability/shap_feature_importance_bar.png)
**Interpretação:**
As variáveis mais acima possuem a maior contribuição absoluta (média de |SHAP|) no momento de realizar a previsão do peso das aves, destacando os drivers mais importantes.

### 2.3. Dependência de GMD (Ganho Médio Diário)
![SHAP Dependence GMD](../../plots/explainability/shap_dependence_gmd.png)
**Interpretação:**
Observamos a relação isolada do GMD (ganho de peso contínuo) com a predição. É esperado que, quanto maior o GMD na fase de crescimento, maior seja o valor SHAP positivo direcionado ao peso de abate, confirmando o aspecto fisiológico do crescimento.

### 2.4. Dependência de Peso do Pintainho (c15)
![SHAP Dependence Chick Weight](../../plots/explainability/shap_dependence_chick_weight.png)
**Interpretação:**
Este gráfico aponta a influência do peso inicial do pintainho (c15) na decisão final. Pintainhos com maior peso ao nascer tendem a empurrar as predições de peso final para cima (valores SHAP positivos).

## 3. Guia Prático para Leigos: Como Cada Dado Afeta a Previsão de Peso

Para facilitar o entendimento por produtores, veterinários e técnicos de campo, a tabela abaixo resume **qual é a importância**, **o quanto interfere no resultado** e **como o modelo reage quando o dado é MAIOR ou MENOR**:

| Variável / Dado | O que Mede | Qual é a Importância para o Modelo? | O Quanto Interfere no Resultado? | O que acontece quando o dado é MAIOR? ⬆️ | O que acontece quando o dado é MENOR? ⬇️ |
|---|---|---|---|---|---|
| **`peso_d35` e `peso_d42`** | Pesagem amostral de campo aos 35 e 42 dias | **Bússola Principal**: É a medição direta do peso real da ave poucas semanas/dias antes do abate. | **Impacto Extremo** ($\pm 300\text{g}$ a $\pm 450\text{g}$) | **Empurra o peso final para cima**: Se a ave está pesada aos 35d/42d, o modelo projeta um abate bem pesado. | **Puxa o peso final para baixo**: Se a pesagem foi fraca, o modelo projeta um lote leve e atrasado no ganho. |
| **`gmd_35_42` e `gmd_28_35`** | Ganho Médio Diário de peso (g/dia) | **Pedal de Aceleração**: Mostra se o frango está arrancando no ganho de carne ou estagnado no final. | **Impacto Forte** ($\pm 100\text{g}$ a $\pm 200\text{g}$) | **Soma gramas extras**: Velocidade alta ($>85\text{g/dia}$) indica ótima conversão alimentar; o modelo eleva o peso final. | **Aplica desconto no peso**: Velocidade baixa ($<65\text{g/dia}$) indica estresse ou estagnação; o modelo reduz a estimativa. |
| **`knn_pred_weight_k15`** | Gêmeos Digitais da RN-12 ($K=15$ lotes parecidos) | **Âncor de Experiência**: O modelo consulta os 15 lotes históricos mais parecidos (mesma fazenda, estrutura e época). | **Impacto Moderado/Estabilizador** ($\pm 100\text{g}$ a $\pm 150\text{g}$) | **Dá confiança de alto peso**: Se os lotes "gêmeos" do passado fecharam pesados, o modelo eleva a previsão. | **Segura o otimismo**: Se o histórico de lotes gêmeos foi fraco, o modelo aplica uma trava de prudência. |
| **`c15`** | Peso do Pintainho de 1 dia ao alojamento (g) | **Largada da Corrida**: Qualidade e vigor do pintainho entregue pelo incubatório. | **Impacto Moderado** ($\pm 40\text{g}$ a $\pm 80\text{g}$) | **Ganho na largada**: Pintainho pesado ($\ge 45\text{g}$) desenvolve órgãos e musculatura mais rápido; o modelo soma até $+60\text{g}$. | **Atraso na 1ª semana**: Pintainho miúdo ($<40\text{g}$) sofre na arrancada; o modelo subtrai até $-80\text{g}$ no abate. |
| **`_mortalidade`** | Taxa de Mortalidade Acumulada (%) | **Termômetro de Saúde**: Reflete a presença de surtos sanitários ou estresse térmico grave. | **Impacto Negativo Direto** (até $-150\text{g}$) | **Desconto pesado**: Mortalidade alta ($>4\%$) sinaliza lote doente; o modelo entende que as aves vivas também cresceram menos. | **Sem penalização**: Mortalidade baixa ($<1,5\%$) mostra lote saudável; o modelo não aplica penalização. |
| **`oof_fazenda_target_enc`** | Histórico Geral da Fazenda | **Reputação Técnica**: Nível de tecnologia, isolamento e habilidade de manejo do produtor. | **Impacto Moderado** ($\pm 60\text{g}$) | **Confiança no produtor**: Fazendas com histórico pesado sustentam estimativas otimistas. | **Prudência no produtor**: Fazendas com histórico leve recebem um desconto preventivo. |

---

## 4. Conclusões
- A explicabilidade valida a sanidade do modelo: as relações não-lineares aprendidas pelo algoritmo estão em sintonia com a fisiologia e o manejo avícola esperados.
- Variáveis de histórico zootécnico e ambiência interagem de maneira forte, e o SHAP consegue capturar que limites superiores dessas variáveis trazem retornos não necessariamente lineares (possíveis efeitos platô de ganho de peso).
