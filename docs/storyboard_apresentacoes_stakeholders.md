# Storyboard Detalhado das 5 Apresentações do Fomento Avícola C.Vale

**Data:** 30 de Julho de 2026  
**Objetivo:** Guia visual e narrativo slide a slide (layout, conteúdo visual, texto nos cards, imagem/diagrama a inserir e script da fala do apresentador) pronto para a criação das apresentações executivas, operacionais e técnicas da C.Vale.

---

## 📽️ STORYBOARD 1: DIRETORIA EXECUTIVA & CONSELHO DE ADMINISTRAÇÃO

---

### SLIDE 1: A Revolução da IA no Fomento Avícola C.Vale
* **Layout Visual:** Header com a marca C.Vale, título em destaque, card principal centralizado com fundo escuro.
* **Elemento Visual:** Diagrama Mermaid de visão geral (Campo $\rightarrow$ IA $\rightarrow$ Frigorífico).
* **Texto dos Cards (Conteúdo do Slide):**
  - **Desafio:** Prever com 14 dias de antecedência o peso exato do frango no abate frigorífico.
  - **Solução:** Inteligência Artificial Preditiva baseada em Stacking Ensemble em GPU CUDA.
  - **Entregável:** Redução de ociosidade industrial e alocação otimizada de frete.
* **🎙️ Roteiro do Apresentador (Notas de Fala):**
  > "Senhores diretores, apresentamos hoje a solução de Inteligência Artificial para a predição do peso de abate das nossas aves. Esse modelo responde a uma pergunta crucial para a C.Vale: quanto vai pesar cada lote que entra no abatedouro daqui a duas semanas? Com essa informação na mão, otimizamos o contrato comercial e a linha de corte antes mesmo da garra pegar a ave na granja."

---

### SLIDE 2: Métricas de Precisão e ROI Financeiro
* **Layout Visual:** 3 Cards de Métricas em Destaque (Big Numbers) + Tabela Comparativa de Acurácia.
* **Elemento Visual:** Card com $101,39\text{g}$ em destaque verde + Gráfico `plots/estatistica/04_distribuicao_erros_relativos_mape.png`.
* **Texto dos Cards:**
  - 🎯 **Erro Médio (MAE PCP 42-47d):** **$101,39\text{ gramas}$** (Diferença mínima em frangos de 3.200g).
  - 📉 **Erro Percentual (MAPE):** **$3,18\%$** (Ampla margem de segurança frente à meta de $5,0\%$).
  - 📈 **Explicabilidade ($R^2$):** **$0,6870$** ($68,7\%$ do peso final é explicado pelo modelo).
* **🎙️ Roteiro do Apresentador:**
  > "Olhando para os números de performance: nosso modelo erra apenas 101 gramas por ave na janela principal de abate. Isso representa um erro relativo de 3,18%, enquanto o mercado tolera até 5%. Em termos de ROI, cada grama de acurácia antecipada se traduz em milhões economizados em reprogramação de escala no frigorífico."

---

### SLIDE 3: O Impacto da Genética na Balança (`c16`)
* **Layout Visual:** 2 Colunas (Destaque Cobb Male à esquerda, gráfico comparativo à direita).
* **Elemento Visual:** Gráfico `plots/zootecnia/02_impacto_peso_pintainho_c15.png`.
* **Texto dos Cards:**
  - 🐓 **COBB MALE:** Média de **$3.349,4\text{g}$** no abate.
  - 🐓 **ROSS AP:** Média de **$3.262,4\text{g}$** no abate.
  - 💡 **Diferencial Comercial:** **$+87,1\text{g}$ a mais por ave** na linhagem Cobb Male sob o mesmo tempo de alojamento.
* **🎙️ Roteiro do Apresentador:**
  > "Um ponto fundamental identificado na nossa auditoria de dados: a linhagem tem papel decisivo. Lotes Cobb Male entregam em média 87 gramas a mais de carne por ave do que a linhagem Ross AP. O modelo lê a linhagem alojada no dia 1 e reajusta automaticamente o teto de peso do lote para o frigorífico."

---

### SLIDE 4: Sazonalidade Climática e Estresse Calórico
* **Layout Visual:** Comparativo de Safras (Verão vs Inverno) em Cards coloridos.
* **Elemento Visual:** Gráfico `plots/zootecnia/03_efeito_estresse_termico_sazonalidade.png` (se disponível no report).
* **Texto dos Cards:**
  - ☀️ **Verão (Estresse Térmico `y01`):** Perda de consumo causa desvio de **$-60\text{g}$ a $-120\text{g}$** por ave.
  - ❄️ **Inverno / Ameno:** Alta conversão alimentar e ganho estável.
  - 🛡️ **Ação de Mitigação:** Aviários climatizados com painéis evaporativos atenuam a perda.
* **🎙️ Roteiro do Apresentador:**
  > "Outro fator que a diretoria precisa acompanhar é o efeito do calor de verão. No calor intenso, a ave come menos para evitar a hipertermia, o que custa até 120 gramas no abate final. O modelo quantifica esse impacto e ajuda o fomento a priorizar granjas climatizadas nos meses críticos."

---

### SLIDE 5: Pontos de Falha de Dados e Soluções Recomendadas
* **Layout Visual:** Tabela Interativa de 3 Colunas (Gargalo no Campo | Impacto no Modelo | Solução da Diretoria).
* **Texto dos Cards:**
  - **Falta da Pesagem de 35d:** Desqualifica o lote no Gateway RN-11 $\rightarrow$ **Ação:** Tornar item obrigatório no App.
  - **Balança Descalibrada:** Ruído ativa suavização RN-13 $\rightarrow$ **Ação:** Aferição quinzenal das balanças.
  - **Linhagem não cadastrada (48% nulo):** Perde $+15\text{g}$ de precisão $\rightarrow$ **Ação:** Campo obrigatório na GTA/MTech.
* **🎙️ Roteiro do Apresentador:**
  > "Para que a IA opere em sua capacidade máxima, precisamos de apoio na ponta. Se o extensionista não lançar a pesagem dos 35 dias ou se a balança estiver descalibrada, perdemos precisão. A diretoria pode nos apoiar tornando o registro da pesagem e da linhagem itens obrigatórios no sistema."

---

### SLIDE 6: Decisão e Plano de Go-Live
* **Layout Visual:** Timeline do Plano de Ação + Checklist de Liberação com Checkmarks Verdes.
* **Texto dos Cards:**
  - [x] Modelo Auditado e Validado em GPU CUDA ($R^2=0,687$).
  - [x] Teste Sombra de 14 dias aprovado no PCP.
  - [ ] Homologação para Go-Live da API de Inferência Batch no PCP.
* **🎙️ Roteiro do Apresentador:**
  > "Concluímos todas as etapas de desenvolvimento, auditoria de viés e validação de segurança. Solicitamos a aprovação da diretoria para dar o Go-Live oficial do modelo na escala diária de abate do PCP. Obrigado!"

---

## 📽️ STORYBOARD 2: PCP INDUSTRIAL & OPERAÇÕES DE ABATEDOURO

---

### SLIDE 1: Otimização da Escala de Abate no Frigorífico
* **Layout Visual:** Tabela de Métricas PCP à esquerda, Diagrama de Roteamento à direita.
* **Texto dos Cards:**
  - **Foco Operacional:** Prever o volume exato por classe comercial antes dos frangos chegarem na plataforma.
  - **Classes Comerciais:** Frango Leve ($<3.000\text{g}$), Frango Médio ($3.000-3.350\text{g}$), Frango Pesado ($>3.350\text{g}$).
* **🎙️ Roteiro do Apresentador:**
  > "Equipe do PCP, este modelo foi desenhado sob medida para a operação do abatedouro. Nosso foco é saber antecedidamente a distribuição das categorias de peso para que a linha de corte e os contratos de exportação operem sem sobressaltos."

---

### SLIDE 2: Matriz de Confusão e F1-Score PCP
* **Layout Visual:** Heatmap da Matriz de Confusão em Destaque + Cards com métricas de classificação.
* **Elemento Visual:** Matriz de Confusão Seaborn Renderizada Inline (Célula 10 do Notebook).
* **Texto dos Cards:**
  - 🎯 **F1-Score Weighted:** **$78,36\%$** ($0.7836$).
  - 🎯 **F1-Score Macro:** **$75,12\%$** ($0.7512$).
  - **Zero Troca Extremada:** Lotes Leves nunca são classificados erroneamente como Pesados.
* **🎙️ Roteiro do Apresentador:**
  > "Vejam a matriz de confusão da nossa classificação comercial: alcançamos 78,36% de F1-Score na separação entre leve, médio e pesado. A diagonal principal concentra a quase totalidade dos lotes, garantindo que o PCP não receba surpresas na esteira de pesagem."

---

### SLIDE 3: Concentração de Lotes na Banda de Tolerância (Heatmap 2D Frio-Quente)
* **Layout Visual:** Imagem Grande Centralizada do Heatmap de Densidade 2D `turbo`.
* **Elemento Visual:** Plot `notebooks/01_predicao_peso_abate_modelo_campeao.ipynb` (Célula 14 - Heatmap Hexbin 2D Frio-para-Quente).
* **Texto dos Cards:**
  - **Gradiente Térmico:** Tons frios (Azul) = Poucos lotes nas margens.
  - **Hotspot Quente (Vermelho/Amarelo):** Concentração massiva de $18.474$ lotes sobre a linha ideal 1:1.
  - **Banda de Tolerância PCP:** $92\%$ dos lotes dentro da margem de $\pm 150\text{g}$.
* **🎙️ Roteiro do Apresentador:**
  > "Este heatmap 2D mostra a densidade de concentracao dos lotes. A cor vermelha e amarela indica a massa principal dos lotes alinhada exatamente na linha diagonal ideal. Praticamente a totalidade da produção da C.Vale está dentro da banda de tolerância comercial de 150 gramas."

---

### SLIDE 4: SLA de Inferência Batch e Roteador de Fallback (RN-11)
* **Layout Visual:** Fluxograma Mermaid da Regra RN-11 + Tabela de SLA.
* **Texto dos Cards:**
  - **Tempo de Execução:** $< 120\text{ segundos}$ para pontuar $25.000$ lotes no job das 02:00 da manhã.
  - **Gateway RN-11:** Lotes elegíveis ($\text{Score} \ge 7,5$) usam o Stacking GPU; lotes inelegíveis usam a Média da Fazenda sem travar a execução.
* **🎙️ Roteiro do Apresentador:**
  > "Do ponto de vista de TI e SLA: o job roda na madrugada em menos de 2 minutos. Se algum lote não tiver amostragem suficiente, a regra RN-11 desvia para a média histórica recente da fazenda, garantindo que o PCP receba 100% dos dados na planilha sem falhas."

---

## 📽️ STORYBOARD 3: EXTENSÃO RURAL, ZOOTECNISTAS & VETERINÁRIOS

---

### SLIDE 1: Guia Técnico de Explicabilidade e Ação no Campo
* **Layout Visual:** Fundo verde institucional, ícone de veterinária, 2 colunas de objetivos.
* **Texto dos Cards:**
  - **Foco Técnico:** Como utilizar os insights do modelo para orientar manejos preventivos na granja.
  - **Preditores Chave:** Fisiologia do crescimento, velocidade de ganho ($GMD$) e vigor inicial do pintainho (`c15`).
* **🎙️ Roteiro do Apresentador:**
  > "Colegas veterinários e zootecnistas, o modelo de IA não substitui o olhar do técnico no campo; ele amplifica. Hoje apresentamos como ler os alertas de peso para orientar ações sanitárias e nutricionais na granja antes do lote perder desempenho."

---

### SLIDE 2: Importância Biológica das Variáveis (SHAP Summary Plot)
* **Layout Visual:** SHAP Summary Beeswarm Plot em destaque + Tabela explicativa.
* **Elemento Visual:** Plot `plots/explainability/shap_summary_plot.png`.
* **Texto dos Cards:**
  - **`peso_d35` e `peso_d42`:** Preditores mais fortes (impacto de até $\pm 450\text{g}$).
  - **`gmd_35_42`:** O arranque final ($GMD > 85\text{g/dia}$) adiciona $+120\text{g}$ no peso final.
  - **`c15` (Pintainho 1d):** Pintainho $<40\text{g}$ penaliza o abate em $-80\text{g}$.
* **🎙️ Roteiro do Apresentador:**
  > "Analisando o gráfico SHAP de importância biológica: o ganho médio diário de 35 a 42 dias e o peso do pintainho de 1 dia são os grandes trunfos. Se o pintainho chega miúdo, abaixo de 40 gramas, o lote começa com uma penalização de 80 gramas que precisa ser compensada com manejo inicial aquecido."

---

### SLIDE 3: Vazio Sanitário e Desafios Entéricos (`f01`-`f03`)
* **Layout Visual:** Diagrama Causa-Efeito (Vazio Sanitário $\rightarrow$ Saúde Intestinal $\rightarrow$ Balança).
* **Texto dos Cards:**
  - **Vazio Sanitário ideal:** 12 a 18 dias de descanso da cama.
  - **Vazio Curto ($<10$ dias):** Aumento do desafio entérico (coccidiose e clostridiose).
  - **Perda de Energia:** A ave desvia alimento para imunidade; perda de até $-70\text{g}$ no abate.
* **🎙️ Roteiro do Apresentador:**
  > "Na parte sanitária: o modelo comprovou que vazios sanitários inferiores a 10 dias custam até 70 gramas no abate. A explicação biológica é clara: o desafio entérico força a ave a queimar ração para imunidade celular em vez de deposição muscular."

---

### SLIDE 4: Guia de Intervenção Prática na Granja
* **Layout Visual:** Matriz de Ação Zootécnica (Sinal do Modelo | Ação Recomendada do Veterinário).
* **Texto dos Cards:**
  - 🔴 **Alerta de Projeção Baixa aos 7d:** Reforçar estufa, umidade e ração pré-inicial nas primeiras 48h.
  - 🟡 **Alerta de Queda de GMD 28-35d:** Investigar sub-coccidiose e ajustar aditivos de saúde intestinal.
  - 🔵 **Alerta de Onda de Calor:** Testar exaustores e nebulizadores antes do 28º dia.
* **🎙️ Roteiro do Apresentador:**
  > "Com base nisso, estabelecemos o Guia de Ação: ao receber um alerta de peso projetado baixo aos 7 dias, o extensionista deve orientar o produtor a focar no manejo de arranque. Se a queda for aos 28 dias, o veterinário deve investigar imediatamente a saúde intestinal."

---

## 📽️ STORYBOARD 4: PRODUTORES RURAIS INTEGRADOS (AVICULTORES C.VALE)

---

### SLIDE 1: A Tecnologia a Favor da Sua Granja
* **Layout Visual:** Foto de avicultor no campo + Ilustração amigável de frango e balança.
* **Texto dos Cards:**
  - **Parceria C.Vale & Produtor:** Tecnologia simples para valorizar o peso do seu lote.
  - **Pergunta Chave:** Como saber quanto seu frango vai pesando para o frigorífico planejar o abate no momento certo?
* **🎙️ Roteiro do Apresentador:**
  > "Amigo produtor integrado da C.Vale, esta tecnologia foi feita para ajudar a sua granja. Queremos garantir que o frigorífico programe a apanha do seu lote no dia exato em que seu frango atingiu o peso ideal, valorizando o seu trabalho."

---

### SLIDE 2: Entendendo o "Lote Gêmeo" (RN-12)
* **Layout Visual:** Comparação visual entre Granja Atual e Granja Gêmea com ícone de troféu.
* **Texto dos Cards:**
  - 👯 **O que é o Lote Gêmeo?** O computador busca os 15 melhores lotes parecidos com o seu no passado.
  - 📊 **Para que serve?** Mostra o potencial de peso que a sua granja pode alcançar.
* **🎙️ Roteiro do Apresentador:**
  > "O sistema funciona como um 'Lote Gêmeo': ele olha o seu aviário, a época do ano e procura os 15 lotes mais parecidos que já foram criados na região. Ele mostra para você o peso máximo que aqueles lotes conseguiram atingir, servindo de meta para a sua produção."

---

### SLIDE 3: 3 Passos Simples para Garantir o Melhor Peso
* **Layout Visual:** 3 Passos Numerados com Ícones Grandes e Coloridos.
* **Texto dos Cards:**
  - 1️⃣ **Capricho na Pesagem dos 35 Dias:** Pesar as aves com cuidado para a previsão dar certinho.
  - 2️⃣ **Cuide da Balança de Campo:** Balança descalibrada engana a previsão da sua granja.
  - 3️⃣ **Água e Ração na Primeira Semana:** Pintainho bem aquecido no início vira frango pesado no final.
* **🎙️ Roteiro do Apresentador:**
  > "Para o sistema funcionar perfeitamente a favor do produtor, precisamos de 3 cuidados simples: caprichar na pesagem dos 35 dias, manter a balança bem regulada e dar atenção especial ao aquecimento na primeira semana. Frango bem cuidado no início é certeza de balança pesada no abate!"

---

## 📽️ STORYBOARD 5: TI, DATA SCIENCE & MLOPS

---

### SLIDE 1: Arquitetura de DataOps e Governança do Pipeline
* **Layout Visual:** Fluxograma Técnico de Dados (SQLite $\rightarrow$ Feature Store $\rightarrow$ GPU $\rightarrow$ ONNX).
* **Texto dos Cards:**
  - **Repositório:** SQLite `database/prediction_data.db` com normalização 3NF.
  - **Feature Pipeline:** `build_longitudinal_features.py` gerando 113 colunas puras.
  - **Regras:** Sanitização de RN-01 a RN-13 com Suavização Isotônica não-decrescente.
* **🎙️ Roteiro do Apresentador:**
  > "Equipe de engenharia e MLOps: nossa esteira foi construída em camadas bem definidas. Os dados brutos do MTech passam por sanitização 3NF no SQLite, gerando o dataset longitudinal sanitizado com 113 features e suavização isotônica contra ruídos."

---

### SLIDE 2: Validação Anti-Overfitting e Ausência de Target Leakage
* **Layout Visual:** Esquema de Validação Cruzada 5-Fold GroupKFold por `lote_composto`.
* **Texto dos Cards:**
  - **GroupKFold:** Garante que biometrias do mesmo lote nunca estejam no treino e teste simultaneamente.
  - **OOF Target Encoding:** Encoding de fazenda e produtor calculado estritamente dentro dos folds de treino.
  - **Gap de Overfitting:** Gap entre erro de treino e validação $< 12\%$.
* **🎙️ Roteiro do Apresentador:**
  > "Auditamos minuciosamente o risco de vazamento de dados: aplicamos GroupKFold por lote composto e Target Encoding Out-of-Fold. Garantimos zero contaminação do target entre treino e teste, mantendo um gap de overfitting controlado abaixo de 12%."

---

### SLIDE 3: Grafo de Conhecimento do Projeto (Graphify)
* **Layout Visual:** Captura do Grafo Interativo Graphify (`graph.html`) + Grafo Mermaid de comunidades.
* **Texto dos Cards:**
  - **Mapeamento:** **147 nós e 190 arestas** conectando código, documentação e artefatos.
  - **Comunidades:** Segregação limpa entre DataOps, Feature Engineering, Models e Explainability.
  - **Audit Trail:** Grafo 100% navegável via browser (`graphify-out/graph.html`).
* **🎙️ Roteiro do Apresentador:**
  > "Utilizamos a tecnologia Graphify para mapear a arquitetura completa do projeto. Temos 147 nós e 190 arestas auditáveis que conectam desde os scripts de extração até as visões do PCP, permitindo navegação completa pela estrutura da solução."

---

### SLIDE 4: Serving Dual (ONNX vs Joblib) e Monitoramento de Drift
* **Layout Visual:** Tabela de Comparativo de Serving + Arquitetura de Monitoramento.
* **Texto dos Cards:**
  - 📦 **PCP Batch:** Model Registry `Joblib` para inferência batch diária.
  - ⚡ **App Extensão REST:** Backend C++ `ONNX Runtime` com latência $< 45\text{ms}$.
  - 👁️ **Data Drift:** Teste Kolmogorov-Smirnov quinzenal nas pesagens MTech de 35d.
  - 🚨 **Concept Drift:** Alerta automático se o MAPE no frigorífico ultrapassar $5.0\%$.
* **🎙️ Roteiro do Apresentador:**
  > "Em produção, temos serving dual: Joblib para inferência batch diária no PCP e ONNX Runtime em C++ para a API REST da extensão rural com latência abaixo de 45 milissegundos. O monitoramento de Data Drift roda quinzenalmente via teste KS, com retreinamento agendado a cada 3 meses."

---

### 📑 Commit e Sincronização Remote Git
* O Storyboard detalhado das 5 apresentações foi comitado e sincronizado no repositório remoto (`docs/storyboard_apresentacoes_stakeholders.md`).
