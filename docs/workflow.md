# Roteiro do Projeto: Predição de Peso e Mortalidade

Este documento descreve o fluxo de trabalho proposto para o desenvolvimento do projeto de predição de peso e mortalidade de aves.

## Fases do Projeto

1.  **Análise Exploratória de Dados (EDA) dos Dados Brutos:**
    *   Utilização de notebooks (na pasta `notebooks/`) para explorar e entender a estrutura, qualidade e características dos dados brutos (`data/raw/extracao_mtech/` e `data/raw/features/`).
    *   Identificação de padrões, anomalias e relações entre as variáveis.

2.  **Filtragem de Dados Relevantes para o Modelo:**
    *   A etapa inicial de extração e racionalização dos dados brutos é realizada pelo script `src/etl/extract_mtech_data.py`, que consolida as informações em um banco de dados SQLite.
    *   Seleção e pré-processamento dos dados brutos para isolar as informações mais pertinentes aos objetivos de predição.
    *   Limpeza de dados, tratamento de valores ausentes e remoção de outliers.

3.  **Incorporação de Features:**
    *   Combinação dos dados da tabela fato (`extracao_mtech/`) com as features adicionais (`features/`) para criar um dataset unificado para modelagem.

4.  **Engenharia de Features:**
    *   Criação de novas variáveis a partir das existentes que possam melhorar o desempenho dos modelos preditivos.
    *   Transformações de variáveis, codificação de categorias, e outras técnicas para preparar os dados para os algoritmos de Machine Learning.

5.  **Desenvolvimento de Modelos Preditivos de Peso e Mortalidade:**
    *   Construção e treinamento de modelos de Machine Learning para prever o peso e a mortalidade dos lotes.
    *   Experimentação com diferentes algoritmos e técnicas de modelagem.

6.  **Validação de Modelos:**
    *   Avaliação rigorosa do desempenho dos modelos utilizando métricas apropriadas (ex: MAE, RMSE para regressão; Acurácia, Precisão, Recall, F1-score para classificação).
    *   Técnicas de validação cruzada para garantir a robustez e generalização dos modelos.

7.  **Criação de um Aplicativo Streamlit:**
    *   Desenvolvimento de um aplicativo interativo usando Streamlit para permitir a inserção de dados com alta importância para o modelo.
    *   Interface amigável para que usuários possam testar o modelo ou fornecer inputs específicos.

8.  **Trabalho na Documentação:**
    *   Atualização contínua da documentação do projeto (incluindo `README.md`, `docs/premissas.md`, e outros documentos técnicos).
    *   Criação de documentação para o código (docstrings) e para o uso dos modelos.

9.  **Geração de Documento para a Equipe de Desenvolvimento (ERP):**
    *   Elaboração de um documento técnico detalhado para a equipe de desenvolvimento do ERP, descrevendo como integrar os modelos preditivos.
    *   Especificações de APIs, formatos de entrada/saída de dados, e requisitos de infraestrutura para a implementação em ambiente de produção.