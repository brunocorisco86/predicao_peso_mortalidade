# Premissas do Projeto

Este projeto tem como objetivo analisar e prever o peso de mortalidade utilizando técnicas de análise de dados e aprendizado de máquina. As premissas incluem:

- **Coleta de Dados:** Os dados brutos serão armazenados na pasta `data/raw/`, subdividida em:
    - `extracao_mtech/`: Contém informações de peso e mortalidade dos lotes, dispostos em vários arquivos Excel com aba 'ag-grid' (tabela fato).
    - `features/`: Contém as features de cada lote a serem incorporadas com os dados de `extracao_mtech` para modelos de predição.
- **Qualidade dos Dados:** Será realizada uma avaliação da qualidade dos dados da tabela `extracao_mtech_data` para identificar e tratar inconsistências, valores ausentes e outliers.
- **Análise Exploratória:** Notebooks Jupyter serão utilizados para a análise exploratória de dados (EDA) e prototipagem de modelos, localizados na pasta `notebooks/`.
- **ETL e Modelagem:** Scripts Python para extração, transformação e carregamento (ETL) de dados, bem como para a construção e treinamento de modelos, serão armazenados na pasta `src/`.
- **Ambiente Virtual:** Um ambiente virtual Python (`.venv`) será utilizado para gerenciar as dependências do projeto.
- **Gerenciamento de Dependências:** Todas as bibliotecas e suas versões serão listadas no arquivo `requirements.txt`.
- **Controle de Versão:** O controle de versão será feito via Git e GitHub.