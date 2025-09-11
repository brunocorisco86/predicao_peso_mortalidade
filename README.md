# Prediction of Mortality Weight

Para as premissas detalhadas do projeto, consulte [docs/premissas.md](docs/premissas.md).
Para o roteiro detalhado do projeto, consulte [docs/workflow.md](docs/workflow.md).
Para o esquema do banco de dados, consulte [docs/modelo_entidade_relacionamento.md](docs/modelo_entidade_relacionamento.md).
Para o conhecimento da biblioteca de features, consulte [docs/conhecimento_biblioteca_features.md](docs/conhecimento_biblioteca_features.md).

## Estrutura do Projeto

```
.
├── data/
│   ├── raw/
│   │   ├── extracao_mtech/ # Informações de peso e mortalidade dos lotes (tabela fato)
│   │   └── features/       # Features de cada lote para modelos de predição
│   └── processed/    # Dados processados
├── assets/           # Elementos de formatação e mídias (ex: logotipos)
├── config/           # Arquivos de configuração do projeto
├── database/         # Arquivos de banco de dados (e.g., SQLite)
├── notebooks/        # Notebooks Jupyter para EDA e prototipagem
├── plots/            # Plots e visualizações geradas
├── src/              # Scripts Python para ETL e modelagem
│   ├── etl/          # Scripts de Extração, Transformação e Carga (ex: extract_mtech_data.py)
│   └── utils/        # Utilitários e módulos auxiliares (ex: logger, export_db_schema.py)
├── .venv/            # Ambiente virtual Python
├── .gitignore        # Arquivo para ignorar arquivos e pastas no Git
├── requirements.txt  # Dependências do projeto
└── README.md         # Este arquivo
```

## Boas Práticas Adicionais

Para garantir a robustez e manutenibilidade do projeto, foram implementadas as seguintes boas práticas:

-   **Gerenciamento de Configurações:** As configurações do projeto (como credenciais de banco de dados e chaves de API) são gerenciadas através da pasta `config/` e carregadas via variáveis de ambiente (usando `.env`). Isso garante a separação de dados sensíveis e flexibilidade entre ambientes.
-   **Sistema de Logging:** Um sistema de logging robusto foi configurado em `src/utils/logger.py` para facilitar a depuração e o monitoramento da aplicação, permitindo diferentes níveis de detalhe e saídas.

## Configuração do Ambiente

Para configurar o ambiente de desenvolvimento, siga os passos abaixo:

1. **Clone o repositório:**
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd prediction_weight_mortality
   ```

2. **Crie e ative o ambiente virtual:**
   ```bash
   python3 -m venv .venv
   # No Windows
   .venv\Scripts\activate
   # No macOS/Linux
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure o kernel Jupyter (opcional, mas recomendado para notebooks):**
   ```bash
   python -m ipykernel install --user --name=.venv --display-name="Python (.venv)"
   ```

## Como Rodar o Projeto

### Processamento de Dados ETL

Para executar os scripts de ETL que processam os dados e os salvam no banco de dados SQLite, utilize os seguintes comandos a partir da raiz do projeto:

-   **Extração e Transformação de Dados de Features (BANCO_VARIAVEIS.xlsx):**
    ```bash
    python -m src.etl.extract_excel_to_db
    ```
    Este script extrai dados das abas 'VARIABLES' e 'CONSTANTES' do arquivo `BANCO_VARIAVEIS.xlsx`, aplica transformações (formatação de cabeçalhos, conversão de datas, tratamento de strings) e os salva nas tabelas `variables` e `constantes` no `prediction_data.db`.

-   **Extração e Transformação de Dados MTECH (Arquivos Semanais):**
    ```bash
    python -m src.etl.extract_mtech_data
    ```
    Este script processa os arquivos semanais de `extracao_mtech`, aplica racionalização de cabeçalhos, processa a coluna `lote_composto` (incluindo a criação de `lote_prefixo`), converte tipos de dados e carrega tudo para a tabela `extracao_mtech_data` no `prediction_data.db`.

### Análise de Qualidade de Dados

-   **Notebook de Qualidade de Dados:**
    Abra o notebook `notebooks/data_quality_extracao_mtech.ipynb` no Jupyter e selecione o kernel `Python (.venv)` para executar as análises de qualidade dos dados da tabela `extracao_mtech_data`.

### Outras Ferramentas

-   **Exportar Esquema do Banco de Dados:**
    Para exportar o esquema do banco de dados para `docs/db_schema.sql` (ou para gerar o MER em `docs/modelo_entidade_relacionamento.md`), utilize:
    ```bash
    python -m src.utils.export_db_schema
    ```

(Instruções detalhadas sobre como rodar outros scripts ou notebooks serão adicionadas aqui conforme o projeto avança.)

## Licença

Este projeto está licenciado sob a Licença Pública Geral GNU (GNU GPLv3). Veja o arquivo `LICENSE` para mais detalhes adicionais.
