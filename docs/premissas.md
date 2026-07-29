# Premissas do Projeto: Predição de Peso e Mortalidade de Aves

Este projeto tem como objetivo analisar e prever o peso corporal e a mortalidade de lotes de frangos de corte utilizando técnicas de análise exploratória, tratamento biológico de outliers, modelos não-lineares de curva de crescimento (Gompertz) e algoritmos de Aprendizado de Máquina (Random Forest / Gradient Boosting).

## 1. Estrutura e Qualidade dos Dados

- **Dados Brutos (`data/raw/`):**
  - `extracao_mtech/`: Arquivos semanais Excel (`ag-grid`) contendo pesagens diárias, mortalidade e estoque de aves.
  - `features/`: Tabela `BANCO_VARIAVEIS.xlsx` contendo características fixas dos lotes (tipo de aviário, linhagem do pintainho, idade da matriz, histórico de sanidade, vazios sanitários, etc.).
- **Banco de Dados SQLite (`database/prediction_data.db`):**
  - Armazena as tabelas `extracao_mtech_data` (104.601 linhas), `variables` (11.283 linhas) e `constantes` (1.110 linhas).
- **Tratamento e Qualidade (`data/processed/cleaned_data.csv`):**
  - Remoção de valores inconsistentes de idade (< 1 dia ou > 60 dias).
  - Normalização de peso em quilogramas (kg) e conversão para gramas (g).
  - Filtragem de outliers biológicos por faixa etária utilizando amplitude interquartil (3.0 x IQR). Total final de 88.021 registros limpos e validados.

## 2. Modelagem Preditiva de Crescimento

- **Modelo Não-Linear de Gompertz:**
  $$W(t) = A \cdot \exp(-b \cdot \exp(-k \cdot t))$$
  - $A$: Peso assintótico de amadurecimento ($6.281,25\text{ g}$).
  - $b$: Constante de escala inicial ($4,7536$).
  - $k$: Taxa de maturação relativa ($0,0449\text{ dia}^{-1}$).
  - Desempenho: $R^2 = 0,9848$, $\text{MAE} = 81,81\text{ g}$.
- **Modelo de Machine Learning (Random Forest):**
  - Incorpora idade, quantidade de aves alojadas, peso do pintainho, tempo de vazio sanitário, idade da matriz e distância do abatedouro.
  - Desempenho: $R^2 = 0,9903$, $\text{MAE} = 66,59\text{ g}$, $\text{RMSE} = 102,00\text{ g}$.
  - Validação Cruzada (5-Fold GroupKFold): $R^2 = 0,9897 \pm 0,0004$, $\text{MAE} = 67,70\text{ g} \pm 0,81\text{ g}$.