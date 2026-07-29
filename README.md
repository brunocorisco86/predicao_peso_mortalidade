# Prediction of Mortality & Weight in Poultry Production

Este repositório contém a solução completa de engenharia de dados, análise exploratória (EDA), tratamento de outliers e modelagem preditiva de peso corporal de frangos de corte, além de diretrizes para integração com a logística de abastecimento de ração.

---

## 📌 Principais Resultados e Métricas

| Modelo | Métrica $R^2$ | MAE (Erro Médio Absoluto) | RMSE (Raiz do Erro Quadrático Médio) |
|---|---|---|---|
| **Curva Não-Linear de Gompertz** | **0,9848** | **81,81 g** | **126,81 g** |
| **Random Forest Regressor (ML)** | **0,9903** | **66,59 g** | **102,00 g** |

### Equação Ajustada de Gompertz:
$$W(t) = 6281,25 \cdot \exp\left(-4,7536 \cdot \exp(-0,0449 \cdot t)\right)$$
onde $W(t)$ representa o peso da ave em gramas na idade de $t$ dias.

---

## 🏗️ Estrutura do Projeto

```
.
├── data/
│   ├── raw/
│   │   ├── extracao_mtech/ # Arquivos semanais de dados brutos (tabela fato)
│   │   └── features/       # BANCO_VARIAVEIS.xlsx (atributos de lote e fazenda)
│   └── processed/
│       ├── unified_data.csv            # Base unificada (104.601 registros)
│       ├── cleaned_data.csv            # Base limpa sem outliers (88.021 registros)
│       └── descriptive_statistics.csv # Estatística descritiva completa
├── database/
│   └── prediction_data.db  # Banco SQLite com as tabelas extracao_mtech_data, variables, constantes
├── docs/
│   ├── modelo_entidade_relacionamento.md # MER e diagrama Mermaid
│   ├── premissas.md                      # Premissas do projeto e os 3 pilares de logística
│   ├── db_schema.sql                     # Esquema exportado do banco
│   └── workflow.md                       # Roteiro do projeto e status das fases
├── plots/                               # Visualizações geradas
│   ├── distribuicao_peso_por_idade.png
│   ├── boxplots_outliers_peso.png
│   ├── distribuicao_mortalidade.png
│   ├── matriz_correlacao_features.png
│   ├── curva_crescimento_gompertz.png
│   ├── predito_vs_observado_peso.png
│   └── importancia_features.png
├── src/
│   ├── etl/
│   │   ├── extract_mtech_data.py
│   │   ├── extract_excel_to_db.py
│   │   └── export_unified_data.py
│   ├── eda_outliers.py                  # Script de estatística e limpeza de outliers
│   ├── models/
│   │   ├── train_predict_weight.py      # Ajuste de Gompertz e treinamento de ML
│   │   └── saved/                       # Modelos serializados (.pkl)
│   └── utils/
│       └── logger.py
├── requirements.txt
└── README.md
```

---

## 🚀 Como Executar o Pipeline Completo

1. **Extração e Unificação de Dados (ETL):**
   ```bash
   python3 -m src.etl.extract_excel_to_db
   python3 -m src.etl.extract_mtech_data
   python3 -m src.etl.export_unified_data
   ```

2. **Análise Exploratória e Limpeza de Outliers (EDA):**
   ```bash
   python3 -m src.eda_outliers
   ```

3. **Treinamento e Avaliação dos Modelos Preditivos:**
   ```bash
   python3 -m src.models.train_predict_weight
   ```

---

## 🚚 Solução para Falhas na Entrega de Ração (3 Pilares)

A predição precisa do peso das aves permite projetar a demanda diária de consumo alimentício por lote. A solução para eliminar falhas no abastecimento de ração apoia-se em três pilares:

1. **Comunicação Eficiente (Plataforma Centralizada):**
   - Unificação das curvas de demanda estimadas em um portal único para integração entre a indústria, logística de frete e o produtor rural.
2. **Processos Otimizados (Redesenho de Fluxo e Confirmação de Pedidos):**
   - Automação da agenda de expedição de ração correlacionada com a curva de ganho de peso predita pelo modelo de Gompertz/ML, com gatilhos obrigatórios de validação antes do envio dos caminhões.
3. **Tecnologia Habilitadora (TMS e Sensores de Nível nos Silos):**
   - Otimização dinâmica de rotas via sistema de gerenciamento de transporte (TMS) combinada com a medição telemétrica em tempo real dos silos das propriedades.

---

## 📄 Licença

Este projeto está licenciado sob a Licença Pública Geral GNU (GNU GPLv3).
