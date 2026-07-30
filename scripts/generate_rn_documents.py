"""
generate_rn_documents.py
-------------------------
Gera um documento Markdown dedicado com cabeçalho YAML Frontmatter para CADA
uma das 13 Regras de Negócio (RN-01 a RN-13) da C.Vale no diretório docs/regras_de_negocio/.
"""

from pathlib import Path

RN_DATA = [
    {
        "filename": "rn01_filtro_idade_abate.md",
        "rule_id": "RN-01",
        "title": "Filtro Biológico de Idade de Abate Frigorífico",
        "category": "Filtro Biológico & Qualidade de Dados",
        "target_table": "peso_abate",
        "status": "HOMOLOGADO",
        "summary": "Restringe os registros de abate para idades comerciais válidas entre 42 e 60 dias de criatório.",
        "content": """---
rule_id: RN-01
title: Filtro Biológico de Idade de Abate Frigorífico
category: Filtro Biológico & Qualidade de Dados
target_table: peso_abate
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-01: Filtro Biológico de Idade de Abate Frigorífico

## 📋 Descrição Geral
A **RN-01** define o intervalo etário comercialmente aceito para o abate de frangos de corte nas plantas industriais da C.Vale. Registros fora da janela etária de $42 \le \text{idade\_abate} \le 60\text{ dias}$ são descartados por caracterizarem erros de digitação ou lotes experimentais/descarte extraordinário.

---

## ⚙️ Regra de Negócio Técnica
```sql
WHERE idade_abate >= 42 AND idade_abate <= 60
```

---

## 🎯 Impacto no Modelo e DataOps
- **Eliminação de Ruído:** Remove lotes abatidos precocemente (ex: 20 dias por emergência sanitária) ou aves matrizes velhas (>60 dias).
- **Consistência do Target:** Garante que a variável dependente `peso_abate_g` reflita a janela operacional de expedição do PCP.
"""
    },
    {
        "filename": "rn02_filtro_peso_plausivel.md",
        "rule_id": "RN-02",
        "title": "Filtro Biológico de Peso de Abate Plausível",
        "category": "Filtro Biológico & Qualidade de Dados",
        "target_table": "peso_abate",
        "status": "HOMOLOGADO",
        "summary": "Limita o peso médio do lote no abate entre 1.800g e 4.800g para eliminar erros de balança e digitação.",
        "content": """---
rule_id: RN-02
title: Filtro Biológico de Peso de Abate Plausível
category: Filtro Biológico & Qualidade de Dados
target_table: peso_abate
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-02: Filtro Biológico de Peso de Abate Plausível

## 📋 Descrição Geral
A **RN-02** valida o peso médio corporativo dos frangos entregues na plataforma do abatedouro. Pesos abaixo de $1.800\text{g}$ ou acima de $4.800\text{g}$ representam inconsistência de digitação (ex: peso em kg gravado como g) ou falha na integração do sistema MTech.

---

## ⚙️ Regra de Negócio Técnica
```sql
WHERE peso_abate_g BETWEEN 1800 AND 4800
```

---

## 🎯 Impacto no Modelo e DataOps
- **Integridade da Escala:** Evita valores fisiologicamente impossíveis no target de regressão.
- **Estabilidade do Loss:** Impede explosão de gradiente ($RMSE$) durante o treinamento em GPU CUDA.
"""
    },
    {
        "filename": "rn03_remocao_outliers_iqr.md",
        "rule_id": "RN-03",
        "title": "Eliminação de Outliers de Peso por Idade via IQR",
        "category": "Tratamento Estatístico & Sanitização",
        "target_table": "extracao_mtech_data",
        "status": "HOMOLOGADO",
        "summary": "Aplica o filtro da Amplitude Interquartil (1.5x IQR) por faixa etária semanal para remover erros de amostragem no campo.",
        "content": """---
rule_id: RN-03
title: Eliminação de Outliers de Peso por Idade via IQR
category: Tratamento Estatístico & Sanitização
target_table: extracao_mtech_data
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-03: Eliminação de Outliers de Peso por Idade via IQR

## 📋 Descrição Geral
A **RN-03** realiza a limpeza estatística dos pesos amostrados pelos técnicos de campo ao longo do criatório (MTech). Pesagens discrepantes são filtradas dentro de cada faixa etária de referência utilizando a regra de Tukey ($1,5 \times \text{IQR}$).

---

## ⚙️ Regra de Negócio Técnica
Para cada idade $t \in \{4, 7, 14, 21, 28, 35, 42\}$:
$$\text{Limite Inferior} = Q_1 - 1,5 \times \text{IQR}$$
$$\text{Limite Superior} = Q_3 + 1,5 \times \text{IQR}$$

Registros fora dessa faixa são removidos da tabela `extracao_mtech_data`.

---

## 🎯 Impacto no Modelo e DataOps
- **Proteção contra balança descalibrada:** Descarta pesagens pontuais errôneas registradas por erro humano no campo.
- **Suavização da Curva:** Garante que a velocidade de ganho diário ($GMD$) reflita a biometria real do lote.
"""
    },
    {
        "filename": "rn04_eliminacao_duplicatas.md",
        "rule_id": "RN-04",
        "title": "Eliminação de Duplicatas Absolutas de Registro",
        "category": "Integridade de Dados & Governança",
        "target_table": "extracao_mtech_data",
        "status": "HOMOLOGADO",
        "summary": "Remove duplicatas perfeitas de amostragem geradas por retransmissão de lote na API MTech.",
        "content": """---
rule_id: RN-04
title: Eliminação de Duplicatas Absolutas de Registro
category: Integridade de Dados & Governança
target_table: extracao_mtech_data
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-04: Eliminação de Duplicatas Absolutas de Registro

## 📋 Descrição Geral
A **RN-04** elimina registros idênticos sincronizados repetidamente pelo sistema de integração do MTech durante falhas de rede ou sincronizações duplas de dispositivos móveis da extensão rural.

---

## ⚙️ Regra de Negócio Técnica
```sql
DELETE FROM extracao_mtech_data 
WHERE rowid NOT IN (
    SELECT MIN(rowid) 
    FROM extracao_mtech_data 
    GROUP BY lote_composto, idade, peso, data_hora_transacao
);
```

---

## 🎯 Impacto no Modelo e DataOps
- **Integridade Relacional:** Evita ponderação dupla de pesagens na média diária do lote.
"""
    },
    {
        "filename": "rn05_saneamento_datas.md",
        "rule_id": "RN-05",
        "title": "Saneamento de Cronologia e Datas Incompatíveis",
        "category": "Governança Temporária & Sanidade",
        "target_table": "variables",
        "status": "HOMOLOGADO",
        "summary": "Valida a sequência lógica de datas: Data Alojamento <= Data Evento <= Data Abate.",
        "content": """---
rule_id: RN-05
title: Saneamento de Cronologia e Datas Incompatíveis
category: Governança Temporária & Sanidade
target_table: variables
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-05: Saneamento de Cronologia e Datas Incompatíveis

## 📋 Descrição Geral
A **RN-05** verifica a sanidade cronológica de todos os eventos da vida do lote. Registros onde a data de pesagem precede a data de alojamento do pintainho ou sucede a data de abate no frigorífico são rejeitados.

---

## ⚙️ Regra de Negócio Técnica
$$\text{Data Alojamento} \le \text{Data Evento Amostragem} \le \text{Data Abate Frigorífico}$$

---

## 🎯 Impacto no Modelo e DataOps
- **Prevenção de Anacronismos:** Evita que pesagens pós-abate contaminem o histórico preditivo do lote.
"""
    },
    {
        "filename": "rn06_padronizacao_chave_lote.md",
        "rule_id": "RN-06",
        "title": "Padronização da Chave Relacional 1:1 (Lote Composto)",
        "category": "Modelagem Dimensional & Arquitetura",
        "target_table": "todas",
        "status": "HOMOLOGADO",
        "summary": "Cria o identificador único 'lote_composto' concatenando fazenda, produtor e lote de origem.",
        "content": """---
rule_id: RN-06
title: Padronização da Chave Relacional 1:1 (Lote Composto)
category: Modelagem Dimensional & Arquitetura
target_table: todas
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-06: Padronização da Chave Relacional 1:1 (Lote Composto)

## 📋 Descrição Geral
A **RN-06** estabelece a chave relacional primária universal em todas as tabelas do banco SQLite `database/prediction_data.db`, concatenando o código da fazenda, o código do produtor e o número do lote.

---

## ⚙️ Regra de Negócio Técnica
```sql
lote_composto = CAST(fazenda AS TEXT) || '-' || CAST(produtor AS TEXT) || '-' || CAST(lote AS TEXT)
```

---

## 🎯 Impacto no Modelo e DataOps
- **Integridade de Join:** Permite fusão exata de 1:1 entre biometrias, variáveis zootécnicas, constantes e abate.
"""
    },
    {
        "filename": "rn07_coercao_fazenda_integer.md",
        "rule_id": "RN-07",
        "title": "Coerção de Formato do Código da Fazenda",
        "category": "Qualidade de Dados & Tipagem",
        "target_table": "constantes",
        "status": "HOMOLOGADO",
        "summary": "Garante a conversão do código da fazenda para INTEGER eliminando sufixos flutuantes (.0).",
        "content": """---
rule_id: RN-07
title: Coerção de Formato do Código da Fazenda
category: Qualidade de Dados & Tipagem
target_table: constantes
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-07: Coerção de Formato do Código da Fazenda

## 📋 Descrição Geral
A **RN-07** força a tipagem do código da fazenda para valor inteiro, eliminando formatações incorretas de ponto flutuante (ex: `105.0`) originadas da importação de arquivos Excel/CSV.

---

## ⚙️ Regra de Negócio Técnica
```python
df['fazenda'] = df['fazenda'].fillna(0).astype(int)
```

---

## 🎯 Impacto no Modelo e DataOps
- **Sanidade de Chave Extrangeira:** Impede falhas de junção entre a tabela `constantes` e as demais tabelas Fato.
"""
    },
    {
        "filename": "rn08_taxas_relativas_percentuais.md",
        "rule_id": "RN-08",
        "title": "Priorização de Taxas Relativas (%) sobre Contagens Absolutas",
        "category": "Engenharia de Features Zootécnicas",
        "target_table": "variables",
        "status": "HOMOLOGADO",
        "summary": "Substitui contagens brutas de mortalidade e descartes por taxas percentuais (_mortalidade, _descartados) ajustadas pela densidade.",
        "content": """---
rule_id: RN-08
title: Priorização de Taxas Relativas (%) sobre Contagens Absolutas
category: Engenharia de Features Zootécnicas
target_table: variables
status: HOMOLOGADO
author: C.Vale DataOps Team (zootecnia-data-rules)
last_updated: 2026-07-30
---

# 📌 RN-08: Priorização de Taxas Relativas (%) sobre Contagens Absolutas

## 📋 Descrição Geral
A **RN-08** estabelece que variáveis de mortalidade e descartes devem ser expressas como **taxas percentuais relativas ao total alojado** (`cab_alojadas`), eliminando o viés de tamanho do aviário.

---

## ⚙️ Regra de Negócio Técnica
$$\text{taxa\_mortalidade\_pct} = \frac{\text{cabeças\_mortas}}{\text{cab\_alojadas}} \times 100$$
$$\text{taxa\_descarte\_pct} = \frac{\text{cabeças\_descartadas}}{\text{cab\_alojadas}} \times 100$$

---

## 🎯 Impacto no Modelo e DataOps
- **Isonomia entre Granjas:** Permite comparar aviários de 15.000 aves com aviários de 40.000 aves sem distorção.
- **Normalização para ML:** Melhora o condicionamento das matrizes em modelos baseados em gradiente.
"""
    },
    {
        "filename": "rn09_idades_referencia_amostragem.md",
        "rule_id": "RN-09",
        "title": "Mapeamento de Idades de Referência de Amostragem",
        "category": "Séries Temporais & Discretização",
        "target_table": "extracao_mtech_data",
        "status": "HOMOLOGADO",
        "summary": "Mapeia as pesagens semanais nas 7 faixas biométricas padrão (4, 7, 14, 21, 28, 35, 42 dias +/- 1 dia).",
        "content": """---
rule_id: RN-09
title: Mapeamento de Idades de Referência de Amostragem
category: Séries Temporais & Discretização
target_table: extracao_mtech_data
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-09: Mapeamento de Idades de Referência de Amostragem

## 📋 Descrição Geral
A **RN-09** discretiza as idades brutas de amostragem do MTech em 7 marcos etários zootécnicos padrão, permitindo o pivotamento longitudinal dos pesos por lote.

---

## ⚙️ Regra de Negócio Técnica
| Idade de Referência (`idade_ref`) | Intervalo de Idade Real Permitido |
|---|---|
| **4d** | [3, 5] dias |
| **7d** | [6, 8] dias |
| **14d** | [13, 15] dias |
| **21d** | [20, 22] dias |
| **28d** | [27, 29] dias |
| **35d** | [34, 36] dias |
| **42d** | [41, 43] dias |

Pesagens fora dessas janelas de $\pm 1$ dia são descartadas da matriz pivô.

---

## 🎯 Impacto no Modelo e DataOps
- **Estandardização Temporal:** Habilita o cálculo exato de velocidades $GMD$ entre semanas consecutivas.
"""
    },
    {
        "filename": "rn10_score_confianca_lote.md",
        "rule_id": "RN-10",
        "title": "Cálculo do Score de Confiança de Amostragem do Lote",
        "category": "Data Quality & Governança",
        "target_table": "lote_sampling_confidence",
        "status": "HOMOLOGADO",
        "summary": "Pontua a qualidade da amostragem do lote de 0.0 a 10.0 com base na cobertura etária e volume de aves pesadas.",
        "content": """---
rule_id: RN-10
title: Cálculo do Score de Confiança de Amostragem do Lote
category: Data Quality & Governança
target_table: lote_sampling_confidence
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-10: Cálculo do Score de Confiança de Amostragem do Lote

## 📋 Descrição Geral
A **RN-10** calcula um indicador sintético de qualidade de dados (`score_confianca_lote`) para medir a confiabilidade biométrica de cada lote de frangos.

---

## ⚙️ Regra de Negócio Técnica
$$\text{Score} = (\text{Qtd Marcos Presentes}) \times 1,25 + (\text{Presença de 35d}) \times 1,25$$

O score varia de $0,0$ (sem dados) a $10,0$ (amostragem perfeita em todos os marcos e 35d presente).

---

## 🎯 Impacto no Modelo e DataOps
- **Filtro de Governança:** Alimenta diretamente o Gateway de Elegibilidade RN-11.
"""
    },
    {
        "filename": "rn11_gateway_elegibilidade_ml.md",
        "rule_id": "RN-11",
        "title": "Gateway de Elegibilidade para o Modelo Direto de ML",
        "category": "Arquitetura Preditiva & Resiliência",
        "target_table": "lote_sampling_confidence",
        "status": "HOMOLOGADO",
        "summary": "Roteia lotes qualificados (score >= 7.5 e pesagem de 35d presente) para o Stacking GPU e os demais para o Fallback da Fazenda.",
        "content": """---
rule_id: RN-11
title: Gateway de Elegibilidade para o Modelo Direto de ML
category: Arquitetura Preditiva & Resiliência
target_table: lote_sampling_confidence
status: HOMOLOGADO
author: C.Vale DataOps & MLOps Team
last_updated: 2026-07-30
---

# 📌 RN-11: Gateway de Elegibilidade para o Modelo Direto de ML

## 📋 Descrição Geral
A **RN-11** é o roteador de resiliência em produção. Ela impede que lotes com dados insuficientes sofram extrapolações errôneas no modelo preditivo direto.

---

## ⚙️ Regra de Negócio Técnica
```python
if score_confianca_lote >= 7.5 and possui_pesagem_35d == True:
    elegivel_rn11 = 1  # Roteia para Stacking GPU (Modelo Direto ML)
else:
    elegivel_rn11 = 0  # Roteia para Fallback Conservador (Média da Fazenda)
```

---

## 🎯 Impacto no Modelo e DataOps
- **Zero Fail em Produção:** Garante $100\%$ de entrega no PCP, mesmo se o campo falhar na coleta dos dados.
"""
    },
    {
        "filename": "rn12_gemeos_digitais_knn.md",
        "rule_id": "RN-12",
        "title": "Gêmeos Digitais por Matriz KNN de Fazenda",
        "category": "Feature Engineering & Imputação Contextual",
        "target_table": "lote_rn12_digital_twins",
        "status": "HOMOLOGADO",
        "summary": "Calcula preditores de vizinhança K=15 para capturar o padrão produtivo histórico da fazenda sem data leakage.",
        "content": """---
rule_id: RN-12
title: Gêmeos Digitais por Matriz KNN de Fazenda
category: Feature Engineering & Imputação Contextual
target_table: lote_rn12_digital_twins
status: HOMOLOGADO
author: C.Vale Data Science Team
last_updated: 2026-07-30
---

# 📌 RN-12: Gêmeos Digitais por Matriz KNN de Fazenda

## 📋 Descrição Geral
A **RN-12** implementa a estratégia de Gêmeos Digitais. O sistema identifica os $K=15$ lotes históricos mais parecidos da mesma fazenda e aviário para fornecer um baseline biológico robusto.

---

## ⚙️ Regra de Negócio Técnica
- Matriz de Distância Euclidiana Padronizada via `StandardScaler`.
- Imputação e projeção via `sklearn.neighbors.NearestNeighbors(n_neighbors=15)`.
- Treinamento estritamente dentro dos folds de validação cruzada `GroupKFold` por `lote_composto` (Zero Leakage).

---

## 🎯 Impacto no Modelo e DataOps
- **Recuperação de Dados Faltantes:** Permite estimar a tendência da linhagem mesmo quando a sigla `c16` não foi cadastrada no incubatório.
"""
    },
    {
        "filename": "rn13_suavizacao_isotonica_monotonica.md",
        "rule_id": "RN-13",
        "title": "Detecção de Inversão Biométrica & Suavização Isotônica Monotônica",
        "category": "DataOps & Filtro Fisiológico",
        "target_table": "longitudinal_dataset",
        "status": "HOMOLOGADO",
        "summary": "Identifica inversões biométricas (queda de peso > 5%) e aplica IsotonicRegression(increasing=True) para restaurar a monotonicidade.",
        "content": """---
rule_id: RN-13
title: Detecção de Inversão Biométrica & Suavização Isotônica Monotônica
category: DataOps & Filtro Fisiológico
target_table: longitudinal_dataset
status: HOMOLOGADO
author: C.Vale DataOps Team
last_updated: 2026-07-30
---

# 📌 RN-13: Detecção de Inversão Biométrica & Suavização Isotônica Monotônica

## 📋 Descrição Geral
A **RN-13** trata da correção de erros de amostragem de balança no campo. Biologicamente, um lote de frangos saudável em crescimento não perde peso corporal significativo entre duas semanas consecutivas ($W_{t+1} < 0,95 W_t$). Quando essa inversão ocorre por erro humano ou amostragem viesada, a RN-13 ajusta a curva.

---

## ⚙️ Regra de Negócio Técnica
```python
from sklearn.isotonic import IsotonicRegression

# Se detectada queda > 5% entre pesagens válidas
if valid_weights[i+1] < valid_weights[i] * 0.95:
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    smoothed_weights = iso.fit_transform(valid_ages, valid_weights)
```

---

## 🎯 Impacto no Modelo e DataOps
- **Fisiologia Garantida:** Restaura a monotonicidade não-decrescente da curva de crescimento do lote.
- **Melhoria no $R^2$:** Eleva a estabilidade das acelerações e velocidades $GMD$ utilizadas pelas árvores de decisão.
"""
    }
]

def generate_all_rn_docs():
    output_dir = Path("docs/regras_de_negocio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=======================================================")
    print(" 🛠️ GERANDO DOCUMENTOS INDIVIDUAIS DE REGRAS DE NEGÓCIO")
    print("=======================================================")
    
    for item in RN_DATA:
        filepath = output_dir / item["filename"]
        filepath.write_text(item["content"].strip() + "\n", encoding="utf-8")
        print(f" ✅ Documento Gerado: {filepath}")
        
    print("\n=======================================================")
    print(f" Total de 13 Documentos de RN Gerados com Frontmatter em: {output_dir}")
    print("=======================================================\n")

if __name__ == '__main__':
    generate_all_rn_docs()
