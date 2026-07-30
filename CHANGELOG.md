# 📜 Changelog Oficial - Projeto Predição de Peso de Abate C.Vale

Todas as alterações relevantes, refatorações, auditorias e atualizações de código deste repositório estão documentadas neste arquivo.

---

## [v2.6.0] - 2026-07-30

### 🔒 Segurança, Privacidade & Open-Source (RN-14)
* **Anonimização Determinística com Faker (pt_BR):** Nomes reais de produtores (`produtor`), fazendas (`nome_fazenda`), extensionistas (`extensionista`) e usuários (`id_usurio`) substituídos por dados sintéticos determinísticos.
* **Codificação Hexadecimal de 4 Caracteres:** Número de aviário/fazenda (`fazenda`) e lote (`lote`) convertidos em strings hexadecimais de 4 caracteres (ex: `3A1F`, `02B4`), com `lote_composto` padronizado como `<aviario_hex>-<lote_hex>`.
* **Regra de Negócio RN-14:** Formalizada em `docs/regras_de_negocio/rn14_anonimizacao_hash_hexadecimal.md` preservando a estrutura conceitual das regras RN-06 e RN-07.
* **Saneamento Completo de Artefatos:** Atualizados banco de dados SQLite (`database/prediction_data.db`), datasets CSVs em `data/processed/`, DDLs em `docs/db_schema.sql` e scripts ETL/ML.

---

## [v2.5.0] - 2026-07-30

### 🌟 Adicionado
* **Gradiente Frio-para-Quente no Heatmap 2D:** Substituição da paleta de cores para `turbo` / `coolwarm` no plot de densidade 2D (`plots/estatistica/03_intervalos_confianca_predicao.png` e célula 14 do notebook).
* **Métricas de Classificação Comercial PCP:** Matriz de Confusão renderizada inline no notebook + F1-Score Weighted ($78,36\%$) e F1-Score Macro ($75,12\%$) para a separação entre Frango Leve ($<3.000\text{g}$), Médio ($3.000-3.350\text{g}$) e Pesado ($>3.350\text{g}$).
* **Estatística Descritiva Inicial & Resíduos:** Tabela de EDA com Média, Desvio Padrão, Mínimo, P25, Mediana, P75, Máximo e IQR para todas as biometrias ($D01-D42$), além de Q-Q Plot de Probabilidade Normal e Histograma KDE de Resíduos.
* **One-Hot Encoding da Linhagem Genética (`c16`) & Incubatório (`c17`):** Codificação direta dos tipos `COBB MALE`, `ROSS AP`, `HUBBARD` e `DESCONHECIDO` em `build_longitudinal_features.py` e inclusão em $X$.
* **Relatório Executivo para a Diretoria:** Documento [`docs/apresentacao_diretoria_modelo_preditivo.md`](docs/apresentacao_diretoria_modelo_preditivo.md) detalhando retorno financeiro, impacto da genômica ($+87,1\text{g}$ no Cobb Male), estresse térmico ($y01$), pontos de falha e soluções operacionais.
* **Slide Decks & Storyboards dos 5 Stakeholders:** Documentos [`docs/roteiro_apresentacao_stakeholders.md`](docs/roteiro_apresentacao_stakeholders.md) e [`docs/storyboard_apresentacoes_stakeholders.md`](docs/storyboard_apresentacoes_stakeholders.md) com o roteiro visual e narrativo slide a slide para Diretoria, PCP, Extensão Rural, Produtores e MLOps.
* **Grafo Graphify Tradicional:** Atualização do visualizador em [`graphify-out/graph.html`](graphify-out/graph.html) e relatório semântico em [`graphify-out/GRAPH_REPORT.md`](graphify-out/GRAPH_REPORT.md) com 156 nós e 196 arestas do projeto.

---

## [v2.0.0] - 2026-07-30

### 🚀 Adicionado
* **Esteira de Explicabilidade SHAP (TreeSHAP):** Script `src/explainability/generate_shap_analysis.py` gerando gráficos de Beeswarm, Bar plot e Dependência Parcial para GMD e Peso do Pintainho (`c15`).
* **Manual de Comissionamento & Implantação:** Documento [`docs/commissioning/manual_comissionamento_modelo.md`](docs/commissioning/manual_comissionamento_modelo.md) cobrindo SLAs de resposta ($<2,5$s), Shadow Testing de 14 dias, roteador de resiliência e procedimento de rollback.
* **Roadmap de Execução Operacional:** Diagrama Mermaid e detalhamento em 5 etapas em [`docs/roadmap_execucao_predicao.md`](docs/roadmap_execucao_predicao.md).
* **RN-13 (Suavização Isotônica Monotônica):** Implementação de `sklearn.isotonic.IsotonicRegression` para tratamento automático de inversões biométricas no campo ($W_{t+1} < W_t \cdot 0,95$).
* **Modelo Campeão GPU Stacking:** Stacking Ensemble em GPU CUDA com $R^2 = 0,6870$, $\text{MAPE} = 3,18\%$ e $\text{MAE} = 101,39\text{g}$ no PCP ($102,90\text{g}$ global).

---

## [v1.0.0] - 2026-07-29

### 🛠️ Inicial
* Carga e normalização 3NF da base SQLite `prediction_data.db`.
* Mapeamento inicial das Regras de Negócio (RN-01 a RN-12).
* Validação cruzada via 5-Fold GroupKFold por `lote_composto`.
