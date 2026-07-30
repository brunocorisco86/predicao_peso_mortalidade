# Regras de Negócio: Predição do Peso de Abate em Frangos de Corte

Este documento define formalmente as regras de negócio, limites biológicos e premissas operacionais para a **predição exclusiva do peso final ao abate** (aves com idade a partir do 42º dia de vida).

---

## 📋 1. Escopo e Objetivo de Negócio

* **Objetivo Principal:** Prever com precisão o peso vivo final corporal das aves no momento do lote de abate ($\text{idade} \ge 42\text{ dias}$), subsidiando o planejamento industrial dos abatedouros e classificação comercial do lote.
* **Público-Alvo:** Equipes de Planejamento e Controle da Produção (PCP), extensão rural e gestão agrícola.

---

## ⚙️ 2. Regras de Negócio e Filtros Biológicos

| Regra | Parâmetro | Valor / Intervalo Aceito | Justificativa Zootécnica / Operacional |
|---|---|---|---|
| **RN-01: Janela de Idade de Abate** | `idade` | **$42 \le \text{idade} \le 60\text{ dias}$** | Idade comercial de abate para frangos de corte de médio/pesado porte. Registros com idade $< 42$ dias são desconsiderados nesta análise. |
| **RN-02: Limites de Peso de Abate** | `peso` | **$1,80\text{ kg} \le \text{peso} \le 4,80\text{ kg}$** ($1.800\text{g}$ a $4.800\text{g}$) | Pesos fora desta faixa no abate representam erro de pesagem, lote fora de padrão comercial extremo ou erro de digitação. |
| **RN-03: Integridade do Lote** | `cab_alojadas` | **$> 0$** | Lotes sem cabeças alojadas válidas são desconsiderados. |
| **RN-04: Sanidade Plausível** | `mortalidade` | **$0 \le \text{mortalidade} \le \text{cab\_alojadas}$** | A mortalidade acumulada até o abate não pode exceder o total de aves alojadas. |
| **RN-05: Filtro Estatístico de Outliers** | `IQR` por dia de abate | **$Q_1 - 3.0 \times \text{IQR} \le \text{peso} \le Q_3 + 3.0 \times \text{IQR}$** | Limpeza de divergências extremas mantendo a variabilidade real entre fazendas e produtores. |
| **RN-06: Identificação de Aviário / Fazenda** | `aviario` / `fazenda` | Número antes do hífen em `LoteComposto` | O aviário (ou **fazenda**) é representado exclusivamente pelo número inteiro localizado antes do primeiro hífen do `LoteComposto` (ex: `100` em `100-39` ou `1223` em `1223-8`). |
| **RN-07: Composição do Lote Composto** | `lote_composto` / `LoteComposto` | `<aviario>-<lote>` ou `<aviario>-<lote>-N<nucleo>` | O identificador `Lote Composto` (ou `LoteComposto`) é formado pela concatenação do aviário e lote (ex: `1223-8`), podendo opcionalmente incluir a identificação do núcleo prefixado por 'N' (ex: `1223-8-N1`). |
| **RN-08: Preferência por Taxas Relativas (%)** | `_mortalidade` / `_descartados` | Porcentagem acumulada (%) | Sempre priorizar taxas relativas em % (`_mortalidade`, `_descartados`) em vez de contagens brutas de cabeças (`mortalidade`, `descartados`) para evitar colinearidade e ruídos em modelos preditivos. |
| **RN-09: Elegibilidade Mínima de Pesagens** | `qtd_pesagens`, `tem_pesagem_35d` | Mínimo 3 pesagens + Pesagem obrigatória de 35d | Para entrada no modelo preditivo, o lote DEVE ter no mínimo 3 pesagens de campo e obrigatoriamente a pesagem dos 35 dias (janela 33-37d). Lotes sem pesagem aos 35d são descartados. Ter pesagem aos 42d é altamente desejável. |
| **RN-10: Índice de Confiança de Amostragem** | `score_confianca_amostragem` | Pontuação contínua de 0,0 a 10,0 | Pontuação atribuída ao lote e agregada por aviário/fazenda baseada na frequência de pesagens, presença dos marcos críticos (35d e 42d) e regularidade do GMD amostral, integrando os datasets preditivos. |
| **RN-11: Delineamento Amostral Mínimo para Modelagem Preditiva** | `elegivel_rn11`, `score_confianca_lote` | Score $\ge 7.5$, $\ge 3$ pesagens, 35d obrigatório | Define o filtro final de entrada para a esteira de Machine Learning. Lotes com `score_confianca_lote < 7.5` ou sem pesagem de 35d utilizam a estratégia de fallback (média histórica da fazenda) em vez de modelo preditivo direto. |

---

## 🎯 3. Classificação de Desempenho do Lote ao Abate (Target Categorizado)

Para avaliação de classificação (Matriz de Confusão), o peso real do lote no dia do abate é categorizado em 3 classes com base nos quartis da idade:

* 🔴 **Abaixo da Meta:** $\text{peso} < P_{25}$ do dia do abate.
* 🟢 **Na Meta:** $P_{25} \le \text{peso} \le P_{75}$ do dia do abate.
* 🔵 **Acima da Meta:** $\text{peso} > P_{75}$ do dia do abate.

---

## 🧠 4. Variáveis Explicativas do Lote

O modelo preditivo de abate utiliza prioritariamente as características fixas e operacionais do lote:

1. **Atributos Iniciais do Pintainho:** Peso do pintainho de 1 dia (`c15`), linhagem (`c16`), fornecedor (`c17`).
2. **Sanidade e Manejo Sanitário:** Idade da matriz (`c05`, `c06`), tempo de vazio sanitário (`f01`-`f03`), número de reutilizações da cama (`f04`-`f06`).
3. **Estrutura e Logística:** Tipo de aviário (`a01`-`a03`), densidade de alojamento (`cab_alojadas`), distância do abatedouro (`x02`).
4. **Mortalidade e Descarte:** Ocorrência de mortes e descartes até o momento do abate.
