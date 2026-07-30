# Manual Oficial de Comissionamento e Implantação do Modelo

## 1. Introdução
Este manual descreve as diretrizes operacionais e de Governança de Machine Learning para a implantação do modelo final (Predição de Peso de Aves e Mortalidade) nos ambientes de homologação e produção da C.Vale.

## 2. Service Level Agreement (SLA)
A arquitetura de inferência do modelo tem os seguintes acordos de nível de serviço (SLA):
- **Tempo de Resposta (Inference Latency):** P95 < 2.5 segundos para requisições em batch de até 500 lotes;
- **Disponibilidade (Uptime):** 99.9% de disponibilidade em horário operacional de PCP (Planejamento e Controle da Produção).
- **Frequência de Atualização dos Dados (Batch Ingestion):** Re-scoring diário dos lotes previstos para abate nas próximas 2 semanas.

## 3. Plano de Testes de Aceitação (PCP do Abatedouro)
Antes da virada de chave no fluxo operacional do abatedouro, a equipe de PCP deve executar testes de aceitação em ambiente *Shadow Mode*:
1. **Período de Comparação:** Acompanhar as predições de modelo vs. amostragens reais de campo (pesagens aos 28 e 35 dias) por 14 dias contínuos.
2. **KPIs a validar:**
   - MAE <= Limiar de Aceitação (Aderência do peso real de abate vs. predito).
   - MAE no segmento de Idade (42-47 dias e 48+ dias) devendo se manter estável;
3. **Aceitação do Processo:** O encarregado de abate ratifica que o uso da predição reduz a divergência do planejamento industrial e aciona a homologação.

## 4. Monitoramento de Saúde do Modelo (Model Health & Drift)
Para evitar a degradação de performance do modelo frente a sazonalidades ou mudanças na genética:
- **Monitoramento de Data Drift:** Monitoramento quinzenal (ex: via teste de Kolmogorov-Smirnov) nas principais features apontadas no relatório SHAP (GMD, peso do pintainho c15, pesos intermediários).
- **Monitoramento de Concept Drift:** Avaliação semanal do MAE do modelo contra os dados reais de abate retornados do frigorífico. Alertas devem ser gerados se o erro médio superar 5% (MAPE > 5%).
- **Retreinamento:** O modelo é agendado para ciclo de avaliação de retreinamento a cada 3 meses.

## 5. Procedimentos de Rollback
Em caso de degradação crítica, indisponibilidade do banco de dados analítico ou falha na integração via API:
1. **Fallback para Histórico (Cold-start):** Utilizar médias históricas estratificadas por idade e produtor como baseline substituto temporário.
2. **Reversão de Versão de Modelo:** Ativar a versão N-1 do modelo campeão no Model Registry, garantindo o retorno da integração em até 1 hora útil.
3. **Comunicado à Operação:** Disparar alerta imediato à operação de que os pesos planejados estão sob baseline, aumentando margem de erro.

## 6. Checklist de Comissionamento Final
- [ ] Validação SHAP e Documentação de Explicabilidade concluída.
- [ ] Gráficos de Resíduos e Otimização gerados.
- [ ] Teste A/B (Shadow Mode) concluído pelo Abatedouro.
- [ ] Integração com repositório unificado / Model Registry feita.
- [ ] Monitoramento de Drift configurado.
- [ ] Manual operacional e de Rollback aprovado pela gestão.
