---
rule_id: RN-14
title: Anonimização de Dados Sensíveis e Codificação Hexadecimal de Identificadores (LGPD / Open-Source)
category: Segurança, Privacidade & Governança de Dados
target_table: todas
status: HOMOLOGADO
author: C.Vale DataOps & Privacy Team
last_updated: 2026-07-30
---

# 📌 RN-14: Anonimização de Dados Sensíveis e Codificação Hexadecimal de Identificadores (LGPD / Open-Source)

## 📋 Descrição Geral
A **RN-14** estabelece os protocolos de privacidade, anonimização e sigilo comercial para a disponibilização pública e open-source do repositório e da base de dados do projeto de Predição de Peso e Mortalidade da C.Vale.

Para garantir estrita conformidade com a LGPD (Lei Geral de Proteção de Dados) e preservar a propriedade intelectual e o sigilo dos integrados:
1. Nomes reais de proprietários/produtores (`produtor`), fazendas (`nome_fazenda`), extensionistas (`extensionista`) e usuários (`id_usurio_criao`, `id_usurio`) são substituídos por dados sintéticos gerados deterministicamente via biblioteca **Faker (pt_BR)**.
2. Identificadores numéricos reais de **aviário/fazenda** e **lote** são transformados em códigos **hexadecimais de 4 caracteres** em caixa alta.

---

## ⚙️ Regra de Negócio Técnica

### 1. Codificação Hexadecimal de Identificadores (4 Caracteres)
- **Aviário / Fazenda (`fazenda`):** Mapeado deterministicamente para uma string hexadecimal de 4 caracteres (ex: `105` $\rightarrow$ `3A1F`).
- **Número do Lote (`lote`):** Mapeado deterministicamente para uma string hexadecimal de 4 caracteres (ex: `8` $\rightarrow$ `02B4`).
- **Lote Composto (`lote_composto`):** Estruturado no formato relacional 1:1 `<aviario_hex>-<lote_hex>` (ex: `3A1F-02B4`).

### 2. Anonimização Sintética de Nomes (Faker pt_BR)
```python
# Mapeamento determinístico com seed fixa (ex: seed=42)
produtor -> fake.name()
nome_fazenda -> "Granja " + fake.first_name()
extensionista -> fake.name()
id_usurio -> "user_" + fake.hexify(text="^^^^")
```

---

## 🎯 Impacto no Modelo e DataOps
- **Conformidade Legal & LGPD:** Elimina qualquer dado pessoal identificável (PII) do banco relacional SQLite e dos arquivos CSV/Excel exportados.
- **Manutenção da Integridade Relacional:** Garantia de joins 1:1 exatos entre tabelas (`peso_abate`, `variables`, `constantes`, `extracao_mtech_data`), pois o mapeamento relacional é 100% preservado.
- **Preservação das Regras RN-06 e RN-07:** As especificações estruturais originais da chave composta (RN-06) e a integridade de junção por fazenda (RN-07) permanecem intactas conceitualmente, sendo aplicadas sobre os identificadores anonimizados em formato de 4 caracteres hexadecimais.
