-- SQLite Database Schema Export
-- Database: /home/brunoconter/Documentos/1_C.VALE/1 - ANALISES/10 - PESO DAS AVES/prediction_weight_mortality/database/prediction_data.db
-- Exported on: 2026-07-30 13:03:24.027201

-- Table: peso_abate
CREATE TABLE "peso_abate" (
"lote_composto" TEXT,
  "idade_abate" INTEGER,
  "data_producao" TEXT,
  "peso_medio_abate_kg" REAL,
  "gmd_abate" REAL,
  "peso_abate_g" REAL,
  "fazenda" TEXT
);

-- Table: extracao_mtech_data
CREATE TABLE "extracao_mtech_data" (
"data_alojamento" TEXT,
  "nome_fazenda" TEXT,
  "data_hora_transao" TEXT,
  "lote_composto" TEXT,
  "data_evento" TEXT,
  "idade" REAL,
  "cab_alojadas" REAL,
  "estoque_aves" REAL,
  "mortalidade" INTEGER,
  "_mortalidade" REAL,
  "descartados" INTEGER,
  "_descartados" REAL,
  "_mortalidade__descartados" REAL,
  "data_criao" TEXT,
  "peso" REAL,
  "id_usurio_criao" TEXT,
  "extensionista" TEXT,
  "id_usurio" TEXT,
  "fazenda" TEXT,
  "idade_ref" INTEGER
);

-- Table: lote_sampling_confidence
CREATE TABLE "lote_sampling_confidence" (
"lote_composto" TEXT,
  "fazenda" TEXT,
  "qtd_pesagens" INTEGER,
  "tem_pesagem_21d" INTEGER,
  "tem_pesagem_28d" INTEGER,
  "tem_pesagem_35d" INTEGER,
  "tem_pesagem_42d" INTEGER,
  "elegivel_modelo" INTEGER,
  "categoria_amostragem" TEXT,
  "score_confianca_lote" REAL,
  "elegivel_rn11" INTEGER,
  "motivo_inelegibilidade" TEXT,
  "estrategia_predicao" TEXT,
  "score_confianca_fazenda" REAL
);

-- Table: variables
CREATE TABLE "variables" (
"data_alojamento" TEXT,
  "lote_composto" TEXT,
  "produtor" TEXT,
  "f01" INTEGER,
  "f02" INTEGER,
  "f03" INTEGER,
  "f04" INTEGER,
  "f05" INTEGER,
  "f06" INTEGER,
  "c05" INTEGER,
  "c06" INTEGER,
  "c11" INTEGER,
  "c12" INTEGER,
  "y03" INTEGER,
  "y04" INTEGER,
  "c15" INTEGER,
  "c16" TEXT,
  "c17" TEXT,
  "f07" INTEGER,
  "fazenda" TEXT
);

-- Table: constantes
CREATE TABLE "constantes" (
"fazenda" TEXT,
  "nucleo" REAL,
  "a01" REAL,
  "a02" REAL,
  "a03" REAL,
  "a08" REAL,
  "a09" REAL,
  "a10" REAL,
  "a11" REAL,
  "c07" REAL,
  "c08" REAL,
  "d01" REAL,
  "d02" REAL,
  "d03" REAL,
  "d04" REAL,
  "d05" REAL,
  "f15" REAL,
  "h01" REAL,
  "h02" REAL,
  "i02" REAL,
  "i03" REAL,
  "i04" REAL,
  "i05" REAL,
  "i06" REAL,
  "i07" REAL,
  "i08" REAL,
  "i09" REAL,
  "i10" REAL,
  "i11" REAL,
  "i12" REAL,
  "x01" REAL,
  "x02" REAL,
  "i01" REAL
);

-- Table: lote_rn12_digital_twins
CREATE TABLE "lote_rn12_digital_twins" (
"lote_composto" TEXT,
  "fazenda" TEXT,
  "score_confianca_lote" REAL,
  "categoria_amostragem" TEXT,
  "elegivel_rn11" INTEGER,
  "knn_pred_weight_k15" REAL,
  "knn_pred_weight_k30" REAL,
  "knn_neighbor_std_k15" REAL,
  "knn_dist_nearest" REAL
);

