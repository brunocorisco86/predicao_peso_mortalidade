-- SQLite Database Schema Export
-- Database: /home/brunoconter/Documentos/1_C.VALE/1 - ANALISES/10 - PESO DAS AVES/prediction_weight_mortality/database/prediction_data.db
-- Exported on: 2026-07-29 03:05:24.668957

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
  "f07" INTEGER
);

-- Table: constantes
CREATE TABLE "constantes" (
"fazenda" INTEGER,
  "nucleo" INTEGER,
  "a01" INTEGER,
  "a02" INTEGER,
  "a03" INTEGER,
  "a08" INTEGER,
  "a09" INTEGER,
  "a10" INTEGER,
  "a11" INTEGER,
  "c07" INTEGER,
  "c08" INTEGER,
  "d01" INTEGER,
  "d02" INTEGER,
  "d03" INTEGER,
  "d04" INTEGER,
  "d05" INTEGER,
  "f15" INTEGER,
  "h01" INTEGER,
  "h02" INTEGER,
  "i02" INTEGER,
  "i03" INTEGER,
  "i04" INTEGER,
  "i05" INTEGER,
  "i06" INTEGER,
  "i07" INTEGER,
  "i08" INTEGER,
  "i09" INTEGER,
  "i10" REAL,
  "i11" REAL,
  "i12" REAL,
  "x01" INTEGER,
  "x02" INTEGER,
  "i01" INTEGER
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
  "fazenda" REAL
);

