-- SQLite Database Schema Export
-- Database: database/prediction_data.db
-- Exported on: 2025-09-10 17:53:39.137159

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
  "id_usurio" TEXT
);

