import sqlite3
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings
from src.utils.logger import logger, setup_logging

def export_unified_data():
    db_path = settings.DATABASE_PATH
    output_dir = settings.PROCESSED_DIR
    output_csv_path = settings.UNIFIED_CSV_PATH

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Connecting to database at {db_path} to export unified data...")

    try:
        conn = sqlite3.connect(db_path)

        query = """
        SELECT
            emd.*,
            v.produtor,
            v.f01, v.f02, v.f03, v.f04, v.f05, v.f06, v.c05, v.c06, v.c11, v.c12, v.y03, v.y04, v.c15, v.c16, v.c17, v.f07,
            c.nucleo,
            c.a01, c.a02, c.a03, c.a08, c.a09, c.a10, c.a11, c.c07, c.c08, c.d01, c.d02, c.d03, c.d04, c.d05, c.f15, c.h01, c.h02, c.i02, c.i03, c.i04, c.i05, c.i06, c.i07, c.i08, c.i09, c.i10, c.i11, c.i12, c.x01, c.x02, c.i01,
            pa.idade_abate,
            pa.data_producao AS data_producao_abate,
            pa.peso_medio_abate_kg,
            pa.peso_abate_g,
            pa.gmd_abate,
            sc.score_confianca_lote,
            sc.categoria_amostragem,
            sc.elegivel_rn11,
            sc.motivo_inelegibilidade,
            sc.estrategia_predicao,
            rn12.knn_pred_weight_k15,
            rn12.knn_pred_weight_k30,
            rn12.knn_neighbor_std_k15,
            rn12.knn_dist_nearest
        FROM
            extracao_mtech_data emd
        LEFT JOIN
            variables v ON emd.lote_composto = v.lote_composto
        LEFT JOIN
            constantes c ON emd.fazenda = c.fazenda
        LEFT JOIN
            peso_abate pa ON emd.lote_composto = pa.lote_composto
        LEFT JOIN
            lote_sampling_confidence sc ON emd.lote_composto = sc.lote_composto
        LEFT JOIN
            lote_rn12_digital_twins rn12 ON emd.lote_composto = rn12.lote_composto;
        """

        df = pd.read_sql_query(query, conn)
        df.to_csv(output_csv_path, index=False)

        file_size_mb = os.path.getsize(output_csv_path) / (1024 * 1024)
        logger.info(f"Successfully exported unified data to {output_csv_path}")
        logger.info(f"Export summary: {len(df)} rows, {len(df.columns)} columns, {file_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Error exporting unified data: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    setup_logging()
    export_unified_data()
