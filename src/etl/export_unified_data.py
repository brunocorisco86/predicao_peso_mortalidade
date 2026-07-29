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

# Define paths
db_path = os.path.join('database', 'prediction_data.db')
output_dir = os.path.join('data', 'processed')
output_csv_path = os.path.join(output_dir, 'unified_data.csv')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

try:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # SQL query to create a unified table
    query = """
    SELECT
        emd.*,
        v.produtor,
        v.f01, v.f02, v.f03, v.f04, v.f05, v.f06, v.c05, v.c06, v.c11, v.c12, v.y03, v.y04, v.c15, v.c16, v.c17, v.f07,
        c.nucleo,
        c.a01, c.a02, c.a03, c.a08, c.a09, c.a10, c.a11, c.c07, c.c08, c.d01, c.d02, c.d03, c.d04, c.d05, c.f15, c.h01, c.h02, c.i02, c.i03, c.i04, c.i05, c.i06, c.i07, c.i08, c.i09, c.i10, c.i11, c.i12, c.x01, c.x02, c.i01
    FROM
        extracao_mtech_data emd
    LEFT JOIN
        variables v ON emd.lote_composto = v.lote_composto
    LEFT JOIN
        constantes c ON emd.fazenda = c.fazenda;
    """

    # Read data into a pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Export to CSV
    df.to_csv(output_csv_path, index=False)

    file_size_mb = os.path.getsize(output_csv_path) / (1024 * 1024)
    print(f"Successfully exported unified data to {output_csv_path}")
    print(f"Export summary: {len(df)} rows, {len(df.columns)} columns, {file_size_mb:.2f} MB")

except sqlite3.Error as e:
    print(f"SQLite error: {e}")
except pd.errors.DatabaseError as e:
    print(f"Pandas database error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Close the database connection
    if 'conn' in locals() and conn:
        conn.close()

