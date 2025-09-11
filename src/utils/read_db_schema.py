import sqlite3

db_path = r'C:\Users\user\code\Git\prediction_weight_mortality\database\prediction_data.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in the database:")
    for table_name in tables:
        print(f"- {table_name[0]}")
        cursor.execute(f"PRAGMA table_info({table_name[0]});")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]}) {'NOT NULL' if col[3] else ''} {'PK' if col[5] else ''}")

except sqlite3.Error as e:
    print(f"SQLite error: {e}")
finally:
    if 'conn' in locals() and conn:
        conn.close()
