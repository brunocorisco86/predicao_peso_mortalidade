import sqlite3

db_path = r'C:\Users\user\code\Git\prediction_weight_mortality\database\prediction_data.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("Variables Table (first 5 rows):")
    cursor.execute('SELECT * FROM variables LIMIT 5')
    for row in cursor.fetchall():
        print(row)

    print("\nConstantes Table (first 5 rows):")
    cursor.execute('SELECT * FROM constantes LIMIT 5')
    for row in cursor.fetchall():
        print(row)

except sqlite3.Error as e:
    print(f"SQLite error: {e}")
finally:
    if 'conn' in locals() and conn:
        conn.close()
