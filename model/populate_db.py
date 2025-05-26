import sqlite3
import json
import os

DB_PATH = "/data/dev.db"
EXAMPLES_DIR = "./ejemplos"

def populate_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          role        TEXT    NOT NULL,
          content     TEXT    NOT NULL,
          timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    for fname in os.listdir(EXAMPLES_DIR):
        if not fname.endswith(".json"):
            continue

        print(f"Procesando archivo: {fname}")

        with open(os.path.join(EXAMPLES_DIR, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        for intent in data.get("intents", []):
            for pattern in intent.get("patterns", []):
                cur.execute("INSERT INTO conversation (role, content) VALUES (?, ?)", ("user", pattern))

            for response in intent.get("responses", []):
                cur.execute("INSERT INTO conversation (role, content) VALUES (?, ?)", ("assistant", response))

    
    conn.commit()
    conn.close()
    print("Base de datos poblada con éxito.")

if __name__ == "__main__":
    populate_db()
