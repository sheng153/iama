import sqlite3
import json
import os

DB_PATH = "/data/dev.db"
EXAMPLES_DIR = "./ejemplos"

def populate_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for fname in os.listdir(EXAMPLES_DIR):
        if not fname.endswith(".json"):
            continue

        print(f"Procesando archivo: {fname}")

        with open(os.path.join(EXAMPLES_DIR, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        for intent in data["intents"]:
            tag = intent["tag"]

            cur.execute("INSERT INTO intents (tag) VALUES (?)", (tag,))
            intent_id = cur.lastrowid

            for pattern in intent["patterns"]:
                cur.execute("INSERT INTO patterns (intent_id, pattern) VALUES (?, ?)", (intent_id, pattern))

            for response in intent["responses"]:
                cur.execute("INSERT INTO responses (intent_id, response) VALUES (?, ?)", (intent_id, response))

    
    conn.commit()
    conn.close()
    print("Base de datos poblada con éxito.")

if __name__ == "__main__":
    populate_db()
