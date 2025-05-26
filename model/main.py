import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

DB_PATH = "/data/dev.db"
MODEL_PATH = "/data/phi2-chatbot"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
chat = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=256, do_sample=true, temperature=.7)

def log_conversation(role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO conversation (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def respuesta(user_input):
    log_conversation("user", user_input)

    prompt = f"user: {user_input}\nassistant: "
    out = chat(prompt, max_new_tokens=128)[0]["generated-text"]

    reply = out.split("assistant:")[-1].strip()

    log_conversation("assistant", reply)
    return reply

if __name__ == "__main__":
    while True:
        msg = input("Tu: ")
        print("Bot: ", respuesta(msg))
