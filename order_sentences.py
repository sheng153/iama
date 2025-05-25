import pandas as pd
import json

file = 'spa_sentences.tsv'

df = pd.read_csv(file, sep="\t", names=["id", "lang", "sentence"], encoding="utf-8")

df = df[df["lang"] == "spa"]

intents_keywords = {
    "saludo": ["hola", "buenos días", "buenas tardes", "qué onda", "hey"],
    "despedida": ["adiós", "chau", "nos vemos", "hasta luego"],
    "comida": ["tengo hambre", "comer", "comida", "me gusta comer"],
    "emociones": ["estoy feliz", "estoy triste", "me siento", "emocionado"]
}

intents_data = []

for tag, keywords in intents_keywords.items():
    patterns = []
    for kw in keywords:
        matched = df[df["sentence"].str.lower().str.contains(kw.lower(), na=False)]["sentence"].tolist()
        patterns.extend(matched)
    
    if patterns:
        intent = {
            "tag": tag,
            "patterns": list(set(patterns[:20])),  # limitar a 20 frases por intent
            "responses": [f"Esto es una respuesta para el intent {tag}."]
        }
        intents_data.append(intent)

output = {"intents": intents_data}

with open("intents_from_tatoeba.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("✅ Archivo JSON generado como intents_from_tatoeba.json")
