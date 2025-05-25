import random
import json
import numpy as np
import sqlite3

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model
from sentence_transformers import SentenceTransformer

DB_PATH = "/data/dev.db"
MODEL_PATH = "/data/chatbot_model.keras"
CLASSES_PATH = "/data/classes.json"

embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = load_model(MODEL_PATH)

with open(CLASSES_PATH, 'r') as f:
    classes = json.load(f)
 
lemmatizer = WordNetLemmatizer()
model = load_model('/data/chatbot_model.keras')

def download_nltk(): 
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word) for word in sentence_words] 

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    print(bag)
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose = 0)[0]
    max_index = np.argmax(res)
    return classes[max_index]

def get_responses_by_tag(tag, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT r.response
        FROM responses r
        JOIN intents i ON r.intent_id = i.id 
        WHERE i.tag = ?
    """, (tag, ))
    results = cur.fetchall()
    conn.close()
    return [r[0] for r in results]

def respuesta(message):
    vec = embedder.encode([message], convert_to_numpy = True)

    probs = model.predict(vec, verbose=0)[0]

    idx = probs.argmax()
    tag = classes[idx]

    options = get_responses_by_tag(tag)
    return random.choice(options) if options else "Perdón, no entendí"

if __name__ == "__main__":
    while True:
        message = input("Tu: ")
        print("Bot: ", respuesta(message))
