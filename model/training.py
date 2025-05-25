import random
import json
import pickle
import numpy as np
import nltk
import sqlite3
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from sentence_transformers import SentenceTransformer

IGNORE_LETTERS = ['?','!','.',',']

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_nltk():
    nltk.download('punkt', halt_on_error=False)
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')

def get_training_data(db_path="/data/dev.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT i.tag, p.pattern
        FROM patterns p
        JOIN intents i ON p.intent_id = i.id
    """)
    rows = cur.fetchall()
    conn.close()

    lemmatizer = WordNetLemmatizer()
    documents = [
        (
            [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(pattern) if w not in IGNORE_LETTERS],
            tag
        )
        for tag, pattern in rows
    ]

    return documents, lemmatizer

def set_embeddings(docs):
    sentences = [" ".join(doc) for doc, _ in docs]
    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    classes = sorted({tag for _, tag in docs})
    with open('/data/classes.json', 'w') as f:
        json.dump(classes, f)

    labels = [
        [1 if c==tag else 0 for c in classes]
        for _,tag in docs
    ]

    return list(zip(embeddings, labels)), classes

def instantiate_model(train_x, train_y):
    model = Sequential()

    model.add(Dense(128, input_shape=(len(train_x[0]),), name="inp_layer", activation='relu'))
    model.add(Dropout(0.5, name="hidden_layer1"))

    model.add(Dense(64, name="hidden_layer2", activation='relu'))
    model.add(Dropout(0.5, name="hidden_layer3"))

    model.add(Dense(len(train_y[0]), name="output_layer", activation='softmax'))

    sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

    return model

def log_training_result(acc, db_path="/data/dev.db"):
    conn = sqlite3.connect(db_path)
    curr = conn.cursor()
    curr.execute("INSERT INTO training_logs (accuracy) VALUES (?)", (acc,))
    conn.commit()
    conn.close()

if __name__ == '__main__': 
    initialize_nltk()

    documents, lemmatizer = get_training_data() 

    embeddings, classes = set_embeddings(documents)

    train_x, train_y = map(list, zip(*embeddings))

    model = instantiate_model(train_x, train_y)

    model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
    score = model.evaluate(np.array(train_x), np.array(train_y), verbose=0)
    log_training_result(score[1])
    model.save('/data/chatbot_model.keras')
