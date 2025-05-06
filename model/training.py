import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

IGNORE_LETTERS = ['?','!','.',',']

def initialize_nltk():
    nltk.download('punkt', halt_on_error=False)
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def initialize_intents():
    return (json.loads(open("intents.json").read()), WordNetLemmatizer())

def set_documents(intents, lemmatizer) -> List[Tuple[List[str], str]]:
    return [
        (
            [lemmatizer.lemmatize(w.lower()) 
                for w in nltk.word_tokenize(pattern)
                if w not in IGNORE_LETTERS],
            intent['tag']
        )
        for intent in intents['intents']
        for pattern in intent['patterns']
    ]

def set_BoW(docs):
    words   = sorted({ w for doc, _ in documents for w in doc }) 
    classes = sorted({ tag for _, tag in documents })

    pickle.dump(words,   open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return [
    (
        [1 if w in doc else 0 for w in words],
        [1 if c == tag else 0 for c in classes]
    )
    for doc, tag in documents
    ]

def instantiate_model(train_x, train_y):
    model = Sequential()

    model.add(Dense(128, input_shape=(len(train_x[0]),), name="inp_layer", activation='relu'))
    model.add(Dropout(0.5, name="hidden_layer1"))

    model.add(Dense(64, name="hidden_layer2", activation='relu'))
    model.add(Dropout(0.5, name="hidden_layer3"))

    model.add(Dense(len(train_y[0]), name="output_layer", activation='softmax'))

    sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

    return model

if __name__ == '__main__': 
    initialize_nltk()

    intents, lemmatizer = initialize_intents() 

    documents = set_documents(intents, lemmatizer)

    train_x, train_y = map(list, zip(*set_BoW(documents)))

    model = instantiate_model(train_x, train_y)

    model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
    model.save("chatbot_model.h5")
