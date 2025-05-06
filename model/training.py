import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#lemmatization utiliza reglas del lenguaje para obtener la base o raiz de una palabra
lemmatizer =WordNetLemmatizer()
#importo los intents
intents = json.loads(open('intents.json').read())

nltk.download('punkt', halt_on_error=False)
nltk.download('wordnet')
nltk.download('omw-1.4')

#creo las listas
words=[]
classes = []
ignore_letters= ['?','!','.',',']

documents = [
    (
        [lemmatizer.lemmatize(w.lower()) 
         for w in nltk.word_tokenize(pattern)
         if w not in ignore_letters],
        intent['tag']
    )
    for intent in intents['intents']
    for pattern in intent['patterns']
]

words   = sorted({ w for doc, _ in documents for w in doc }) 
classes = sorted({ tag for _, tag in documents })

pickle.dump(words,   open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = [
    (
        [1 if w in doc else 0 for w in words],
        [1 if c == tag else 0 for c in classes]
    )
    for doc, tag in documents
]

#ahora divido en 2 variables
train_x, train_y = map(list, zip(*training))

#creo la red neuronal secuencial
model = Sequential()
#primera capa
model.add(Dense(128, input_shape=(len(train_x[0]),), name="inp_layer", activation='relu'))
model.add(Dropout(0.5, name="hidden_layer1"))
#segunda capa
model.add(Dense(64, name="hidden_layer2", activation='relu'))
model.add(Dropout(0.5, name="hidden_layer3"))
#capa de salida
model.add(Dense(len(train_y[0]), name="output_layer", activation='softmax'))

#creo el optimizador y compilo
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#se entrena el modelo y lo guardo
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save("chatbot_model.h5")
