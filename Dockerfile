# Dockerfile, Image, Container

FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/* \ 
    &&pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --ignore-installed blinker>=1.9.0 \
    && pip install --no-cache-dir -r requirements.txt 

COPY model/ ./model
COPY database/ ./database
COPY ejemplos/ ./ejemplos
COPY intents.json /app/intents.json

RUN yes | python -m nltk.downloader punkt wordnet omw-1.4

COPY .entrypoint .
RUN chmod +x .entrypoint

ENTRYPOINT ["./.entrypoint"]
CMD ["python3", "model/main.py"]
