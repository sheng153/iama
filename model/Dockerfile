# Dockerfile, Image, Container

FROM python:3.11-slim

WORKDIR /model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN yes | python -m nltk.downloader punkt wordnet omw-1.4

ENTRYPOINT ["python"]

CMD ["main.py"]
