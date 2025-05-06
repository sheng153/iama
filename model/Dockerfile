# Dockerfile, Image, Container

FROM python:3.13-slim

WORKDIR /model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m nltk.downloader punkt wordnet omw-1.4

ENTRYPOINT ["python"]

CMD ["main.py"]
