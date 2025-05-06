# Dockerfile, Image, Container

FROM python:3.13-slim

WORKDIR /model

RUN apt-get update && apt-get install -y dnsutils iputils-ping
RUN ping -c 3 8.8.8.8
RUN ping -c 3 pypi.org

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m nltk.downloader punkt wordnet omw-1.4

ENTRYPOINT ["python"]

CMD ["main.py"]
