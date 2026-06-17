FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Création du dossier
RUN mkdir -p /app/pdfs

# Copie permanente des PDFs dans l'image[5/8] RUN pip install --no-cache-dir -r requirements.txt                                                                          1136.6s
COPY pdfs/ /app/pdfs/

ENV PDF_FOLDER=/app/pdfs \
    MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]