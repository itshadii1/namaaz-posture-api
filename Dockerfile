FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ml && \
    curl -L -o ml/body_language.pkl \
        https://github.com/itshadii1/namaaz-posture-api/raw/main/ml/body_language.pkl && \
    curl -L -o ml/pose_landmarker_lite.task \
        https://github.com/itshadii1/namaaz-posture-api/raw/main/ml/pose_landmarker_lite.task

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
