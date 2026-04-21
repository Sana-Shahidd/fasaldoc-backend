FROM python:3.10-slim

WORKDIR /app

# System deps for opencv (used by gradcam) and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Copy model artifacts (built in Phase 2)
# model.tflite, model.h5, labels.json, treatments.json
# These should be in fasaldoc-backend/ before building
# (copy them from FasalDoc root: cp ../model.tflite ../labels.json ../treatments.json .)

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
