# Lean image for serving the attrition API. Only what's needed to run the API,
# not the notebook/Jupyter tooling used for exploration during development.
FROM python:3.11-slim

WORKDIR /app

# Install Python packages first (separate layer from the code below), so Docker
# reuses this layer on rebuilds unless requirements-api.txt actually changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what's needed to train + serve: the code and the training data.
COPY src/ src/
COPY api/ api/
COPY data/ data/

# Train the model INSIDE the image at build time, so the image is always
# self-contained and the model always matches the code that produced it -
# no separately-built .joblib file to keep in sync.
RUN python src/train.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
