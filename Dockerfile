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

# Shell form (not the ["uvicorn", ...] array form) so $PORT actually gets substituted.
# Render, Railway, Fly.io etc. inject PORT at runtime and expect the app to bind to it;
# ${PORT:-8000} falls back to 8000 for plain `docker run` where PORT isn't set.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
