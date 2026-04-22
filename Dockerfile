# HuggingFace Spaces — Emotion Pro
FROM python:3.11-slim

# HF Spaces runs as non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY --chown=user . .

# HuggingFace Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Init DB then start with gunicorn + eventlet
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:7860", "--timeout", "120", "main:app"]
