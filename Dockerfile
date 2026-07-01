FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Hugging Face requires running as a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . $HOME/app

EXPOSE 7860

# Start everything
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:7860 & python bot.py & python ping.py"]
