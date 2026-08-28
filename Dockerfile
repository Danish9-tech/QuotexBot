FROM python:3.12

# Install git (needed for pyquotex)
RUN apt-get update && apt-get install -y git

# Install playwright globally as root to install system dependencies
RUN pip install playwright
RUN playwright install-deps chromium

# Hugging Face requires running as a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install them as the user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the actual Chromium browser for the user
RUN playwright install chromium

# Copy the rest of the code
COPY --chown=user . $HOME/app

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

# Start everything. Make bot.py run in foreground so if it crashes, the container logs it.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:7860 & python ping.py & python bot.py & python dashboard_runner.py"]
