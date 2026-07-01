FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Playwright image already has 'pwuser' as UID 1000
USER pwuser
ENV HOME=/home/pwuser \
    PATH=/home/pwuser/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=pwuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=pwuser . $HOME/app

EXPOSE 7860

# Start everything
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:7860 & python bot.py & python ping.py"]
