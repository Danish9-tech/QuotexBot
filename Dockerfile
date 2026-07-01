FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium)
RUN playwright install chromium
RUN playwright install-deps

COPY . .

# Expose port for the keep-alive ping
EXPOSE 7860

# Run all three scripts: web server, bot, and pinger
CMD gunicorn app:app --bind 0.0.0.0:7860 & python bot.py & python ping.py
