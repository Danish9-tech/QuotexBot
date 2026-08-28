#!/bin/bash

echo "===================================================="
echo " Setting up Quotex Bot 24/7 Ubuntu Systemd Service"
echo "===================================================="

SERVICE_FILE="/etc/systemd/system/quotexbot.service"

# Kill old tmux sessions or standalone python processes
tmux kill-session -t mybot 2>/dev/null
pkill -f "python bot.py" 2>/dev/null

cat << 'EOF' > $SERVICE_FILE
[Unit]
Description=Quotex Trading Bot 24/7 Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/QuotexBot
ExecStart=/root/QuotexBot/.venv/bin/python /root/QuotexBot/bot.py
Restart=always
RestartSec=5
KillSignal=SIGINT
StandardOutput=append:/root/QuotexBot/bot.log
StandardError=append:/root/QuotexBot/bot.log

[Install]
WantedBy=multi-user.target
EOF

DASHBOARD_SERVICE_FILE="/etc/systemd/system/quotex-dashboard.service"

cat << 'EOF' > $DASHBOARD_SERVICE_FILE
[Unit]
Description=Quotex Bot Live Verification Dashboard (background runner)
After=network-online.target quotexbot.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/QuotexBot
ExecStart=/root/QuotexBot/.venv/bin/python /root/QuotexBot/dashboard_runner.py
Restart=always
RestartSec=5
KillSignal=SIGINT
StandardOutput=append:/root/QuotexBot/dashboard.log
StandardError=append:/root/QuotexBot/dashboard.log

[Install]
WantedBy=multi-user.target
EOF

WEB_SERVICE_FILE="/etc/systemd/system/quotex-web.service"

cat << 'EOF' > $WEB_SERVICE_FILE
[Unit]
Description=Quotex Bot Dashboard Web (gunicorn)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/QuotexBot
ExecStart=/root/QuotexBot/.venv/bin/gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --timeout 120
Restart=always
RestartSec=5
KillSignal=SIGINT
StandardOutput=append:/root/QuotexBot/web.log
StandardError=append:/root/QuotexBot/web.log

[Install]
WantedBy=multi-user.target
EOF

chmod 644 $SERVICE_FILE
chmod 644 $DASHBOARD_SERVICE_FILE
chmod 644 $WEB_SERVICE_FILE
systemctl daemon-reload
systemctl enable quotexbot.service
systemctl enable quotex-dashboard.service
systemctl enable quotex-web.service
systemctl restart quotexbot.service
systemctl restart quotex-dashboard.service
systemctl restart quotex-web.service

echo ""
echo "✅ Systemd Services Successfully Installed & Started!"
echo "----------------------------------------------------"
echo "Trading bot (bot.py):"
echo "  Status : systemctl status quotexbot"
echo "  Log    : tail -f ~/QuotexBot/bot.log"
echo "----------------------------------------------------"
echo "Dashboard runner (kill-switch + Telegram pings):"
echo "  Status : systemctl status quotex-dashboard"
echo "  Log    : tail -f ~/QuotexBot/dashboard.log"
echo "----------------------------------------------------"
echo "Dashboard web (HTML page on port 5000):"
echo "  Status : systemctl status quotex-web"
echo "  Log    : tail -f ~/QuotexBot/web.log"
echo "  URL    : http://<server-ip>:5000/dashboard"
echo "====================================================""
