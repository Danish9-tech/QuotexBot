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

chmod 644 $SERVICE_FILE
systemctl daemon-reload
systemctl enable quotexbot.service
systemctl restart quotexbot.service

echo ""
echo "✅ Systemd Service Successfully Installed & Started!"
echo "----------------------------------------------------"
echo "Commands to manage your 24/7 Bot:"
echo "  Check Status : systemctl status quotexbot"
echo "  View Live Log: tail -f ~/QuotexBot/bot.log"
echo "  Restart Bot  : systemctl restart quotexbot"
echo "  Stop Bot     : systemctl stop quotexbot"
echo "===================================================="
