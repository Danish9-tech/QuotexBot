# DigitalOcean 24/7 Quotex Bot Cheat Sheet

This document contains every command and trick we used to deploy the Quotex Trading Bot to the cloud for 24/7 operation, bypassing Cloudflare's strict anti-bot security. Keep this file safe for future reference!

---

## 1. Entering the Bot & Activating Python
Every time you open your DigitalOcean Web Console, you start at the root directory (`~#`). You must always navigate to your bot folder and activate the Python environment before doing anything.

```bash
cd QuotexBot
source .venv/bin/activate
```

---

## 1.1 Recommended 24/7 Linux Systemd Daemon Mode (Bulletproof Auto-Restart)
This is the recommended industry-standard way to run the bot 24/7 on Ubuntu. It runs completely in the background without needing `tmux` or active SSH terminals, automatically restarts if the server reboots, and auto-heals if network timeouts occur.

**1-Click Installation Command:**
```bash
cd ~/QuotexBot && git pull && chmod +x setup_systemd.sh && ./setup_systemd.sh
```

**Essential Systemd Management Commands:**
* **Check Live Status:**
  ```bash
  systemctl status quotexbot
  ```
* **View Real-Time Live Logs:**
  ```bash
  tail -f ~/QuotexBot/bot.log
  ```
* **Restart Bot:**
  ```bash
  systemctl restart quotexbot
  ```
* **Stop Bot:**
  ```bash
  systemctl stop quotexbot
  ```

---

## 2. Managing 24/7 Background Sessions (tmux)
If you just run the bot normally, it will die the moment you close your browser. `tmux` creates a "virtual screen" that stays alive forever.

**Create a new background screen:**
```bash
tmux new -s mybot
```

**Detach from the screen (Leave it running in the background):**
- Press and hold **`Ctrl`**, tap **`B`**.
- Let go of both keys.
- Press **`D`**.
*(Or simply click the 'X' to close the DigitalOcean website tab!)*

**Return to the background screen later:**
```bash
tmux attach -t mybot
```

**Kill the background screen (Force stop everything inside it):**
```bash
tmux kill-session -t mybot
```

---

## 3. The "Mega Command" (Instant Fixes & Updates)
If you ever push new code to GitHub, or if your bot gets temporarily banned by Telegram for spamming, use this Mega Command inside your QuotexBot folder to instantly update the code, bypass IP blocks, delete corrupted sessions, and start a fresh background screen:

```bash
git pull && sed -i 's/API_ID=2040/API_ID=17349/g' .env && sed -i 's/API_HASH=b18441a1ff607e10a989891a5462e627/API_HASH=344583e45741c457fe1862106095a5eb/g' .env && sed -i 's/QUOTEX_HOST="market-qx.trade"/QUOTEX_HOST="qxbroker.com"/g' .env && rm -f *.session session.json && tmux new -s mybot
```

---

## 3.1 Bypassing Cloudflare HTTP 403 WebSocket Blocks with Cloudflare WARP
If Cloudflare blocks DigitalOcean's IP address on WebSocket connections (`HTTP 403`), install Cloudflare's free WARP tunnel to route outgoing VPS traffic through Cloudflare's trusted residential edge network.

**1-Click Installation & Activation:**
```bash
curl -fsSL https://pkg.cloudflareclient.com/pubkey.gpg | gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg && echo "deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/cloudflare-client.list && apt-get update && apt-get install cloudflare-warp -y && warp-cli registration new && warp-cli connect
```

---

## 4. Force Killing Hidden Bots
If you get `KeyError: 0` or your bot is stuck in a weird loop, it usually means there is a "hidden" ghost bot still running in the background fighting your new bot. Run this command to brutally assassinate all hidden Python bots:

```bash
pkill -f python
```

---

## 5. Starting the Bot (Native 2FA Login)
Quotex uses strict Cloudflare protection. By using the `qxbroker.com` mirror, we can bypass Cloudflare on DigitalOcean natively!

1. Make sure you are inside your `tmux` background session.
2. Run the bot:
   ```bash
   python bot.py
   ```
3. **DO NOT** type `/start` in Telegram. 
4. Wait for the bot to automatically message you: `"Quotex requires a PIN"`.
5. Check your email for the 6-digit Quotex PIN.
6. Reply to the bot's message in Telegram with those 6 digits.

The bot will securely log in, automatically save its session, and run flawlessly 24/7!

---

## 6. Top Asset Pairs to Add
When adding assets in Telegram, format them like `ASSETNAME,AMOUNT,PAYOUT`. 
Example: `EURUSD,1,60`

**Best Major Pairs (High Liquidity):**
- `EURUSD,1,60`
- `GBPUSD,1,60`
- `USDJPY,1,60`
- `AUDUSD,1,60`
- `USDCAD,1,60`

*(Note: The bot automatically searches for `_otc` versions of these pairs on weekends!)*
