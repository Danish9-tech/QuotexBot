import json
import time
from flask import Flask, request, jsonify

# Create a Flask application
app = Flask(__name__)

# Register the live verification dashboard blueprint
try:
    from dashboard.web import bp as dashboard_bp
    app.register_blueprint(dashboard_bp)
except Exception as _dash_err:
    # Don't fail the existing health endpoint if dashboard init errors
    import logging
    logging.getLogger("app").warning(f"dashboard blueprint not registered: {_dash_err}")

# Define a route for the homepage
@app.route('/')
def hello_world():
    return 'QuotexBot AI Execution & Signal Server is Active'

# Extension Signal Relay Endpoint
@app.route('/api/signal', methods=['POST'])
def receive_extension_signal():
    try:
        data = request.get_json(force=True)
        signal = data.get('signal')
        confidence = data.get('confidence', 0)
        timeframe = data.get('timeframe', '1m')
        reasoning = data.get('reasoning', '')
        
        print(f"[EXTENSION SIGNAL RECEIVED] {signal} | Confidence: {confidence}% | TF: {timeframe} | Reason: {reasoning}")
        
        return jsonify({
            'status': 'success',
            'message': f'Signal {signal} received with {confidence}% confidence',
            'signal': signal,
            'confidence': confidence
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# Zero-Key Extension Vision Analysis Endpoint
@app.route('/api/analyze', methods=['POST'])
def analyze_chart_endpoint():
    import os
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        data = request.get_json(force=True)
        base64_img = data.get('image', '')
        timeframe = data.get('timeframe', '1m')

        if not base64_img:
            return jsonify({'status': 'error', 'message': 'No image data provided'}), 400

        # System OpenRouter API Key from .env
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        
        prompt = f"""You are an elite quantitative binary options trader specializing in Quotex OTC algorithm breakdown & 1m/5m scalping.
Analyze this live chart screenshot for timeframe: {timeframe}.

ALGORITHMIC OTC CONFLUENCE SECRETS FOR HIGH WIN-RATES:

1. ROUND NUMBER & KEY S/R BOUNCES (High Win Rate):
   - Check price levels on the right axis. If price touches a round number (ending in .000, .500, .800) or major S/R low and forms a lower rejection wick -> CALL (90%-98% confidence).
   - If price touches a round number resistance high and forms an upper rejection wick -> PUT (90%-98% confidence).

2. CANDLE COLOR PATTERN SEQUENCES:
   - 3 consecutive RED candles touching lower Donchian/Bollinger level -> Expect Bullish Reversal CALL (88%-95% confidence).
   - 3 consecutive GREEN candles touching upper level -> Expect Bearish Reversal PUT (88%-95% confidence).

3. TREND CONTINUATION:
   - Strong full-body green candle breaking previous high -> CALL (85%-92% confidence).
   - Strong full-body red candle breaking previous low -> PUT (85%-92% confidence).

4. EXECUTION GUIDANCE:
   - If confidence >= 85%, specify 1-Step Martingale (MTG-1) recovery recommendation.

Respond ONLY with a raw JSON object (no markdown, no backticks):
{{
  "signal": "CALL" | "PUT" | "NO_TRADE",
  "confidence": 88,
  "timeframe": "{timeframe}",
  "reasoning": "1 concise sentence stating pattern/level rejection and MTG recommendation",
  "suggested_duration": "1m"
}}"""

        openrouter_models = [
            "minimax/minimax-m3:free",
            "openai/gpt-4o-mini"
        ]

        last_error = None
        result_text = None

        for model_name in openrouter_models:
            try:
                res = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{base64_img}"}
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 250
                    },
                    timeout=15
                )

                if res.status_code == 200:
                    result_text = res.json()['choices'][0]['message']['content']
                    break
                else:
                    last_error = f"Vision API error ({res.status_code}): {res.text}"
            except Exception as e:
                last_error = str(e)

        if not result_text:
            return jsonify({'status': 'error', 'message': last_error or 'All vision models failed'}), 500

        cleaned = result_text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(cleaned)
        
        return jsonify({
            'signal': parsed.get('signal', 'NO_TRADE').upper(),
            'confidence': int(parsed.get('confidence', 85)),
            'timeframe': parsed.get('timeframe', timeframe),
            'reasoning': parsed.get('reasoning', 'Chart pattern analyzed successfully.'),
            'timestamp': int(time.time() * 1000)
        }), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

