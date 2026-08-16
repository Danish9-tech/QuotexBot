import os
import json
import logging
import pandas as pd
import aiohttp
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables just in case
load_dotenv()

logger = logging.getLogger(__name__)

# Global state for key rotation
OPENROUTER_KEYS_LIST = []
CURRENT_KEY_INDEX = 0

# Initialize Groq client
try:
    client = AsyncGroq()
except Exception as e:
    logger.error(f"Failed to initialize Groq client. Make sure GROQ_API_KEY is in .env: {e}")
    client = None

def get_openrouter_keys():
    global OPENROUTER_KEYS_LIST
    if not OPENROUTER_KEYS_LIST:
        keys_str = os.getenv("OPENROUTER_KEYS")
        if keys_str:
            OPENROUTER_KEYS_LIST = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            single_key = os.getenv("OPENROUTER_API_KEY")
            if single_key:
                OPENROUTER_KEYS_LIST = [single_key.strip()]
    return OPENROUTER_KEYS_LIST

async def get_groq_trading_signal(candles: list, asset_name: str, candle_size: int = 60, recent_trades: list = None) -> dict:
    """
    Takes a list of candle dictionaries, calculates indicators, 
    injects past trade memory, and asks Groq or OpenRouter for a trading decision.
    """
    global CURRENT_KEY_INDEX
    
    if not candles or len(candles) < 50:
        logger.warning(f"[{asset_name}] Not enough candles for AI analysis (need at least 50).")
        return {"signal": "doji", "confidence": 0, "reason": "Not enough data"}

    try:
        # Convert candles to DataFrame
        df = pd.DataFrame(candles)
        
        # Calculate EMA 50
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate RSI 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands 20
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['STD_20'] = df['close'].rolling(window=20).std()
        df['BBU_20_2.0'] = df['SMA_20'] + (df['STD_20'] * 2)
        df['BBL_20_2.0'] = df['SMA_20'] - (df['STD_20'] * 2)

        df = df.dropna()
        if df.empty:
            return {"signal": "doji", "confidence": 0, "reason": "Not enough data after TA calculation"}
            
        current_rsi = df['RSI_14'].iloc[-1]
        
        recent_data = df.tail(10).to_dict(orient="records")
        market_context = json.dumps(recent_data, indent=2)
        
        memory_context = "No recent trades for this asset."
        if recent_trades:
            memory_context = "RECENT TRADES MEMORY (Learn from these past mistakes and successes):\n"
            for t in recent_trades:
                memory_context += f"- Asset: {t.get('asset')}, Signal: {t.get('ai_signal')}, Reason: {t.get('ai_reason')}, Result: {t.get('result')}, Profit: {t.get('profit')}\n"

        prompt = f"""You are an elite quantitative forex trader. I am giving you the latest OHLC data ({candle_size}-second candles) and technical indicator values for {asset_name}.

MARKET DATA:
{market_context}

{memory_context}

YOUR TRADING RULES FOR OTC MARKETS (PATIENCE IS PROFIT):
1. OTC UNSTOPPABLE MOMENTUM: Quotex OTC markets are controlled by broker algorithms. When RSI hits extreme levels, the broker pushes the trend forever to liquidate reversal traders. You must NEVER trade against strong momentum.
   - If RSI_14 is > 70, the bullish trend is UNSTOPPABLE. Your ONLY options are "call" (trend continuation) or "doji" (wait). You are strictly forbidden from guessing a reversal ("put").
   - If RSI_14 is < 30, the bearish trend is UNSTOPPABLE. Your ONLY options are "put" (trend continuation) or "doji" (wait). You are strictly forbidden from guessing a reversal ("call").
2. THE CHOP ZONE (EXPANDED): If RSI_14 is between 35 and 65, the market has NO clear direction. Retail traders lose 100% of their money trading in this zone. You MUST signal "doji". There are ZERO exceptions to this rule.
3. CONFIDENCE RULES: You must ONLY set confidence above 80 when ALL of the following are true: (a) RSI is in an extreme zone (above 70 or below 30), (b) Price action confirms the direction with strong candles, (c) The setup does not match any recent losing trade. If even ONE condition is missing, your confidence MUST be below 80.
4. PATIENCE OVER PROFIT: It is better to skip 100 trades and miss opportunities than to take 1 bad trade and lose money. When in doubt, ALWAYS signal "doji". The best traders in the world only trade 2-3 times per day.
5. If your Recent Trades Memory shows you lost a trade recently with a similar setup, signal "doji" immediately.
6. If there is a strong momentum setup that respects ALL rules above, signal "call" or "put" with confidence 80+.

You must respond ONLY with a valid JSON object in this exact format. Do not include any markdown formatting or extra text:
{{"signal": "call" | "put" | "doji", "confidence": 0-100, "reason": "Brief explanation of the setup"}}
"""

        messages = [
            {"role": "system", "content": "You are a highly logical AI trading assistant that strictly outputs JSON."},
            {"role": "user", "content": prompt}
        ]

        # Use Custom API (like Bynara or OpenRouter) if keys are present
        keys = get_openrouter_keys()
        
        if keys:
            attempts = 0
            # Allow user to specify custom base URL and model in .env
            api_url = os.getenv("AI_API_URL", "https://openrouter.ai/api/v1/chat/completions")
            ai_model = os.getenv("AI_MODEL", "openrouter/free")
            
            while attempts < len(keys):
                current_key = keys[CURRENT_KEY_INDEX]
                logger.info(f"[{asset_name}] Using Custom AI API ({ai_model}) (Key #{CURRENT_KEY_INDEX + 1}/{len(keys)})...")
                
                headers = {
                    "Authorization": f"Bearer {current_key}",
                    "HTTP-Referer": "https://github.com/Danish9-tech/QuotexBot",
                    "X-Title": "Quotex AI Bot",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": ai_model,
                    "messages": messages,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(api_url, headers=headers, json=payload, timeout=20.0) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            result_text = data['choices'][0]['message']['content'].strip()
                            if result_text.startswith("```json"):
                                result_text = result_text[7:]
                            if result_text.endswith("```"):
                                result_text = result_text[:-3]
                            result_json = json.loads(result_text.strip())
                            
                            # PROGRAMMATIC OVERRIDE FOR OTC MOMENTUM
                            if current_rsi > 70 and result_json.get("signal") == "put":
                                result_json["signal"] = "doji"
                                result_json["reason"] = f"OVERRIDE: AI hallucinated a reversal put. RSI is {current_rsi:.2f} (Extreme Momentum UP). Forced skip."
                            elif current_rsi < 30 and result_json.get("signal") == "call":
                                result_json["signal"] = "doji"
                                result_json["reason"] = f"OVERRIDE: AI hallucinated a reversal call. RSI is {current_rsi:.2f} (Extreme Momentum DOWN). Forced skip."
                            elif 35 <= current_rsi <= 65 and result_json.get("signal") in ["call", "put"]:
                                result_json["signal"] = "doji"
                                result_json["confidence"] = 0
                                result_json["reason"] = f"OVERRIDE: Market is chopping (RSI {current_rsi:.2f}). No clear direction. Forced skip."
                            
                            logger.info(f"[{asset_name}] AI Prediction: {result_json}")
                            return result_json
                        elif resp.status in [429, 402]:
                            error_text = await resp.text()
                            logger.warning(f"[{asset_name}] API Key #{CURRENT_KEY_INDEX + 1} hit rate limit (429/402). Rotating to next key...")
                            CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(keys)
                            attempts += 1
                        else:
                            error_text = await resp.text()
                            raise Exception(f"AI API Error: {resp.status} - {error_text}")
            
            logger.error(f"[{asset_name}] ALL API keys are currently rate-limited! Waiting for reset.")
            return {"signal": "doji", "confidence": 0, "reason": "All API keys exhausted. Waiting for daily reset."}

        # Fallback to Groq if no custom keys
        if not client:
            return {"signal": "doji", "confidence": 0, "reason": "No AI client initialized (Add OPENROUTER_KEYS or GROQ_API_KEY)"}
            
        logger.info(f"[{asset_name}] Using Groq AI (Mixtral) for analysis...")
        response = await client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        # PROGRAMMATIC OVERRIDE FOR OTC MOMENTUM
        if current_rsi > 70 and result_json.get("signal") == "put":
            result_json["signal"] = "doji"
            result_json["reason"] = f"OVERRIDE: AI hallucinated a reversal put. RSI is {current_rsi:.2f} (Extreme Momentum UP). Forced skip."
        elif current_rsi < 30 and result_json.get("signal") == "call":
            result_json["signal"] = "doji"
            result_json["reason"] = f"OVERRIDE: AI hallucinated a reversal call. RSI is {current_rsi:.2f} (Extreme Momentum DOWN). Forced skip."
        elif 35 <= current_rsi <= 65 and result_json.get("signal") in ["call", "put"]:
            result_json["signal"] = "doji"
            result_json["confidence"] = 0
            result_json["reason"] = f"OVERRIDE: Market is chopping (RSI {current_rsi:.2f}). No clear direction. Forced skip."
        
        logger.info(f"[{asset_name}] Groq AI Prediction: {result_json}")
        return result_json

    except Exception as e:
        logger.error(f"[{asset_name}] Error during AI analysis: {e}", exc_info=True)
        return {"signal": "doji", "confidence": 0, "reason": f"AI Error: {str(e)}"}
