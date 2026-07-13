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

# Initialize Groq client
# API key is automatically picked up from os.environ["GROQ_API_KEY"]
try:
    client = AsyncGroq()
except Exception as e:
    logger.error(f"Failed to initialize Groq client. Make sure GROQ_API_KEY is in .env: {e}")
    client = None

async def get_groq_trading_signal(candles: list, asset_name: str, candle_size: int = 60, recent_trades: list = None) -> dict:
    """
    Takes a list of candle dictionaries, calculates indicators, 
    injects past trade memory, and asks Groq or OpenRouter for a trading decision.
    """
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

        # Drop NaN values (first 50 rows will have NaNs because of EMA_50)
        df = df.dropna()
        if df.empty:
            return {"signal": "doji", "confidence": 0, "reason": "Not enough data after TA calculation"}
            
        # Get the last 10 candles for the AI to analyze the immediate context
        recent_data = df.tail(10).to_dict(orient="records")
        
        # Format for the prompt
        market_context = json.dumps(recent_data, indent=2)
        
        # Build memory context from recent trades
        memory_context = "No recent trades for this asset."
        if recent_trades:
            memory_context = "RECENT TRADES MEMORY (Learn from these past mistakes and successes):\n"
            for t in recent_trades:
                memory_context += f"- Asset: {t.get('asset')}, Signal: {t.get('ai_signal')}, Reason: {t.get('ai_reason')}, Result: {t.get('result')}, Profit: {t.get('profit')}\n"

        prompt = f"""You are an elite quantitative forex trader. I am giving you the latest OHLC data ({candle_size}-second candles) and technical indicator values for {asset_name}.

MARKET DATA:
{market_context}

{memory_context}

YOUR TRADING RULES:
1. Identify if the current trend is bullish (close > EMA_50) or bearish (close < EMA_50).
2. Look for reversals if RSI_14 is > 60 (Overbought) or < 40 (Oversold), OR if price touches the outer Bollinger Bands (BBU_20_2.0 or BBL_20_2.0).
3. Look for strong candlestick patterns (Engulfing, Pin bar) on the very last candle.
4. If your Recent Trades Memory shows you lost a trade recently with a specific setup, DO NOT repeat that mistake. Adjust your strategy.
5. If there is no clear setup, signal "doji" (which means wait).
6. If there is a strong buy setup, signal "call".
7. If there is a strong sell setup, signal "put".

You must respond ONLY with a valid JSON object in this exact format. Do not include any markdown formatting or extra text:
{{"signal": "call" | "put" | "doji", "confidence": 0-100, "reason": "Brief explanation of the setup"}}
"""

        messages = [
            {
                "role": "system",
                "content": "You are a highly logical AI trading assistant that strictly outputs JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Use OpenRouter if API key is present
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            logger.info(f"[{asset_name}] Using OpenRouter AI (DeepSeek) for analysis...")
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "HTTP-Referer": "https://github.com/Danish9-tech/QuotexBot", # Optional
                "X-Title": "Quotex AI Bot", # Optional
                "Content-Type": "application/json"
            }
            payload = {
                "model": "openrouter/free",
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result_text = data['choices'][0]['message']['content']
                        # Sometimes OpenRouter DeepSeek adds markdown block backticks despite json_object mode
                        result_text = result_text.strip()
                        if result_text.startswith("```json"):
                            result_text = result_text[7:]
                        if result_text.endswith("```"):
                            result_text = result_text[:-3]
                        result_json = json.loads(result_text.strip())
                        logger.info(f"[{asset_name}] OpenRouter AI Prediction: {result_json}")
                        return result_json
                    else:
                        error_text = await resp.text()
                        raise Exception(f"OpenRouter API Error: {resp.status} - {error_text}")

        # Fallback to Groq if OpenRouter is not configured
        if not client:
            return {"signal": "doji", "confidence": 0, "reason": "No AI client initialized (Add OPENROUTER_API_KEY or GROQ_API_KEY)"}
            
        logger.info(f"[{asset_name}] Using Groq AI (Mixtral) for analysis...")
        response = await client.chat.completions.create(
            messages=messages,
            model="mixtral-8x7b-32768",
            temperature=0.1,  # Low temperature for highly logical/consistent answers
            max_tokens=200,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        logger.info(f"[{asset_name}] Groq AI Prediction: {result_json}")
        return result_json

    except Exception as e:
        logger.error(f"[{asset_name}] Error during AI analysis: {e}", exc_info=True)
        return {"signal": "doji", "confidence": 0, "reason": f"AI Error: {str(e)}"}
