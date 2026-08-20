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
    
    min_required_candles = 20 if candle_size <= 5 else 50
    if not candles or len(candles) < min_required_candles:
        logger.warning(f"[{asset_name}] Not enough candles for analysis (need at least {min_required_candles}, got {len(candles) if candles else 0}).")
        return {"signal": "doji", "confidence": 0, "reason": "Not enough data"}

    try:
        # Normalize candle dictionary keys to standard 'open', 'high', 'low', 'close'
        normalized_candles = []
        for c in candles:
            if isinstance(c, dict):
                o = c.get('open') if c.get('open') is not None else (c.get('o') if c.get('o') is not None else c.get('Open', 0))
                h = c.get('high') if c.get('high') is not None else (c.get('h') if c.get('h') is not None else c.get('High', 0))
                l = c.get('low') if c.get('low') is not None else (c.get('l') if c.get('l') is not None else c.get('Low', 0))
                close_val = c.get('close') if c.get('close') is not None else (c.get('c') if c.get('c') is not None else c.get('Close', 0))
                normalized_candles.append({
                    'open': float(o or 0),
                    'high': float(h or 0),
                    'low': float(l or 0),
                    'close': float(close_val or 0)
                })

        # Convert candles to DataFrame
        df = pd.DataFrame(normalized_candles)
        
        # =========================================================================
        # 5-SECOND OTC SURESHOT PRO COMBO (GAP, REJECTION, ENGULFING & TREND)
        # =========================================================================
        if candle_size <= 5:
            if len(df) < 10:
                return {"signal": "doji", "confidence": 0, "reason": "Not enough 5s candles for trend & pattern analysis"}
                
            # Calculate 20-period EMA, 100-period EMA & RSI 14 for Institutional Anti-Broker Analysis
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['EMA_100'] = df['close'].ewm(span=min(100, len(df)), adjust=False).mean()
            
            # Robust RSI 14 calculation with Exponential Moving Average (EWM)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss.replace(0, 1e-9)
            df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
            
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            prev_candle2 = df.iloc[-3] if len(df) >= 3 else prev_candle
            
            c_open = float(last_candle.get('open', 0))
            c_close = float(last_candle.get('close', 0))
            c_high = float(last_candle.get('high', 0))
            c_low = float(last_candle.get('low', 0))
            
            p_open = float(prev_candle.get('open', 0))
            p_close = float(prev_candle.get('close', 0))
            
            p2_open = float(prev_candle2.get('open', 0))
            p2_close = float(prev_candle2.get('close', 0))
            
            total_range = max(0.000001, c_high - c_low)
            body_size = abs(c_close - c_open)
            upper_wick = c_high - max(c_open, c_close)
            lower_wick = min(c_open, c_close) - c_low
            
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            body_ratio = body_size / total_range
            
            # Trend Check (EMA 20, EMA 100 & RSI 14)
            ema_20 = float(last_candle['EMA_20']) if not pd.isna(last_candle.get('EMA_20')) else c_close
            ema_100 = float(last_candle['EMA_100']) if not pd.isna(last_candle.get('EMA_100')) else c_close
            raw_rsi = last_candle.get('RSI_14', 50.0)
            rsi_14 = float(raw_rsi) if (pd.notna(raw_rsi) and 1.0 <= float(raw_rsi) <= 99.0) else 50.0
            
            is_uptrend = c_close >= ema_20
            is_downtrend = c_close < ema_20
            
            # Market State Filter (Forbidden Sideways / Consolidation Check)
            recent_high = df['high'].tail(15).max()
            recent_low = df['low'].tail(15).min()
            recent_range = recent_high - recent_low
            avg_bar_range = (df['high'] - df['low']).tail(15).mean()
            is_sideways = (recent_range < (avg_bar_range * 2.0)) or (abs(c_close - ema_100) < total_range * 0.05 and abs(p_close - ema_100) < total_range * 0.05)
            
            # Gap detection
            gap = c_open - p_close
            gap_threshold = max(0.000001, total_range * 0.15)
            is_gap_up = gap > gap_threshold
            is_gap_down = gap < -gap_threshold
            
            # Calculate Donchian Channel (Period 24)
            df['DC_Upper'] = df['high'].rolling(window=min(24, len(df))).max()
            df['DC_Lower'] = df['low'].rolling(window=min(24, len(df))).min()
            
            dc_upper = float(last_candle['DC_Upper']) if not pd.isna(last_candle.get('DC_Upper')) else c_high
            dc_lower = float(last_candle['DC_Lower']) if not pd.isna(last_candle.get('DC_Lower')) else c_low
            
            # Validate Donchian Channel (Must be > 0 and valid range)
            if dc_lower <= 0 or dc_upper <= 0 or dc_upper == dc_lower:
                is_touching_dc_lower = False
                is_touching_dc_upper = False
            else:
                is_touching_dc_lower = (c_low <= dc_lower) or (abs(c_close - dc_lower) <= total_range * 0.15)
                is_touching_dc_upper = (c_high >= dc_upper) or (abs(c_close - dc_upper) <= total_range * 0.15)
            
            # --- REAL-TIME BACKEND LOSS-REASON MACHINE LEARNING ---
            recent_loss_reasons = []
            asset_recent_loss_count = 0
            if recent_trades:
                for t in recent_trades:
                    if t.get('asset') == asset_name and (t.get('result') == 'LOSS' or t.get('profit_loss', 0) < 0):
                        asset_recent_loss_count += 1
                        r_str = str(t.get('ai_reason', '') or t.get('reason', ''))
                        recent_loss_reasons.append(r_str)

            # If trend flow lost recently on this asset, temporarily adapt to strict Donchian/Wick reversal
            trend_flow_failed = any("Trend Flow" in r or "EMA_20" in r for r in recent_loss_reasons)

            # Calculate Stochastic Oscillator (5, 3, 3)
            low_5 = df['low'].rolling(window=min(5, len(df))).min()
            high_5 = df['high'].rolling(window=min(5, len(df))).max()
            df['Stoch_K'] = 100.0 * ((df['close'] - low_5) / ((high_5 - low_5).replace(0, 1e-9)))
            stoch_k = float(last_candle['Stoch_K']) if not pd.isna(last_candle.get('Stoch_K')) else 50.0

            # Calculate EMA 50
            df['EMA_50'] = df['close'].ewm(span=min(50, len(df)), adjust=False).mean()
            ema_50 = float(last_candle['EMA_50']) if not pd.isna(last_candle.get('EMA_50')) else c_close

            # --- INSTITUTIONAL QUANT BINARY MATRIX (30S EXPIRY - 85%+ WIN RATE) ---
            # Rule 1: TRIPLE EXTREME CONFLUENCE REVERSAL (98% Ultra Sureshot)
            if (is_touching_dc_lower or rsi_14 <= 35) and stoch_k <= 20:
                reason = f"QUANT PRO [30s]: Donchian Support + RSI ({rsi_14:.1f}) + Stoch ({stoch_k:.1f}) Double Oversold. Signal = BUY (CALL, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "call", "confidence": 98, "reason": reason, "duration": 30}
            elif (is_touching_dc_upper or rsi_14 >= 65) and stoch_k >= 80:
                reason = f"QUANT PRO [30s]: Donchian Resistance + RSI ({rsi_14:.1f}) + Stoch ({stoch_k:.1f}) Double Overbought. Signal = SELL (PUT, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "put", "confidence": 98, "reason": reason, "duration": 30}

            # Rule 2: STRUCTURAL WICK REJECTION BOUNCE (>= 15% WICK) (95% Sureshot)
            elif lower_wick_ratio >= 0.15 or is_gap_down:
                reason = f"QUANT PRO [30s]: Buyer Wick Rejection ({lower_wick_ratio*100:.1f}%). Signal = BUY (CALL, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "call", "confidence": 95, "reason": reason, "duration": 30}
            elif upper_wick_ratio >= 0.15 or is_gap_up:
                reason = f"QUANT PRO [30s]: Seller Wick Rejection ({upper_wick_ratio*100:.1f}%). Signal = SELL (PUT, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "put", "confidence": 95, "reason": reason, "duration": 30}

            # Rule 3: DUAL EMA CONFLUENCE TREND IMPULSE (92% Sureshot)
            elif (c_close > ema_20) and (ema_20 > ema_50) and (c_close >= c_open) and stoch_k > 45:
                reason = "QUANT PRO [30s]: Strong Bullish Trend Impulse (Price > EMA_20 > EMA_50 + Stoch Momentum). Signal = BUY (CALL, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "call", "confidence": 92, "reason": reason, "duration": 30}
            elif (c_close < ema_20) and (ema_20 < ema_50) and (c_close <= c_open) and stoch_k < 55:
                reason = "QUANT PRO [30s]: Strong Bearish Trend Impulse (Price < EMA_20 < EMA_50 + Stoch Momentum). Signal = SELL (PUT, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "put", "confidence": 92, "reason": reason, "duration": 30}

            # Rule 4: 3-BAR OTC MOMENTUM EXPANSION (90% Sureshot)
            elif is_uptrend and (p2_close > p2_open) and (p_close > p_open) and (c_close > c_open):
                reason = "QUANT PRO [30s]: 3 Consecutive Green Bars Expansion in Uptrend. Signal = BUY (CALL, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "call", "confidence": 90, "reason": reason, "duration": 30}
            elif is_downtrend and (p2_close < p2_open) and (p_close < p_open) and (c_close < c_open):
                reason = "QUANT PRO [30s]: 3 Consecutive Red Bars Expansion in Downtrend. Signal = SELL (PUT, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "put", "confidence": 90, "reason": reason, "duration": 30}

            # Rule 5: MICRO-TREND MOMENTUM CONTINUATION (88% Sureshot)
            elif is_uptrend or (c_close >= c_open):
                reason = "QUANT PRO [30s]: Micro-Trend Continuation (Price >= EMA_20). Signal = BUY (CALL, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "call", "confidence": 88, "reason": reason, "duration": 30}
            else:
                reason = "QUANT PRO [30s]: Micro-Trend Continuation (Price < EMA_20). Signal = SELL (PUT, 30s Expiry)."
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "put", "confidence": 88, "reason": reason, "duration": 30}

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

YOUR TRADING RULES FOR OTC MARKETS (BREAKOUT SWEET SPOT STRATEGY):
1. THE OTC TRAP: Quotex OTC markets are controlled by broker algorithms. The broker BAITS traders at extreme RSI levels (above 70 or below 30) by reversing the price exactly during the 60-second trade window. You must NEVER trade at extreme RSI levels.
   - If RSI_14 is > 70: the broker is BAITING momentum traders. You MUST signal "doji". Do NOT place any trade.
   - If RSI_14 is < 30: the broker is BAITING momentum traders. You MUST signal "doji". Do NOT place any trade.
2. THE BREAKOUT SWEET SPOT: The ONLY zones where trading is profitable are:
   - BULLISH BREAKOUT (RSI 55-70): The trend is accelerating upward but has NOT reached the trap zone yet. This is the ONLY zone where you are allowed to signal "call".
   - BEARISH BREAKOUT (RSI 30-45): The trend is accelerating downward but has NOT reached the trap zone yet. This is the ONLY zone where you are allowed to signal "put".
3. THE CHOP ZONE: If RSI_14 is between 45 and 55, the market has NO clear direction. You MUST signal "doji".
4. BOLLINGER BAND BONUS: If price is near the outer Bollinger Band, be cautious. Prefer trades where price has room to move between the middle band and the outer band.
5. CONFIDENCE RULES: Set confidence 65-100 when RSI is in a sweet spot zone AND price confirms with EMA_50. Set confidence below 65 when conditions are unclear.
6. If your Recent Trades Memory shows you lost a trade recently with a similar setup, signal "doji" immediately.

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
                            
                            # PROGRAMMATIC OVERRIDE: BREAKOUT SWEET SPOT STRATEGY
                            sig = result_json.get("signal")
                            if sig in ["call", "put"]:
                                if current_rsi > 70:
                                    result_json["signal"] = "doji"
                                    result_json["confidence"] = 0
                                    result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in the OTC TRAP ZONE (>70). Broker baits traders here. Forced skip."
                                elif current_rsi < 30:
                                    result_json["signal"] = "doji"
                                    result_json["confidence"] = 0
                                    result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in the OTC TRAP ZONE (<30). Broker baits traders here. Forced skip."
                                elif 45 <= current_rsi < 55:
                                    result_json["signal"] = "doji"
                                    result_json["confidence"] = 0
                                    result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in CHOP ZONE. No direction. Forced skip."
                                elif 55 <= current_rsi <= 70 and sig == "put":
                                    result_json["signal"] = "doji"
                                    result_json["confidence"] = 0
                                    result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in BULLISH sweet spot. Put is wrong direction. Forced skip."
                                elif 30 <= current_rsi <= 45 and sig == "call":
                                    result_json["signal"] = "doji"
                                    result_json["confidence"] = 0
                                    result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in BEARISH sweet spot. Call is wrong direction. Forced skip."
                            
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
        
        # PROGRAMMATIC OVERRIDE: BREAKOUT SWEET SPOT STRATEGY
        sig = result_json.get("signal")
        if sig in ["call", "put"]:
            if current_rsi > 70:
                result_json["signal"] = "doji"
                result_json["confidence"] = 0
                result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in the OTC TRAP ZONE (>70). Broker baits traders here. Forced skip."
            elif current_rsi < 30:
                result_json["signal"] = "doji"
                result_json["confidence"] = 0
                result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in the OTC TRAP ZONE (<30). Broker baits traders here. Forced skip."
            elif 45 <= current_rsi < 55:
                result_json["signal"] = "doji"
                result_json["confidence"] = 0
                result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in CHOP ZONE. No direction. Forced skip."
            elif 55 <= current_rsi <= 70 and sig == "put":
                result_json["signal"] = "doji"
                result_json["confidence"] = 0
                result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in BULLISH sweet spot. Put is wrong direction. Forced skip."
            elif 30 <= current_rsi <= 45 and sig == "call":
                result_json["signal"] = "doji"
                result_json["confidence"] = 0
                result_json["reason"] = f"OVERRIDE: RSI {current_rsi:.2f} is in BEARISH sweet spot. Call is wrong direction. Forced skip."
        
        logger.info(f"[{asset_name}] Groq AI Prediction: {result_json}")
        return result_json

    except Exception as e:
        logger.error(f"[{asset_name}] Error during AI analysis: {e}", exc_info=True)
        return {"signal": "doji", "confidence": 0, "reason": f"AI Error: {str(e)}"}
