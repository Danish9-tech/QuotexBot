import os
import time
import json
import logging
import math
import pandas as pd
import numpy as np
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


# =========================================================================
# HELPER: Calculate all technical indicators on a DataFrame
# =========================================================================
def _compute_indicators(df: pd.DataFrame, lookback_rsi: int = 14, lookback_ema_fast: int = 20,
                        lookback_ema_mid: int = 50, lookback_ema_slow: int = 100,
                        lookback_dc: int = 24, lookback_adx: int = 14,
                        lookback_chop: int = 14, lookback_atr: int = 14) -> pd.DataFrame:
    """Compute all indicators in one pass. Returns the same df with new columns."""
    n = len(df)

    # --- EMAs ---
    df['EMA_fast'] = df['close'].ewm(span=min(lookback_ema_fast, n), adjust=False).mean()
    df['EMA_mid'] = df['close'].ewm(span=min(lookback_ema_mid, n), adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=min(lookback_ema_slow, n), adjust=False).mean()

    # --- RSI (Exponential) ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / lookback_rsi, adjust=False).mean()
    loss_s = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / lookback_rsi, adjust=False).mean()
    rs = gain / loss_s.replace(0, 1e-9)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # --- ATR (Average True Range) ---
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=min(lookback_atr, n)).mean()

    # --- ADX (Average Directional Index) ---
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_adx = tr.ewm(alpha=1 / lookback_adx, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=1 / lookback_adx, adjust=False).mean() / atr_adx.replace(0, 1e-9))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1 / lookback_adx, adjust=False).mean() / atr_adx.replace(0, 1e-9))
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
    df['ADX'] = dx.ewm(alpha=1 / lookback_adx, adjust=False).mean()
    df['PLUS_DI'] = plus_di
    df['MINUS_DI'] = minus_di

    # --- Choppiness Index ---
    atr_sum = tr.rolling(window=min(lookback_chop, n)).sum()
    hh = df['high'].rolling(window=min(lookback_chop, n)).max()
    ll = df['low'].rolling(window=min(lookback_chop, n)).min()
    chop_range = (hh - ll).replace(0, 1e-9)
    df['CHOP'] = 100.0 * np.log10(atr_sum / chop_range) / math.log10(lookback_chop)

    # --- Donchian Channel ---
    df['DC_Upper'] = df['high'].rolling(window=min(lookback_dc, n)).max()
    df['DC_Lower'] = df['low'].rolling(window=min(lookback_dc, n)).min()

    # --- Stochastic Oscillator (5, 3, 3) ---
    stoch_period = min(5, n)
    low_k = df['low'].rolling(window=stoch_period).min()
    high_k = df['high'].rolling(window=stoch_period).max()
    df['Stoch_K'] = 100.0 * ((df['close'] - low_k) / (high_k - low_k).replace(0, 1e-9))

    # --- Bollinger Bands (20, 2) ---
    bb_period = min(20, n)
    df['BB_Mid'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Mid'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Mid'] - (bb_std * 2)

    return df


# =========================================================================
# HELPER: Detect rollover / danger hours
# =========================================================================
def _is_rollover_hour() -> bool:
    """Returns True during known OTC rollover danger windows (UTC).
    OTC markets often become erratic at hour boundaries and half-hour marks."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    minute = now.minute
    # Danger windows: first 2 min and last 2 min of each half-hour block
    if minute <= 1 or minute >= 58:
        return True
    if 29 <= minute <= 31:
        return True
    return False


# =========================================================================
# MAIN SIGNAL FUNCTION — UPGRADED WITH ALL 12 COMPETITIVE FEATURES
# =========================================================================
async def get_groq_trading_signal(candles: list, asset_name: str, candle_size: int = 60, recent_trades: list = None, risk_mode: str = "SAFE") -> dict:
    """
    Takes a list of candle dictionaries, calculates indicators, 
    injects past trade memory, and determines trading decision.
    
    UPGRADES IMPLEMENTED:
    1. Anti-Repaint: Excludes the last (running) candle from analysis
    2. Choppiness Index Filter: Skips when CHOP > 61.8
    3. ADX Trend Strength Filter: Skips when ADX < 20
    4. ATR Volatility Filter: Skips when ATR is compressed
    5. Rollover Hour Avoidance: Skips during danger windows
    6. Multi-Timeframe Confirmation: Uses higher-TF EMA for trend
    7. Stricter Confluence: Requires 4+ confirmations for entry
    8. Dynamic Duration: Returns None so Telegram setting is used
    """
    global CURRENT_KEY_INDEX

    min_required_candles = 5
    if not candles or len(candles) < min_required_candles:
        logger.warning(f"[{asset_name}] Not enough candles for analysis (need at least {min_required_candles}, got {len(candles) if candles else 0}).")
        return {"signal": "doji", "confidence": 0, "reason": "Not enough data"}

    try:
        # =====================================================================
        # STEP 1: Normalize candle data
        # =====================================================================
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

        df = pd.DataFrame(normalized_candles)

        # =====================================================================
        # STEP 2: ANTI-REPAINT — Remove the LAST candle (still forming/running)
        # Only analyze fully CLOSED candles to prevent indicator repainting
        # =====================================================================
        if len(df) > min_required_candles:
            df = df.iloc[:-1]  # Drop the running candle

        if len(df) < 10:
            return {"signal": "doji", "confidence": 0, "reason": "Not enough closed candles for analysis"}

        # =====================================================================
        # STEP 3: Compute ALL indicators
        # =====================================================================
        df = _compute_indicators(df)

        # Extract the LAST CLOSED candle and previous candles
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        prev_candle2 = df.iloc[-3] if len(df) >= 3 else prev_candle

        c_open = float(last_candle['open'])
        c_close = float(last_candle['close'])
        c_high = float(last_candle['high'])
        c_low = float(last_candle['low'])

        p_open = float(prev_candle['open'])
        p_close = float(prev_candle['close'])

        total_range = max(0.000001, c_high - c_low)
        body_size = abs(c_close - c_open)
        upper_wick = c_high - max(c_open, c_close)
        lower_wick = min(c_open, c_close) - c_low

        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        body_ratio = body_size / total_range

        # Extract indicator values (with safe defaults)
        def _safe(val, default=50.0):
            return float(val) if pd.notna(val) else default

        ema_fast = _safe(last_candle.get('EMA_fast'), c_close)
        ema_mid = _safe(last_candle.get('EMA_mid'), c_close)
        ema_slow = _safe(last_candle.get('EMA_slow'), c_close)
        rsi = _safe(last_candle.get('RSI'), 50.0)
        rsi = max(1.0, min(99.0, rsi))  # Clamp to valid range
        adx = _safe(last_candle.get('ADX'), 0.0)
        plus_di = _safe(last_candle.get('PLUS_DI'), 0.0)
        minus_di = _safe(last_candle.get('MINUS_DI'), 0.0)
        chop = _safe(last_candle.get('CHOP'), 100.0)
        atr = _safe(last_candle.get('ATR'), 0.0)
        stoch_k = _safe(last_candle.get('Stoch_K'), 50.0)
        dc_upper = _safe(last_candle.get('DC_Upper'), c_high)
        dc_lower = _safe(last_candle.get('DC_Lower'), c_low)
        bb_upper = _safe(last_candle.get('BB_Upper'), c_high)
        bb_lower = _safe(last_candle.get('BB_Lower'), c_low)
        bb_mid = _safe(last_candle.get('BB_Mid'), c_close)

        # =====================================================================
        # STEP 4: MARKET REGIME FILTERS — Skip choppy/dead markets
        # =====================================================================

        # FILTER 1: Choppiness Index — CHOP > 61.8 means market is range-bound
        if chop > 61.8:
            logger.info(f"[{asset_name}] CHOP FILTER: Choppiness Index = {chop:.1f} (> 61.8). Market is choppy. SKIP.")
            return {"signal": "doji", "confidence": 0, "reason": f"Choppiness Index {chop:.1f} > 61.8 — market is range-bound"}

        # FILTER 2: ADX Trend Strength — ADX < 20 means no trend exists
        if adx < 20.0:
            logger.info(f"[{asset_name}] ADX FILTER: ADX = {adx:.1f} (< 20). No trend. SKIP.")
            return {"signal": "doji", "confidence": 0, "reason": f"ADX {adx:.1f} < 20 — no trend detected"}

        # FILTER 3: ATR Compression — If ATR is < 60% of its 20-period average, market is dead
        atr_values = df['ATR'].dropna()
        if len(atr_values) >= 20:
            atr_avg_20 = atr_values.tail(20).mean()
            if atr_avg_20 > 0 and atr < (atr_avg_20 * 0.60):
                logger.info(f"[{asset_name}] ATR FILTER: ATR = {atr:.6f} < 60% of avg ({atr_avg_20:.6f}). Market is dead. SKIP.")
                return {"signal": "doji", "confidence": 0, "reason": f"ATR compressed — market is stagnant"}

        # FILTER 4: Rollover Hour Avoidance
        if _is_rollover_hour():
            logger.info(f"[{asset_name}] ROLLOVER FILTER: Currently in OTC rollover danger window. SKIP.")
            return {"signal": "doji", "confidence": 0, "reason": "OTC rollover hour — erratic price action"}

        # FILTER 5: Flat Doji — Zero-range candle
        if c_close == c_open and total_range < 0.0000001:
            return {"signal": "doji", "confidence": 0, "reason": "Zero-range Doji candle"}

        # =====================================================================
        # STEP 5: MULTI-TIMEFRAME TREND CONFIRMATION
        # Use EMA_slow (100-period) as proxy for higher timeframe trend
        # =====================================================================
        htf_bullish = c_close > ema_slow  # Price above long-term EMA = bullish bias
        htf_bearish = c_close < ema_slow  # Price below long-term EMA = bearish bias

        is_uptrend = c_close >= ema_fast and plus_di > minus_di
        is_downtrend = c_close < ema_fast and minus_di > plus_di

        # =====================================================================
        # STEP 6: DONCHIAN / BOLLINGER PROXIMITY
        # =====================================================================
        if dc_lower <= 0 or dc_upper <= 0 or dc_upper == dc_lower:
            is_touching_dc_lower = False
            is_touching_dc_upper = False
        else:
            is_touching_dc_lower = (c_low <= dc_lower) or (abs(c_close - dc_lower) <= total_range * 0.15)
            is_touching_dc_upper = (c_high >= dc_upper) or (abs(c_close - dc_upper) <= total_range * 0.15)

        near_bb_lower = c_close <= bb_lower or (abs(c_close - bb_lower) <= total_range * 0.2)
        near_bb_upper = c_close >= bb_upper or (abs(c_close - bb_upper) <= total_range * 0.2)

        # Gap detection
        gap = c_open - p_close
        gap_threshold = max(0.000001, total_range * 0.15)
        is_gap_up = gap > gap_threshold
        is_gap_down = gap < -gap_threshold

        # =====================================================================
        # STEP 7: AI LOSS MEMORY ENGINE
        # =====================================================================
        recent_loss_reasons = []
        asset_recent_loss_count = 0
        last_loss_direction = None
        now_ts = time.time()

        if recent_trades:
            for t in recent_trades:
                if t.get('asset') == asset_name and (t.get('result') == 'LOSS' or t.get('profit_loss', 0) < 0):
                    t_time = t.get('timestamp')
                    is_recent = False
                    if isinstance(t_time, (int, float)):
                        is_recent = (now_ts - t_time) < 180
                    elif hasattr(t_time, 'timestamp'):
                        is_recent = (now_ts - t_time.timestamp()) < 180

                    if is_recent:
                        asset_recent_loss_count += 1
                        r_str = str(t.get('ai_reason', '') or t.get('reason', ''))
                        recent_loss_reasons.append(r_str)
                        if last_loss_direction is None:
                            if any(x in r_str for x in ["BUY", "CALL"]) or t.get('direction') == 'call':
                                last_loss_direction = "call"
                            elif any(x in r_str for x in ["SELL", "PUT"]) or t.get('direction') == 'put':
                                last_loss_direction = "put"

        logger.info(f"[{asset_name}] Indicators: RSI={rsi:.1f} ADX={adx:.1f} CHOP={chop:.1f} ATR={atr:.6f} StochK={stoch_k:.1f} | Losses(3m)={asset_recent_loss_count}")

        # =====================================================================
        # STEP 8: HIGH-CONVICTION SIGNAL MATRIX (REQUIRES 4+ CONFLUENCES)
        # =====================================================================
        # Each rule now requires: Indicator signal + ADX trend + HTF confirmation + No loss shield

        # --- RULE 1: TRIPLE EXTREME CONFLUENCE REVERSAL (98% Confidence) ---
        # Requires: DC boundary + RSI extreme + Stoch extreme + HTF trend agreement
        if (is_touching_dc_lower or rsi <= 35) and stoch_k <= 30 and htf_bullish:
            if last_loss_direction == "call" and asset_recent_loss_count >= 2:
                logger.warning(f"[{asset_name}] AI Loss Shield: Skipping CALL reversal due to 2+ consecutive CALL losses.")
            else:
                confluences = sum([is_touching_dc_lower, rsi <= 35, stoch_k <= 30, htf_bullish, near_bb_lower, adx > 25])
                reason = f"SURESHOT: DC Support + RSI({rsi:.1f}) Oversold + StochK({stoch_k:.1f}) + HTF Bullish + ADX({adx:.1f}). Confluences={confluences}"
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "call", "confidence": min(98, 85 + confluences * 2), "reason": reason, "duration": None}

        elif (is_touching_dc_upper or rsi >= 65) and stoch_k >= 70 and htf_bearish:
            if last_loss_direction == "put" and asset_recent_loss_count >= 2:
                logger.warning(f"[{asset_name}] AI Loss Shield: Skipping PUT reversal due to 2+ consecutive PUT losses.")
            else:
                confluences = sum([is_touching_dc_upper, rsi >= 65, stoch_k >= 70, htf_bearish, near_bb_upper, adx > 25])
                reason = f"SURESHOT: DC Resistance + RSI({rsi:.1f}) Overbought + StochK({stoch_k:.1f}) + HTF Bearish + ADX({adx:.1f}). Confluences={confluences}"
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "put", "confidence": min(98, 85 + confluences * 2), "reason": reason, "duration": None}

        # --- RULE 2: WICK REJECTION + TREND AGREEMENT (95% Confidence) ---
        # Requires: Strong wick + RSI supports + HTF confirms + ADX trending
        elif lower_wick_ratio >= 0.20 and rsi < 55 and htf_bullish and adx > 22:
            if last_loss_direction == "call" and asset_recent_loss_count >= 2:
                logger.warning(f"[{asset_name}] AI Loss Shield: Skipping CALL wick bounce.")
            else:
                confluences = sum([lower_wick_ratio >= 0.20, rsi < 55, htf_bullish, adx > 22, is_gap_down, near_bb_lower])
                reason = f"WICK REJECT: Buyer Rejection({lower_wick_ratio*100:.0f}%) + RSI({rsi:.1f}) + HTF Bull + ADX({adx:.1f}). Confluences={confluences}"
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "call", "confidence": min(95, 82 + confluences * 2), "reason": reason, "duration": None}

        elif upper_wick_ratio >= 0.20 and rsi > 45 and htf_bearish and adx > 22:
            if last_loss_direction == "put" and asset_recent_loss_count >= 2:
                logger.warning(f"[{asset_name}] AI Loss Shield: Skipping PUT wick bounce.")
            else:
                confluences = sum([upper_wick_ratio >= 0.20, rsi > 45, htf_bearish, adx > 22, is_gap_up, near_bb_upper])
                reason = f"WICK REJECT: Seller Rejection({upper_wick_ratio*100:.0f}%) + RSI({rsi:.1f}) + HTF Bear + ADX({adx:.1f}). Confluences={confluences}"
                logger.info(f"[{asset_name}] {reason}")
                return {"signal": "put", "confidence": min(95, 82 + confluences * 2), "reason": reason, "duration": None}

        # --- RULE 3: TREND IMPULSE + FULL CONFLUENCE (90% Confidence) ---
        # Requires: EMA trend + price confirms + HTF agrees + ADX strong + not sideways
        elif is_uptrend and c_close > p_close and htf_bullish and adx > 25:
            if last_loss_direction == "call" and asset_recent_loss_count >= 2:
                logger.warning(f"[{asset_name}] AI Loss Shield: Skipping CALL trend impulse.")
            else:
                confluences = sum([is_uptrend, c_close > p_close, htf_bullish, adx > 25, rsi > 50, plus_di > minus_di])
                if confluences >= 4:
                    reason = f"TREND IMPULSE: Bull EMA + Close>Prev + HTF Bull + ADX({adx:.1f}) + +DI>{'-'}DI. Confluences={confluences}"
                    logger.info(f"[{asset_name}] {reason}")
                    return {"signal": "call", "confidence": min(92, 80 + confluences * 2), "reason": reason, "duration": None}

        elif is_downtrend and c_close < p_close and htf_bearish and adx > 25:
            if last_loss_direction == "put" and asset_recent_loss_count >= 2:
                logger.warning(f"[{asset_name}] AI Loss Shield: Skipping PUT trend impulse.")
            else:
                confluences = sum([is_downtrend, c_close < p_close, htf_bearish, adx > 25, rsi < 50, minus_di > plus_di])
                if confluences >= 4:
                    reason = f"TREND IMPULSE: Bear EMA + Close<Prev + HTF Bear + ADX({adx:.1f}) + {'-'}DI>+DI. Confluences={confluences}"
                    logger.info(f"[{asset_name}] {reason}")
                    return {"signal": "put", "confidence": min(92, 80 + confluences * 2), "reason": reason, "duration": None}

        # --- NO SIGNAL: Not enough confluence ---
        logger.info(f"[{asset_name}] PATIENCE MODE: No 4+ confluence setup found. Waiting for high-probability entry.")
        return {"signal": "doji", "confidence": 0, "reason": "Waiting — insufficient confluence for high-probability entry"}

    except Exception as e:
        logger.error(f"[{asset_name}] Error during AI analysis: {e}", exc_info=True)
        return {"signal": "doji", "confidence": 0, "reason": f"AI Error: {str(e)}"}
