"""
LLM prompt template and response parser.

The prompt is the heart of the abstention discipline. We explicitly teach
the LLM to say "ABSTAIN" when there's no information edge. This is critical:
without this, the LLM will produce direction calls on every minute just to
be helpful, defeating the entire point of the layer.
"""

import re
import logging

logger = logging.getLogger("llm.prompts")


def build_signal_prompt(
    asset: str,
    candles_summary: list[dict],
    indicators: dict,
    news_articles: list[dict],
    events: list[dict],
    sentiment: dict,
) -> str:
    """
    Build the prompt that asks the LLM whether to trade.

    Inputs are pre-computed by the caller. The LLM's job is to combine them
    and either call CALL/PUT with confidence, or ABSTAIN.

    The prompt enforces strict abstention when there's no scheduled event.
    """
    # Format candle summary
    candle_lines = []
    for c in candles_summary[-5:]:
        candle_lines.append(
            f"  {c.get('time', '?')}: O={c.get('open', '?')} "
            f"H={c.get('high', '?')} L={c.get('low', '?')} C={c.get('close', '?')}"
        )
    candle_block = "\n".join(candle_lines) if candle_lines else "  (no candles)"

    # Format indicators
    ind_lines = []
    for k, v in indicators.items():
        ind_lines.append(f"  {k}: {v}")
    ind_block = "\n".join(ind_lines) if ind_lines else "  (no indicators)"

    # Format events
    if events:
        event_lines = []
        for e in events:
            event_lines.append(
                f"  - {e.get('time_utc', '?')} UTC ({e.get('hours_away', '?')}h away): "
                f"{e.get('title', '?')} [{e.get('impact', '?')}]"
            )
        event_block = "\n".join(event_lines)
    else:
        event_block = "  (no high-impact events in next 4 hours)"

    # Format news
    if news_articles:
        news_lines = []
        for a in news_articles[:5]:
            news_lines.append(f"  - [{a.get('source', '?')}] {a.get('title', '?')}")
        news_block = "\n".join(news_lines)
    else:
        news_block = "  (no recent news available)"

    sentiment_line = (
        f"Sentiment: {sentiment.get('score', 0):+d} "
        f"({sentiment.get('breakdown', {}).get('pos', 0)} positive, "
        f"{sentiment.get('breakdown', {}).get('neg', 0)} negative, "
        f"{sentiment.get('breakdown', {}).get('neutral', 0)} neutral out of {sentiment.get('n', 0)} articles)"
    )

    prompt = f"""You are a trading signal assistant. You decide whether to place a binary option (CALL = price up, PUT = price down) on {asset} for a 1-5 minute horizon.

**Your primary instruction: ABSTAIN by default.** Most minutes, you should return ABSTAIN. Only return CALL or PUT when there is a SCHEDULED HIGH-IMPACT EVENT in the next 15 minutes AND news/directional flow supports a clear direction.

**ABSTAIN rules (return ABSTAIN if any are true):**
1. No high-impact economic event in the next 15 minutes
2. News is mixed or unclear
3. Indicators do not confirm a direction
4. You are not at least 60% confident
5. The asset is range-bound or choppy

**Recent candles ({len(candles_summary)} shown, last 5):**
{candle_block}

**Indicators:**
{ind_block}

**Upcoming high-impact events (next 4h):**
{event_block}

**Recent news (last {sentiment.get('n', 0)} articles):**
{news_block}

{sentiment_line}

**Decision format — respond with EXACTLY one line, in this format:**

CALL <confidence 0-100> | <one-sentence reason>
PUT <confidence 0-100> | <one-sentence reason>
ABSTAIN | <one-sentence reason>

Examples:
CALL 72 | ECB rate hold signals EUR strength vs USD
PUT 68 | Hot CPI print likely to boost USD across pairs
ABSTAIN | No high-impact event in next 15 min

Respond with your decision now. One line only."""
    return prompt


def parse_llm_response(text: str) -> dict:
    """
    Parse the LLM's text response into a structured signal dict.

    Always returns a dict with keys: signal, confidence, reason.
    On any parse failure, returns ABSTAIN.
    """
    text = (text or "").strip()
    if not text:
        return {"signal": "abstain", "confidence": 0, "reason": "empty response"}

    # Take the first non-empty line that looks like a decision
    line = None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        up = ln.upper()
        if up.startswith(("CALL", "PUT", "ABSTAIN")):
            line = ln
            break
    if line is None:
        return {"signal": "abstain", "confidence": 0, "reason": f"unparseable: {text[:80]}"}

    parts = re.split(r"[|]", line, maxsplit=1)
    head = parts[0].strip()
    reason = parts[1].strip() if len(parts) > 1 else ""

    # Detect direction word (case-insensitive, first token)
    head_up = head.upper()
    if head_up.startswith("CALL"):
        signal = "call"
    elif head_up.startswith("PUT"):
        signal = "put"
    elif head_up.startswith("ABSTAIN"):
        signal = "abstain"
    else:
        return {"signal": "abstain", "confidence": 0, "reason": f"unknown direction: {head[:40]}"}

    # Extract confidence (first integer 0-100 in the head portion)
    if signal == "abstain":
        confidence = 0
    else:
        # Match integers 0-100 (anywhere in head, after the direction word).
        # We accept the first one in the 0-100 range, since LLMs sometimes
        # write "CALL with 72% confidence" and we want 72.
        m = re.search(r"\b(\d{1,3})\b", head[4:])
        if m:
            n = int(m.group(1))
            if 0 <= n <= 100:
                confidence = n
            else:
                confidence = max(0, min(100, n))
        else:
            return {"signal": "abstain", "confidence": 0, "reason": f"no confidence in: {head[:40]}"}

    return {
        "signal": signal,
        "confidence": confidence,
        "reason": reason[:200] or f"LLM {signal} {confidence}",
    }
