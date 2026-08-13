import asyncio
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

async def add_all_pairs():
    load_dotenv()
    client = AsyncIOMotorClient(os.getenv('MONGO_URI'))
    db = client.quotexTraderBot
    
    pairs = [
        "USDEGP", "USDMXN", "AUDCAD", "EURAUD", "GBPAUD", "GBPCAD", 
        "USDDZD", "USDPHP", "GBPNZD", "NZDCAD", "USDARS", "CADCHF", 
        "USDIDR", "USDBRL", "CHFJPY", "GBPJPY", "AUDJPY", "EURNZD", 
        "USDBDT", "AUDNZD", "USDNGN", "AUDCHF", "EURCHF", "CADJPY", 
        "EURUSD", "USDCAD", "USDCOP", "USDPKR", "USDCHF", "AUDUSD", 
        "EURGBP", "EURJPY", "NZDCHF", "NZDJPY", "USDJPY", "EURCAD", 
        "USDINR", "NZDUSD", "USDZAR", "GBPUSD"
    ]
    
    new_assets = []
    for pair in pairs:
        new_assets.append({
            "name": pair,
            "base_amount": 1,
            "candle_size": 60,
            "duration": 60,
            "timeframe": 60,
            "mode": "TIMER",
            "is_active": True
        })
        
    # Get all users (trade_settings)
    cursor = db.trade_settings.find({})
    async for account in cursor:
        await db.trade_settings.update_one({"_id": account["_id"]}, {"$set": {"assets": new_assets}})
        
    print(f"Success! Updated DB. Total pairs added: {len(pairs)}")

asyncio.run(add_all_pairs())
