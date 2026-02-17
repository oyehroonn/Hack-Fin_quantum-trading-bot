"""Market API: OHLCV for charts, latest price for ticker."""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from typing import Optional

import pandas as pd
import pytz
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/market", tags=["market"])


def _days_for_timeframe(timeframe: str, days: int) -> int:
    """Limit days for intraday to stay under API limits (e.g. Binance 1000 bars)."""
    tf = (timeframe or "1d").lower()
    if tf in ("1m", "1min"): return min(days, 2)
    if tf in ("5m", "5min"): return min(days, 7)
    if tf in ("15m", "15min"): return min(days, 14)
    if tf in ("30m", "30min"): return min(days, 30)
    if tf in ("1h", "60m"): return min(days, 60)
    return days


async def _fetch_ohlcv(symbol: str, asset_class: str, timeframe: str, days: int) -> pd.DataFrame:
    utc = pytz.UTC
    end = pd.Timestamp.now(tz=utc)
    days = _days_for_timeframe(timeframe, days)
    start = end - pd.Timedelta(days=days)

    if asset_class == "crypto":
        from data.ingest.binance_public import BinancePublicIngestor
        ingestor = BinancePublicIngestor()
    else:
        from data.ingest.equities_yfinance import EquitiesYFinanceIngestor
        ingestor = EquitiesYFinanceIngestor()

    sym = symbol.upper().replace("/", "")
    df = await ingestor.fetch_ohlcv(symbol=sym, timeframe=timeframe, start=start, end=end)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    return df


@router.get("/ohlcv")
async def get_ohlcv(
    symbol: str,
    asset_class: str = "equities",
    timeframe: str = "1d",
    days: int = 60,
):
    """Get OHLCV data for charts. Returns [{time, open, high, low, close, volume}, ...]."""
    df = await _fetch_ohlcv(symbol, asset_class, timeframe, days)

    # Normalize columns
    col_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = ["open", "high", "low", "close"]
    for c in required:
        if c not in df.columns:
            raise HTTPException(status_code=500, detail=f"Missing column: {c}")

    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        df["time"] = df.index.astype(str)

    records = df[["time", "open", "high", "low", "close"]].copy()
    if "volume" in df.columns:
        records["volume"] = df["volume"].values
    else:
        records["volume"] = 0

    return {"ohlcv": records.to_dict("records"), "symbol": symbol}


@router.get("/price")
async def get_price(symbol: str, asset_class: str = "equities"):
    """Get latest price for a symbol (lightweight, for ticker)."""
    df = await _fetch_ohlcv(symbol, asset_class, "1d", 7)
    close_col = "close" if "close" in df.columns else "Close"
    price = float(df[close_col].iloc[-1])
    return {"symbol": symbol, "price": price, "asset_class": asset_class}
