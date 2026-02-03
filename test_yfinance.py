#!/usr/bin/env python3
"""Quick test script to verify yfinance data fetching."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta
import asyncio
from data.ingest.equities_yfinance import EquitiesYFinanceIngestor

async def test_fetch():
    """Test fetching data from yfinance."""
    ingestor = EquitiesYFinanceIngestor()
    
    end = datetime.now()
    start = end - timedelta(days=30)
    
    print(f"Testing yfinance data fetch for AAPL...")
    print(f"Date range: {start.date()} to {end.date()}")
    
    try:
        df = await ingestor.fetch_ohlcv(
            symbol="AAPL",
            timeframe="1d",
            start=start,
            end=end,
        )
        
        print(f"\n✅ Success! Fetched {len(df)} bars")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fetch())
