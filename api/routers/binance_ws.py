"""Binance WebSocket proxy: backend connects to Binance WS and forwards to frontend clients."""

import json

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from loguru import logger

router = APIRouter(tags=["websocket"])

BINANCE_COMBINED = "wss://stream.binance.com:9443/stream"


async def binance_relay(websocket: WebSocket, symbol: str, interval: str) -> None:
    """Connect to Binance WebSocket and relay kline + ticker to client."""
    symbol_lower = symbol.lower().replace("/", "")
    streams = [
        f"{symbol_lower}@kline_{interval}",
        f"{symbol_lower}@ticker",
    ]
    url = f"{BINANCE_COMBINED}?streams={'/'.join(streams)}"

    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as bn_ws:
            logger.info(f"Binance WS connected: {symbol} @ {interval}")

            async def forward():
                try:
                    async for raw in bn_ws:
                        try:
                            msg = json.loads(raw)
                            # Normalize: combined stream returns { stream, data }
                            if "stream" in msg and "data" in msg:
                                data = msg["data"]
                                e = data.get("e")
                                out = {"event": e, "stream": msg["stream"]}
                                if e == "kline":
                                    k = data.get("k", {})
                                    out["kline"] = {
                                        "time": int(k.get("t", 0)),
                                        "open": float(k.get("o", 0)),
                                        "high": float(k.get("h", 0)),
                                        "low": float(k.get("l", 0)),
                                        "close": float(k.get("c", 0)),
                                        "volume": float(k.get("v", 0)),
                                        "isClosed": k.get("x", False),
                                    }
                                elif e == "24hrTicker":
                                    out["price"] = float(data.get("c", 0))
                                else:
                                    out["raw"] = data
                            else:
                                out = msg
                            await websocket.send_json(out)
                        except Exception as e:
                            logger.warning(f"WS relay parse error: {e}")
                except Exception as e:
                    logger.warning(f"Binance WS forward error: {e}")

            await forward()
    except Exception as e:
        logger.error(f"Binance WS connect error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@router.websocket("/ws/crypto")
async def websocket_crypto(websocket: WebSocket):
    """WebSocket endpoint for Binance crypto stream. Query params: symbol, interval."""
    await websocket.accept()
    symbol = websocket.query_params.get("symbol", "btcusdt").upper().replace("/", "")
    interval = websocket.query_params.get("interval", "1h")

    # Normalize interval for Binance
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "4h", "1d": "1d",
    }
    interval = interval_map.get(interval.lower(), "1h")

    try:
        await binance_relay(websocket, symbol, interval)
    except WebSocketDisconnect:
        logger.info("Client disconnected from /ws/crypto")
    except Exception as e:
        logger.error(f"WS crypto error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
