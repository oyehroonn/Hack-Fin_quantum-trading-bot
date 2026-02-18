"""Simulated futures trader: executes decisions, manages positions, handles liquidations."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from analytics.monte_carlo import run_monte_carlo
from analytics.statistical import full_statistical_analysis
from bot.futures_db import FuturesDB, FuturesPosition
from bot.futures_decision_engine import FuturesDecisionEngine, FuturesDecision
from bot.bot_db import BotDB


class FuturesTrader:
    """Simulated futures trading bot with leverage."""

    def __init__(
        self,
        session_id: str,
        symbol: str = "BTCUSDT",
        base_margin: float = 100.0,
        max_leverage: int = 50,
        check_interval: int = 60,
        mc_paths: int = 5000,
        mc_horizon: int = 3,
    ):
        self.session_id = session_id
        self.symbol = symbol
        self.base_margin = base_margin
        self.max_leverage = max_leverage
        self.check_interval = check_interval
        self.mc_paths = mc_paths
        self.mc_horizon = mc_horizon

        self.futures_db = FuturesDB("data/terminal.db")
        self.bot_db = BotDB("data/terminal.db")
        self.engine = FuturesDecisionEngine(
            base_margin=base_margin,
            max_leverage=max_leverage,
        )

        self.is_running = False
        self._trade_count = 0
        self._total_pnl = 0.0

    async def _fetch_prices(self, days: int = 90) -> np.ndarray:
        """Fetch historical close prices from Binance."""
        from data.ingest.binance_public import BinancePublicIngestor
        ingestor = BinancePublicIngestor()
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=days)
        df = await ingestor.fetch_ohlcv(symbol=self.symbol, timeframe="1d", start=start, end=end)
        if df.empty or "close" not in df.columns:
            raise ValueError(f"No price data for {self.symbol}")
        return df["close"].values.astype(float)

    async def _get_current_price(self) -> float:
        """Get latest price."""
        from data.ingest.binance_public import BinancePublicIngestor
        ingestor = BinancePublicIngestor()
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(hours=2)
        df = await ingestor.fetch_ohlcv(symbol=self.symbol, timeframe="1m", start=start, end=end, limit=5)
        if df.empty:
            raise ValueError(f"Cannot get current price for {self.symbol}")
        return float(df["close"].iloc[-1])

    def _get_ml_signal(self, prices: np.ndarray) -> tuple[str, float]:
        """ML signal based on SMA crossover + momentum + RSI."""
        if len(prices) < 30:
            return "HOLD", 0.5

        sma_fast = np.mean(prices[-10:])
        sma_slow = np.mean(prices[-30:])

        deltas = np.diff(prices[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 50

        momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        momentum_10 = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0

        score = 0.5

        if sma_fast > sma_slow * 1.01:
            score += 0.15
        elif sma_fast < sma_slow * 0.99:
            score -= 0.15

        if rsi < 30:
            score += 0.12
        elif rsi > 70:
            score -= 0.12

        if momentum_5 > 0.02:
            score += 0.08
        elif momentum_5 < -0.02:
            score -= 0.08

        if momentum_10 > 0.03:
            score += 0.05
        elif momentum_10 < -0.03:
            score -= 0.05

        score = max(0.1, min(0.9, score))

        if score >= 0.58:
            return "BUY", score
        elif score <= 0.42:
            return "SELL", 1 - score
        return "HOLD", 0.5

    def _check_liquidations(self, current_price: float) -> list[FuturesPosition]:
        """Check and liquidate positions that hit liquidation price."""
        liquidated = []
        positions = self.futures_db.get_open_positions(self.session_id, self.symbol)

        for pos in positions:
            should_liquidate = False
            if pos.direction == "LONG" and current_price <= pos.liquidation_price:
                should_liquidate = True
            elif pos.direction == "SHORT" and current_price >= pos.liquidation_price:
                should_liquidate = True

            if should_liquidate:
                closed = self.futures_db.close_position(pos.id, current_price, "LIQUIDATION")
                if closed:
                    liquidated.append(closed)
                    logger.warning(f"LIQUIDATED {pos.direction} {pos.symbol} @ ${current_price:.2f} (entry: ${pos.entry_price:.2f}, liq: ${pos.liquidation_price:.2f})")

        return liquidated

    async def _run_cycle(self) -> Optional[FuturesDecision]:
        """Run one trading cycle."""
        try:
            prices = await self._fetch_prices(days=90)
            current_price = float(prices[-1])

            liquidated = self._check_liquidations(current_price)
            for liq in liquidated:
                self._total_pnl += liq.realized_pnl

            ml_signal, ml_confidence = self._get_ml_signal(prices)

            mc_result = run_monte_carlo(
                symbol=self.symbol,
                prices=prices,
                horizon_days=self.mc_horizon,
                num_paths=self.mc_paths,
            )

            stats = full_statistical_analysis(prices)

            session = self.futures_db.get_session(self.session_id)
            available_margin = session.get("current_margin", 0) if session else 0

            open_positions = self.futures_db.get_open_positions(self.session_id, self.symbol)
            positions_dicts = [
                {"direction": p.direction, "entry_price": p.entry_price, "margin": p.margin,
                 "leverage": p.leverage, "unrealized_pnl": p.unrealized_pnl}
                for p in open_positions
            ]

            for pos in open_positions:
                self.futures_db.update_position_price(pos.id, current_price)

            decision = self.engine.decide(
                ml_signal=ml_signal,
                ml_confidence=ml_confidence,
                mc_result=mc_result.to_dict(),
                stats=stats,
                open_positions=positions_dicts,
                available_margin=available_margin,
                current_price=current_price,
            )

            if decision.action.startswith("OPEN_"):
                pos = self.futures_db.open_position(
                    session_id=self.session_id,
                    symbol=self.symbol,
                    direction=decision.direction,
                    entry_price=current_price,
                    margin=decision.margin,
                    leverage=decision.leverage,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                )
                self._trade_count += 1
                logger.info(f"OPEN {decision.direction}: {self.symbol} @ ${current_price:.2f}, {decision.leverage}x, margin=${decision.margin:.2f}")

            elif decision.action.startswith("CLOSE_"):
                for pos in open_positions:
                    if pos.direction == decision.direction:
                        closed = self.futures_db.close_position(pos.id, current_price, "SIGNAL")
                        if closed:
                            self._trade_count += 1
                            self._total_pnl += closed.realized_pnl
                            logger.info(f"CLOSE {decision.direction}: {self.symbol} @ ${current_price:.2f}, PnL=${closed.realized_pnl:.2f}")

            else:
                logger.info(f"HOLD: {self.symbol} @ ${current_price:.2f} | {decision.reasoning[:80]}")

            self.bot_db.log_decision(
                session_id=self.session_id,
                symbol=self.symbol,
                action=decision.action,
                amount_usd=decision.margin,
                qty=decision.position_size,
                price=current_price,
                ml_signal=decision.ml_signal,
                ml_confidence=decision.confidence,
                mc_var=decision.mc_var,
                mc_expected_return=decision.mc_expected_return,
                regime=decision.regime,
                rsi=stats.get("rsi", 50),
                trend=stats.get("trend_strength", 0),
                reasoning=f"[FUTURES {decision.leverage}x] {decision.reasoning}",
                trade_id="",
            )

            return decision

        except Exception as e:
            logger.error(f"Futures cycle error: {e}", exc_info=True)
            return None

    async def run(self) -> None:
        """Run the futures trading loop."""
        self.is_running = True
        logger.info(f"Futures trader started: {self.symbol}, session={self.session_id}, max_leverage={self.max_leverage}x")

        session = self.futures_db.get_session(self.session_id)
        if not session:
            logger.info("Creating new futures session...")
            self.futures_db.create_session(initial_margin=100000, session_id=self.session_id)

        while self.is_running:
            await self._run_cycle()
            await asyncio.sleep(self.check_interval)

    def stop(self):
        self.is_running = False
        logger.info(f"Futures trader stopped. Trades: {self._trade_count}, Total PnL: ${self._total_pnl:.2f}")


async def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Run futures trading bot")
    parser.add_argument("--session", type=str, default="futures-default")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--margin", type=float, default=100.0)
    parser.add_argument("--leverage", type=int, default=50)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    bot = FuturesTrader(
        session_id=args.session,
        symbol=args.symbol,
        base_margin=args.margin,
        max_leverage=args.leverage,
        check_interval=args.interval,
    )
    try:
        await bot.run()
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
