"""Autonomous trading bot: runs as a background loop, makes $100 trades using ML + Monte Carlo."""

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
from bot.bot_db import BotDB
from bot.decision_engine import DecisionEngine, Decision
from bot.performance_tracker import compute_performance
from infra.terminal_db import TerminalDB


class AutonomousTrader:
    """Autonomous trading bot that runs continuously."""

    def __init__(
        self,
        session_id: str,
        symbol: str = "BTCUSDT",
        asset_class: str = "crypto",
        trade_amount: float = 100.0,
        check_interval: int = 300,
        mc_paths: int = 10000,
        mc_horizon: int = 7,
        performance_interval: int = 10,
    ):
        self.session_id = session_id
        self.symbol = symbol
        self.asset_class = asset_class
        self.trade_amount = trade_amount
        self.check_interval = check_interval
        self.mc_paths = mc_paths
        self.mc_horizon = mc_horizon
        self.performance_interval = performance_interval

        self.terminal_db = TerminalDB("data/terminal.db")
        self.bot_db = BotDB("data/terminal.db")
        self.engine = DecisionEngine(trade_amount=trade_amount)

        self.is_running = False
        self._trade_count = 0

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
        start = end - pd.Timedelta(days=2)
        df = await ingestor.fetch_ohlcv(symbol=self.symbol, timeframe="1h", start=start, end=end, limit=5)
        if df.empty:
            raise ValueError(f"Cannot get current price for {self.symbol}")
        return float(df["close"].iloc[-1])

    def _get_ml_signal(self, prices: np.ndarray) -> tuple[str, float]:
        """Simple ML signal based on SMA crossover + momentum.

        Placeholder: replace with trained XGBoost model once historical data is ingested.
        """
        if len(prices) < 30:
            return "HOLD", 0.5

        sma_fast = np.mean(prices[-10:])
        sma_slow = np.mean(prices[-30:])
        rsi = self._simple_rsi(prices, 14)
        momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0

        score = 0.5
        if sma_fast > sma_slow:
            score += 0.15
        else:
            score -= 0.15

        if rsi < 30:
            score += 0.1
        elif rsi > 70:
            score -= 0.1

        if momentum > 0.02:
            score += 0.1
        elif momentum < -0.02:
            score -= 0.1

        score = max(0, min(1, score))

        if score >= 0.6:
            return "BUY", score
        elif score <= 0.4:
            return "SELL", 1 - score
        return "HOLD", 0.5

    def _simple_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    async def _run_cycle(self) -> Optional[Decision]:
        """Run one decision cycle."""
        try:
            prices = await self._fetch_prices(days=90)
            current_price = float(prices[-1])

            ml_signal, ml_confidence = self._get_ml_signal(prices)

            mc_result = run_monte_carlo(
                symbol=self.symbol,
                prices=prices,
                horizon_days=self.mc_horizon,
                num_paths=self.mc_paths,
            )
            self.bot_db.save_simulation(mc_result.to_dict())

            stats = full_statistical_analysis(prices)

            positions = self.terminal_db.get_positions(self.session_id)
            position = None
            for p in positions:
                if p.symbol == self.symbol:
                    position = {"qty": p.qty, "avg_cost": p.avg_cost}
                    break

            portfolio = self.terminal_db.get_portfolio(self.session_id)
            cash = portfolio.get("cash", 0)
            portfolio_value = portfolio.get("equity", cash)

            recent_trades = self.terminal_db.get_trades(self.session_id, symbol=self.symbol, limit=10)
            recent_trades_dicts = [
                {"side": t.side, "pnl": t.pnl, "price": t.price, "cost": t.cost, "qty": t.qty, "timestamp": t.timestamp}
                for t in recent_trades
            ]

            decision = self.engine.decide(
                ml_signal=ml_signal,
                ml_confidence=ml_confidence,
                mc_result=mc_result.to_dict(),
                stats=stats,
                position=position,
                cash=cash,
                portfolio_value=portfolio_value,
                recent_trades=recent_trades_dicts,
            )

            trade_id = ""
            exec_qty = 0
            if decision.action == "BUY":
                qty = decision.amount_usd / current_price
                decimals = 8 if self.asset_class == "crypto" else 4
                qty = round(qty, decimals)
                if qty > 0 and cash >= decision.amount_usd:
                    exec_qty = qty
                    trade = self.terminal_db.execute_trade(
                        session_id=self.session_id,
                        symbol=self.symbol,
                        side="BUY",
                        qty=qty,
                        price=current_price,
                        asset_class=self.asset_class,
                    )
                    trade_id = trade.id
                    self._trade_count += 1
                    logger.info(f"BOT BUY: {qty} {self.symbol} @ ${current_price:.2f} (${decision.amount_usd})")

            elif decision.action == "SELL" and position and position.get("qty", 0) > 0:
                exec_qty = position["qty"]
                trade = self.terminal_db.execute_trade(
                    session_id=self.session_id,
                    symbol=self.symbol,
                    side="SELL",
                    qty=position["qty"],
                    price=current_price,
                    asset_class=self.asset_class,
                )
                trade_id = trade.id
                self._trade_count += 1
                pnl = trade.pnl or 0
                logger.info(f"BOT SELL: {position['qty']} {self.symbol} @ ${current_price:.2f} (P&L: ${pnl:.2f})")

            else:
                logger.info(f"BOT HOLD: {self.symbol} @ ${current_price:.2f} — {decision.reasoning[:80]}")

            self.bot_db.log_decision(
                session_id=self.session_id,
                symbol=self.symbol,
                action=decision.action,
                amount_usd=decision.amount_usd,
                qty=exec_qty,
                price=current_price,
                ml_signal=decision.ml_signal,
                ml_confidence=decision.ml_confidence,
                mc_var=decision.mc_var,
                mc_expected_return=decision.mc_expected_return,
                regime=decision.regime,
                rsi=decision.rsi,
                trend=decision.trend_strength,
                reasoning=decision.reasoning,
                trade_id=trade_id,
            )

            if self._trade_count > 0 and self._trade_count % self.performance_interval == 0:
                all_trades = self.terminal_db.get_trades(self.session_id, limit=10000)
                trade_dicts = [{"side": t.side, "pnl": t.pnl, "cost": t.cost, "price": t.price, "qty": t.qty, "timestamp": t.timestamp} for t in all_trades]
                perf = compute_performance(trade_dicts, portfolio_value=portfolio_value, initial_cash=portfolio.get("initial_cash", 100000))
                self.bot_db.save_performance(self.session_id, perf)
                logger.info(f"Performance snapshot: P&L=${perf['total_pnl']:.2f}, Win rate={perf['win_rate']:.1%}, Trades={perf['total_trades']}")

            return decision

        except Exception as e:
            logger.error(f"Bot cycle error: {e}", exc_info=True)
            return None

    async def run(self) -> None:
        """Run the autonomous trading loop."""
        self.is_running = True
        logger.info(f"Autonomous trader started: {self.symbol}, session={self.session_id}, amount=${self.trade_amount}")

        sess = self.terminal_db.get_session(self.session_id)
        if not sess:
            logger.info("Creating new session for bot...")
            sess = self.terminal_db.create_session(initial_cash=100000, session_id=self.session_id)

        while self.is_running:
            await self._run_cycle()
            await asyncio.sleep(self.check_interval)

    def stop(self):
        self.is_running = False
        logger.info("Autonomous trader stopped")


# --- Standalone entry point ---
async def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Run autonomous trading bot")
    parser.add_argument("--session", type=str, default="bot-default")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--amount", type=float, default=100.0)
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    args = parser.parse_args()

    bot = AutonomousTrader(
        session_id=args.session,
        symbol=args.symbol,
        trade_amount=args.amount,
        check_interval=args.interval,
    )
    try:
        await bot.run()
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
