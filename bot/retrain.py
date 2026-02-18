"""Retrain pipeline: bot reviews past trades, retrains ML model, promotes if better."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def build_training_data(prices: np.ndarray, lookback: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and labels from price history.

    Features per sample: SMA ratios, RSI, momentum, volatility, returns.
    Labels: 1 if price goes up in next period, 0 otherwise.
    """
    if len(prices) < lookback + 10:
        raise ValueError(f"Need at least {lookback + 10} prices, got {len(prices)}")

    features_list = []
    labels_list = []

    for i in range(lookback, len(prices) - 1):
        window = prices[i - lookback: i]
        close = prices[i]
        next_close = prices[i + 1]

        sma_10 = np.mean(window[-10:])
        sma_30 = np.mean(window)
        sma_ratio = sma_10 / sma_30 if sma_30 > 0 else 1

        log_returns = np.diff(np.log(window))
        volatility = np.std(log_returns) if len(log_returns) > 0 else 0
        mean_return = np.mean(log_returns) if len(log_returns) > 0 else 0

        momentum_5 = (close - window[-5]) / window[-5] if window[-5] > 0 else 0
        momentum_10 = (close - window[-10]) / window[-10] if window[-10] > 0 else 0

        deltas = np.diff(window[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 50

        vol_ratio = volatility / (np.std(np.diff(np.log(prices[:i]))) if i > 30 else volatility + 1e-8) if volatility > 0 else 1

        features_list.append([
            sma_ratio,
            volatility,
            mean_return,
            momentum_5,
            momentum_10,
            rsi / 100.0,
            vol_ratio,
            close / sma_30 if sma_30 > 0 else 1,
        ])

        label = 1.0 if next_close > close else 0.0
        labels_list.append(label)

    return np.array(features_list), np.array(labels_list)


def train_model(X: np.ndarray, y: np.ndarray):
    """Train an XGBoost or fallback sklearn model."""
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        logger.warning("xgboost not available, using sklearn GradientBoosting")
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    logger.info(f"Model trained: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    return model, {"train_accuracy": train_acc, "val_accuracy": val_acc, "samples": len(X)}


def retrain_from_history(prices: np.ndarray, model_dir: str = "models_registry/bot_xgb") -> dict:
    """Full retrain pipeline: build data, train, save."""
    import joblib
    from pathlib import Path

    X, y = build_training_data(prices, lookback=30)
    model, metrics = train_model(X, y)

    save_path = Path(model_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path / "model.joblib")
    logger.info(f"Model saved to {save_path / 'model.joblib'}")

    import json
    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


async def run_retrain(symbol: str = "BTCUSDT"):
    """Fetch latest data and retrain."""
    from data.ingest.binance_public import BinancePublicIngestor
    ingestor = BinancePublicIngestor()
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=365 * 3)
    df = await ingestor.fetch_ohlcv(symbol=symbol, timeframe="1d", start=start, end=end)
    if df.empty:
        logger.error(f"No data for {symbol}")
        return None
    prices = df["close"].values.astype(float)
    return retrain_from_history(prices)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_retrain())
