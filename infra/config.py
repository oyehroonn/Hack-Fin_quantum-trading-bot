"""Configuration management with YAML and environment variable overrides."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingConfig(BaseSettings):
    """Trading system configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Data settings
    data_dir: str = Field(default="data/", description="Data directory")
    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])

    # Trading settings
    initial_cash: float = Field(default=100000.0, description="Initial cash")
    max_position_size: float = Field(default=10000.0, description="Max position size")
    max_leverage: float = Field(default=2.0, description="Max leverage")

    # Risk settings
    max_drawdown: float = Field(default=0.2, description="Max drawdown (20%)")
    stop_loss_pct: float = Field(default=0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(default=0.05, description="Take profit percentage")

    # Execution settings
    slippage_bps: float = Field(default=5.0, description="Slippage in basis points")
    commission_bps: float = Field(default=1.0, description="Commission in basis points")

    # Logging settings
    log_level: str = Field(default="INFO", description="Log level")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TradingConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Environment variables override YAML
        env_overrides: dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith("TRADING_"):
                config_key = key.replace("TRADING_", "").lower()
                # Try to parse as appropriate type
                if isinstance(value, str):
                    if value.lower() in ("true", "false"):
                        env_overrides[config_key] = value.lower() == "true"
                    elif value.replace(".", "", 1).isdigit():
                        if "." in value:
                            env_overrides[config_key] = float(value)
                        else:
                            env_overrides[config_key] = int(value)
                    elif value.startswith("[") and value.endswith("]"):
                        # Simple list parsing
                        env_overrides[config_key] = [
                            item.strip().strip('"').strip("'")
                            for item in value[1:-1].split(",")
                            if item.strip()
                        ]
                    else:
                        env_overrides[config_key] = value

        merged = {**yaml_data, **env_overrides}
        return cls(**merged)


def load_config(config_path: Optional[str | Path] = None) -> TradingConfig:
    """Load configuration from file or environment."""
    if config_path is None:
        config_path = Path("config.yaml")
        if not config_path.exists():
            # Return default config if no file exists
            return TradingConfig()

    return TradingConfig.from_yaml(config_path)
