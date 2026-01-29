"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from infra.config import TradingConfig, load_config


def test_default_config() -> None:
    """Test default configuration."""
    config = TradingConfig()
    assert config.initial_cash == 100000.0
    assert config.max_position_size == 10000.0
    assert config.max_leverage == 2.0


def test_config_from_yaml() -> None:
    """Test loading configuration from YAML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_data = {
            "initial_cash": 200000.0,
            "max_position_size": 20000.0,
            "max_leverage": 3.0,
            "symbols": ["TSLA", "NVDA"],
        }
        yaml.dump(yaml_data, f)
        yaml_path = f.name

    try:
        config = TradingConfig.from_yaml(yaml_path)
        assert config.initial_cash == 200000.0
        assert config.max_position_size == 20000.0
        assert config.max_leverage == 3.0
        assert config.symbols == ["TSLA", "NVDA"]
    finally:
        os.unlink(yaml_path)


def test_config_env_override() -> None:
    """Test environment variable overrides."""
    # Set environment variable
    os.environ["TRADING_INITIAL_CASH"] = "300000.0"
    os.environ["TRADING_MAX_LEVERAGE"] = "4.0"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_data = {
                "initial_cash": 200000.0,
                "max_leverage": 3.0,
            }
            yaml.dump(yaml_data, f)
            yaml_path = f.name

        try:
            config = TradingConfig.from_yaml(yaml_path)
            # Environment should override YAML
            assert config.initial_cash == 300000.0
            assert config.max_leverage == 4.0
        finally:
            os.unlink(yaml_path)
    finally:
        # Clean up environment
        os.environ.pop("TRADING_INITIAL_CASH", None)
        os.environ.pop("TRADING_MAX_LEVERAGE", None)


def test_load_config() -> None:
    """Test load_config function."""
    # Test with non-existent file (should return default)
    config = load_config(Path("/nonexistent/path.yaml"))
    assert isinstance(config, TradingConfig)
