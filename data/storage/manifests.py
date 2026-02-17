"""Data manifests: metadata tracking for ingested datasets.

Tracks what data has been ingested, data quality stats, and
time ranges for each symbol/timeframe combination.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class DataManifest:
    """Metadata manifest for ingested data.

    Tracks:
      - Symbols and timeframes with their date ranges
      - Data quality stats per dataset
      - Last ingestion timestamps
      - Row counts and file sizes
    """

    def __init__(self, manifest_path: str = "data_manifest.json") -> None:
        self.manifest_path = Path(manifest_path)
        self._data: dict[str, Any] = self._load()

    def record_ingestion(
        self,
        symbol: str,
        asset_class: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        num_rows: int,
        file_path: Optional[str] = None,
        quality_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a data ingestion event."""
        key = f"{asset_class}/{symbol}/{timeframe}"

        if key not in self._data:
            self._data[key] = {
                "symbol": symbol,
                "asset_class": asset_class,
                "timeframe": timeframe,
                "first_ingested": datetime.now().isoformat(),
                "date_ranges": [],
                "total_rows": 0,
                "files": [],
                "quality": {},
            }

        entry = self._data[key]
        entry["last_updated"] = datetime.now().isoformat()
        entry["date_ranges"].append({
            "start": start_date,
            "end": end_date,
            "rows": num_rows,
            "ingested_at": datetime.now().isoformat(),
        })
        entry["total_rows"] += num_rows

        if file_path:
            entry["files"].append(file_path)

        if quality_stats:
            entry["quality"] = quality_stats

        self._save()
        logger.debug(f"Manifest updated: {key} ({num_rows} rows, {start_date} to {end_date})")

    def get_coverage(self, symbol: str, asset_class: str, timeframe: str) -> Optional[dict[str, Any]]:
        """Get data coverage for a symbol."""
        key = f"{asset_class}/{symbol}/{timeframe}"
        return self._data.get(key)

    def list_symbols(self, asset_class: Optional[str] = None) -> list[dict[str, Any]]:
        """List all symbols with their coverage."""
        results = []
        for key, info in self._data.items():
            if asset_class and info.get("asset_class") != asset_class:
                continue
            results.append({
                "key": key,
                "symbol": info["symbol"],
                "asset_class": info["asset_class"],
                "timeframe": info["timeframe"],
                "total_rows": info.get("total_rows", 0),
                "last_updated": info.get("last_updated", ""),
            })
        return results

    def get_gap_report(self, symbol: str, asset_class: str, timeframe: str) -> list[dict[str, str]]:
        """Identify gaps in data coverage."""
        coverage = self.get_coverage(symbol, asset_class, timeframe)
        if not coverage:
            return []

        ranges = sorted(coverage.get("date_ranges", []), key=lambda x: x["start"])
        gaps = []

        for i in range(1, len(ranges)):
            prev_end = ranges[i - 1]["end"]
            curr_start = ranges[i]["start"]
            if prev_end < curr_start:
                gaps.append({"gap_start": prev_end, "gap_end": curr_start})

        return gaps

    def _load(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
