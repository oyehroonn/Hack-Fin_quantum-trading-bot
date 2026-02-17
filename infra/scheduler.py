"""Task scheduler for periodic trading system operations.

Schedules:
  - Data ingestion (daily/hourly)
  - Model retraining (weekly)
  - Walk-forward evaluation (daily)
  - Drift detection (daily)
  - Report generation (daily)
  - Champion/challenger evaluation (weekly)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Optional

from loguru import logger


class ScheduledTask:
    """A single scheduled task."""

    def __init__(
        self,
        name: str,
        func: Callable[..., Coroutine],
        interval_seconds: int,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        run_immediately: bool = False,
    ) -> None:
        self.name = name
        self.func = func
        self.interval_seconds = interval_seconds
        self.args = args
        self.kwargs = kwargs or {}
        self.run_immediately = run_immediately
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.run_count: int = 0
        self.error_count: int = 0
        self.is_running: bool = False

    async def execute(self) -> bool:
        """Execute the task. Returns True on success."""
        self.is_running = True
        try:
            await self.func(*self.args, **self.kwargs)
            self.last_run = datetime.now()
            self.run_count += 1
            self.next_run = self.last_run + timedelta(seconds=self.interval_seconds)
            logger.info(f"Task '{self.name}' completed (run #{self.run_count})")
            return True
        except Exception as e:
            self.error_count += 1
            logger.error(f"Task '{self.name}' failed (error #{self.error_count}): {e}")
            return False
        finally:
            self.is_running = False


class TaskScheduler:
    """Simple async task scheduler for the trading system."""

    def __init__(self) -> None:
        self.tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_task(
        self,
        name: str,
        func: Callable[..., Coroutine],
        interval_seconds: int,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        run_immediately: bool = False,
    ) -> None:
        """Register a periodic task.

        Args:
            name: Unique task name
            func: Async function to call
            interval_seconds: Seconds between runs
            args: Positional args to pass
            kwargs: Keyword args to pass
            run_immediately: Run once immediately on start
        """
        task = ScheduledTask(
            name=name,
            func=func,
            interval_seconds=interval_seconds,
            args=args,
            kwargs=kwargs,
            run_immediately=run_immediately,
        )
        self.tasks[name] = task
        logger.info(f"Scheduled task '{name}' every {interval_seconds}s")

    def remove_task(self, name: str) -> None:
        """Remove a task."""
        self.tasks.pop(name, None)

    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        logger.info(f"Scheduler started with {len(self.tasks)} tasks")

        # Run immediate tasks
        for task in self.tasks.values():
            if task.run_immediately:
                await task.execute()

        # Main loop
        while self._running:
            now = datetime.now()

            for task in self.tasks.values():
                if task.is_running:
                    continue

                should_run = False
                if task.last_run is None:
                    should_run = True
                elif task.next_run and now >= task.next_run:
                    should_run = True

                if should_run:
                    asyncio.create_task(task.execute())

            await asyncio.sleep(1)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        logger.info("Scheduler stopped")

    def status(self) -> list[dict[str, Any]]:
        """Get status of all tasks."""
        return [
            {
                "name": t.name,
                "interval_seconds": t.interval_seconds,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "next_run": t.next_run.isoformat() if t.next_run else None,
                "run_count": t.run_count,
                "error_count": t.error_count,
                "is_running": t.is_running,
            }
            for t in self.tasks.values()
        ]


def create_default_scheduler() -> TaskScheduler:
    """Create a scheduler with default trading system tasks.

    This is a factory that creates placeholder tasks.
    Real implementations should replace these with actual functions.
    """
    scheduler = TaskScheduler()

    async def _placeholder(task_name: str) -> None:
        logger.debug(f"Placeholder task: {task_name}")

    scheduler.add_task(
        name="data_ingest",
        func=_placeholder, args=("data_ingest",),
        interval_seconds=3600,  # Hourly
    )

    scheduler.add_task(
        name="drift_detection",
        func=_placeholder, args=("drift_detection",),
        interval_seconds=86400,  # Daily
    )

    scheduler.add_task(
        name="model_evaluation",
        func=_placeholder, args=("model_evaluation",),
        interval_seconds=86400,  # Daily
    )

    scheduler.add_task(
        name="model_retrain",
        func=_placeholder, args=("model_retrain",),
        interval_seconds=604800,  # Weekly
    )

    scheduler.add_task(
        name="daily_report",
        func=_placeholder, args=("daily_report",),
        interval_seconds=86400,  # Daily
    )

    return scheduler
