"""
Token budget — tracks LLM spend and enforces limits.

The budget system works in two dimensions:
1. **Token budget**: daily/hourly cap on total tokens consumed
2. **Throughput**: max concurrent or per-minute LLM calls (for local LLMs)

Both dimensions can independently throttle the router. The budget
is queryable so the router can make informed decisions about whether
a given decision is worth the token spend.

All state is plain data — the budget can be serialised, inspected,
and reset without side effects.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BudgetSnapshot:
    """Current state of the budget — returned by get_snapshot()."""
    tokens_used: int
    tokens_remaining: int
    tokens_limit: int
    calls_used: int
    budget_pressure: float    # 0.0 = flush, 1.0 = depleted
    reserve_available: bool   # True if reserve tokens still available
    throughput_available: bool # True if within throughput limits


class TokenBudget:
    """
    Tracks token consumption and enforces configurable limits.

    The budget resets on a configurable period (default: daily).
    A reserve fraction is held back for high-value decisions
    (e.g. conversations) so routine decisions can't starve them.

    Pluggable: subclass and override can_spend() to implement
    custom budget strategies (e.g. burst budgets, priority queues).
    """

    def __init__(
        self,
        daily_limit: int = 500_000,
        reserve_fraction: float = 0.2,
        max_calls_per_minute: int = 50,
        max_concurrent: int = 10,
    ) -> None:
        self.daily_limit = daily_limit
        self.reserve_fraction = reserve_fraction
        self.max_calls_per_minute = max_calls_per_minute
        self.max_concurrent = max_concurrent

        # Tracking state
        self._tokens_used: int = 0
        self._calls_total: int = 0
        self._call_timestamps: list[float] = []
        self._active_calls: int = 0
        self._reset_time: float = time.time()

        # Per-purpose tracking
        self._tokens_by_purpose: dict[str, int] = {}

    # ---------- Core API ----------

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.daily_limit - self._tokens_used)

    @property
    def unreserved_remaining(self) -> int:
        """Tokens available for non-priority decisions."""
        reserve = int(self.daily_limit * self.reserve_fraction)
        return max(0, self.tokens_remaining - reserve)

    @property
    def budget_pressure(self) -> float:
        """0.0 = flush (full budget), 1.0 = depleted (no tokens left)."""
        if self.daily_limit <= 0:
            return 0.0  # unlimited budget
        return min(1.0, self._tokens_used / self.daily_limit)

    def can_spend(
        self,
        estimated_tokens: int,
        is_priority: bool = False,
    ) -> bool:
        """
        Check if a spend of estimated_tokens is allowed.

        Priority calls can dip into the reserve. Non-priority
        calls are capped at unreserved_remaining.

        Override this method for custom budget strategies.
        """
        if self.daily_limit <= 0:
            return True  # unlimited (local LLM)

        if is_priority:
            return self.tokens_remaining >= estimated_tokens
        return self.unreserved_remaining >= estimated_tokens

    def can_call(self) -> bool:
        """Check throughput limits (calls/minute and concurrent)."""
        if self._active_calls >= self.max_concurrent:
            return False
        self._prune_timestamps()
        return len(self._call_timestamps) < self.max_calls_per_minute

    def record_spend(
        self,
        tokens: int,
        purpose: str = "general",
    ) -> None:
        """Record a completed LLM call's token usage."""
        self._tokens_used += tokens
        self._calls_total += 1
        self._call_timestamps.append(time.time())
        self._tokens_by_purpose[purpose] = (
            self._tokens_by_purpose.get(purpose, 0) + tokens
        )

    def begin_call(self) -> None:
        """Mark a call as in-flight (for concurrency tracking)."""
        self._active_calls += 1

    def end_call(self) -> None:
        """Mark a call as complete."""
        self._active_calls = max(0, self._active_calls - 1)

    def get_snapshot(self) -> BudgetSnapshot:
        """Get a read-only snapshot of current budget state."""
        return BudgetSnapshot(
            tokens_used=self._tokens_used,
            tokens_remaining=self.tokens_remaining,
            tokens_limit=self.daily_limit,
            calls_used=self._calls_total,
            budget_pressure=self.budget_pressure,
            reserve_available=self.tokens_remaining > int(
                self.daily_limit * self.reserve_fraction
            ) if self.daily_limit > 0 else True,
            throughput_available=self.can_call(),
        )

    # ---------- Reset and stats ----------

    def reset(self) -> None:
        """Reset the budget (e.g. at the start of a new day)."""
        self._tokens_used = 0
        self._calls_total = 0
        self._call_timestamps.clear()
        self._active_calls = 0
        self._tokens_by_purpose.clear()
        self._reset_time = time.time()
        logger.info("Token budget reset (limit: %d)", self.daily_limit)

    def check_auto_reset(self, period_seconds: float = 86400) -> bool:
        """Auto-reset if the period has elapsed. Returns True if reset."""
        if time.time() - self._reset_time >= period_seconds:
            self.reset()
            return True
        return False

    def get_stats(self) -> dict:
        return {
            "tokens_used": self._tokens_used,
            "tokens_remaining": self.tokens_remaining,
            "daily_limit": self.daily_limit,
            "budget_pressure": round(self.budget_pressure, 3),
            "calls_total": self._calls_total,
            "active_calls": self._active_calls,
            "by_purpose": dict(self._tokens_by_purpose),
        }

    # ---------- Internals ----------

    def _prune_timestamps(self) -> None:
        """Remove call timestamps older than 60 seconds."""
        cutoff = time.time() - 60
        self._call_timestamps = [
            t for t in self._call_timestamps if t > cutoff
        ]
