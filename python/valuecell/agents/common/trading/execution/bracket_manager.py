from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from loguru import logger

from valuecell.agents.common.trading.models import (
    ExitOrdersSpec,
    ExitQtyMode,
    PositionSnapshot,
    StopLossSpec,
    TakeProfitSpec,
    TradeSide,
)


@dataclass
class OpenOrderState:
    """Minimal open order state used for reconciliation."""

    client_order_id: str
    symbol: str
    side: TradeSide
    type: Optional[str] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None
    quantity: Optional[float] = None
    reduce_only: bool = False
    close_position: bool = False
    purpose: Optional[str] = None


@dataclass
class FillEvent:
    """Fill event from userTrades/websocket."""

    symbol: str
    qty: float
    price: float
    client_order_id: Optional[str] = None


@dataclass
class ExitOrderPlan:
    """Planned exit order submission."""

    client_order_id: str
    symbol: str
    side: TradeSide
    type: str
    stop_price: Optional[float]
    price: Optional[float]
    quantity: Optional[float]
    reduce_only: bool
    close_position: bool
    purpose: str


@dataclass
class ExitReconcilePlan:
    """Final reconcile result containing create/cancel sets."""

    create: List[ExitOrderPlan]
    cancel: List[str]


class BracketOrderManager:
    """Constructs and reconciles stop-loss / take-profit orders."""

    def __init__(self, *, strategy_id: str, quantity_precision: float = 1e-9) -> None:
        self._strategy_id = strategy_id
        self._quantity_precision = quantity_precision

    def build_exit_plan(
        self,
        *,
        cycle_ts: int,
        position: PositionSnapshot,
        decision_exits: Optional[ExitOrdersSpec],
        open_orders: Sequence[OpenOrderState] | None = None,
        fills: Iterable[FillEvent] | None = None,
    ) -> ExitReconcilePlan:
        """Return cancel/create instructions respecting idempotency and fills."""

        open_map = {order.client_order_id: order for order in open_orders or []}
        fills = list(fills or [])

        if abs(position.quantity) <= self._quantity_precision:
            # position closed: cancel remaining reduceOnly exits
            cancel_ids = [cid for cid, order in open_map.items() if order.reduce_only]
            return ExitReconcilePlan(create=[], cancel=cancel_ids)

        side = TradeSide.SELL if position.quantity > 0 else TradeSide.BUY
        create: List[ExitOrderPlan] = []
        cancel: List[str] = []

        tp_id, sl_id = self._client_ids(position.instrument.symbol, cycle_ts, side)
        # sibling cancellation if one filled
        for fill in fills:
            cid = fill.client_order_id or ""
            if cid and cid.startswith(tp_id.rsplit(":", 1)[0]):
                cancel.append(sl_id)
            if cid and cid.startswith(sl_id.rsplit(":", 1)[0]):
                cancel.append(tp_id)

        if decision_exits is None:
            return ExitReconcilePlan(create=create, cancel=cancel)

        create.extend(
            self._maybe_create_order(
                symbol=position.instrument.symbol,
                side=side,
                spec=decision_exits.take_profit,
                purpose="tp",
                client_order_id=tp_id,
                position_qty=abs(position.quantity),
                existing=open_map.get(tp_id),
            )
        )
        create.extend(
            self._maybe_create_order(
                symbol=position.instrument.symbol,
                side=side,
                spec=decision_exits.stop_loss,
                purpose="sl",
                client_order_id=sl_id,
                position_qty=abs(position.quantity),
                existing=open_map.get(sl_id),
            )
        )

        # If TP created and SL filled (or vice versa), ensure sibling cancellation not duplicated
        cancel = list({cid for cid in cancel if cid})
        return ExitReconcilePlan(create=create, cancel=cancel)

    def _client_ids(self, symbol: str, cycle_ts: int, side: TradeSide) -> tuple[str, str]:
        prefix = f"{self._strategy_id}:{symbol}:{cycle_ts}"
        tp = f"{prefix}:tp:{side.value.lower()}"
        sl = f"{prefix}:sl:{side.value.lower()}"
        return tp, sl

    def _maybe_create_order(
        self,
        *,
        symbol: str,
        side: TradeSide,
        spec: Optional[StopLossSpec | TakeProfitSpec],
        purpose: str,
        client_order_id: str,
        position_qty: float,
        existing: Optional[OpenOrderState],
    ) -> List[ExitOrderPlan]:
        if spec is None:
            return []

        qty = self._resolve_quantity(spec, position_qty)
        close_position = spec.qty_mode == ExitQtyMode.CLOSE_POSITION
        stop_price = getattr(spec, "trigger_price", None)
        limit_price = getattr(spec, "price", None)

        if existing and self._is_same(existing, spec, qty, close_position):
            logger.debug("Skipping duplicate exit order {}", client_order_id)
            return []

        return [
            ExitOrderPlan(
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                type=spec.type,
                stop_price=stop_price,
                price=limit_price,
                quantity=None if close_position else qty,
                reduce_only=True,
                close_position=close_position,
                purpose=purpose,
            )
        ]

    def _resolve_quantity(self, spec: StopLossSpec | TakeProfitSpec, position_qty: float) -> float:
        if spec.qty_mode == ExitQtyMode.CLOSE_POSITION:
            return position_qty
        return max(self._quantity_precision, min(position_qty, float(spec.qty or 0.0)))

    def _is_same(
        self,
        existing: OpenOrderState,
        spec: StopLossSpec | TakeProfitSpec,
        qty: float,
        close_position: bool,
    ) -> bool:
        if existing.type and existing.type != spec.type:
            return False
        if close_position != bool(existing.close_position):
            return False
        if not close_position and existing.quantity is not None:
            if abs(existing.quantity - qty) > self._quantity_precision:
                return False
        if getattr(spec, "trigger_price", None) is not None and existing.stop_price is not None:
            if abs(existing.stop_price - float(spec.trigger_price)) > 1e-8:
                return False
        if getattr(spec, "price", None) is not None and existing.price is not None:
            if abs(existing.price - float(spec.price)) > 1e-8:
                return False
        return True
