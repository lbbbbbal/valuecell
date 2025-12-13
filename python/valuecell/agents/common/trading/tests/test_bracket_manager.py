import pytest

from valuecell.agents.common.trading.execution.bracket_manager import (
    BracketOrderManager,
    ExitOrderPlan,
    ExitReconcilePlan,
    FillEvent,
    OpenOrderState,
)
from valuecell.agents.common.trading.models import (
    ExitOrdersSpec,
    ExitQtyMode,
    InstrumentRef,
    PositionSnapshot,
    StopLossSpec,
    TakeProfitSpec,
    TradeSide,
)


@pytest.mark.asyncio
async def test_exit_orders_created_with_reduce_only_close_position():
    position = PositionSnapshot(
        instrument=InstrumentRef(symbol="BTCUSDT"), quantity=1.0
    )
    exits = ExitOrdersSpec(
        stop_loss=StopLossSpec(trigger_price=45000, qty_mode=ExitQtyMode.CLOSE_POSITION),
        take_profit=TakeProfitSpec(
            trigger_price=52000, qty_mode=ExitQtyMode.CLOSE_POSITION
        ),
    )
    mgr = BracketOrderManager(strategy_id="s1")

    plan = mgr.build_exit_plan(
        cycle_ts=1, position=position, decision_exits=exits, open_orders=[]
    )

    assert plan == ExitReconcilePlan(create=plan.create, cancel=[])
    assert len(plan.create) == 2
    for order in plan.create:
        assert order.reduce_only is True
        assert order.close_position is True
        assert order.quantity is None


@pytest.mark.asyncio
async def test_tp_fill_cancels_stop_loss():
    position = PositionSnapshot(
        instrument=InstrumentRef(symbol="BTCUSDT"), quantity=1.0
    )
    exits = ExitOrdersSpec(
        stop_loss=StopLossSpec(trigger_price=44000, qty_mode=ExitQtyMode.CLOSE_POSITION),
        take_profit=TakeProfitSpec(
            trigger_price=51000, qty_mode=ExitQtyMode.CLOSE_POSITION
        ),
    )
    mgr = BracketOrderManager(strategy_id="s1")
    base_plan = mgr.build_exit_plan(
        cycle_ts=1, position=position, decision_exits=exits, open_orders=[]
    )
    tp_order = next(o for o in base_plan.create if o.purpose == "tp")

    fill = FillEvent(
        symbol="BTCUSDT",
        qty=1.0,
        price=51000,
        client_order_id=tp_order.client_order_id,
    )
    plan = mgr.build_exit_plan(
        cycle_ts=1,
        position=position,
        decision_exits=exits,
        open_orders=[OpenOrderState(**tp_order.__dict__)],
        fills=[fill],
    )

    assert plan.cancel == [cid for cid in plan.cancel if cid]
    assert tp_order.client_order_id not in plan.cancel
    assert any(cid.endswith("sl:sell") for cid in plan.cancel)


@pytest.mark.asyncio
async def test_sl_fill_cancels_take_profit():
    position = PositionSnapshot(
        instrument=InstrumentRef(symbol="BTCUSDT"), quantity=-1.0
    )
    exits = ExitOrdersSpec(
        stop_loss=StopLossSpec(trigger_price=52000, qty_mode=ExitQtyMode.CLOSE_POSITION),
        take_profit=TakeProfitSpec(
            trigger_price=48000, qty_mode=ExitQtyMode.CLOSE_POSITION
        ),
    )
    mgr = BracketOrderManager(strategy_id="s1")
    base_plan = mgr.build_exit_plan(
        cycle_ts=2, position=position, decision_exits=exits, open_orders=[]
    )
    sl_order = next(o for o in base_plan.create if o.purpose == "sl")

    fill = FillEvent(
        symbol="BTCUSDT",
        qty=1.0,
        price=52000,
        client_order_id=sl_order.client_order_id,
    )
    plan = mgr.build_exit_plan(
        cycle_ts=2,
        position=position,
        decision_exits=exits,
        open_orders=[OpenOrderState(**sl_order.__dict__)],
        fills=[fill],
    )

    assert any(cid.endswith("tp:buy") for cid in plan.cancel)


@pytest.mark.asyncio
async def test_fill_between_snapshots_skips_invalid_reorders():
    position = PositionSnapshot(
        instrument=InstrumentRef(symbol="BTCUSDT"), quantity=0.0
    )
    open_orders = [
        OpenOrderState(
            client_order_id="s1:BTCUSDT:1:tp:sell",
            symbol="BTCUSDT",
            side=TradeSide.SELL,
            reduce_only=True,
        )
    ]
    mgr = BracketOrderManager(strategy_id="s1")
    plan = mgr.build_exit_plan(
        cycle_ts=1,
        position=position,
        decision_exits=None,
        open_orders=open_orders,
        fills=[],
    )

    assert plan.create == []
    assert plan.cancel == ["s1:BTCUSDT:1:tp:sell"]


@pytest.mark.asyncio
async def test_idempotent_exit_orders_skip_duplicates():
    position = PositionSnapshot(
        instrument=InstrumentRef(symbol="BTCUSDT"), quantity=1.5
    )
    exits = ExitOrdersSpec(
        take_profit=TakeProfitSpec(
            trigger_price=50000, qty_mode=ExitQtyMode.PARTIAL, qty=1.0
        )
    )
    mgr = BracketOrderManager(strategy_id="s1")
    base_plan = mgr.build_exit_plan(
        cycle_ts=3, position=position, decision_exits=exits, open_orders=[]
    )
    tp_order = next(o for o in base_plan.create if o.purpose == "tp")

    duplicated = mgr.build_exit_plan(
        cycle_ts=3,
        position=position,
        decision_exits=exits,
        open_orders=[
            OpenOrderState(
                client_order_id=tp_order.client_order_id,
                symbol=tp_order.symbol,
                side=tp_order.side,
                type=tp_order.type,
                stop_price=tp_order.stop_price,
                price=tp_order.price,
                quantity=tp_order.quantity,
                reduce_only=True,
                close_position=False,
            )
        ],
        fills=[],
    )

    assert duplicated.create == []
    assert duplicated.cancel == []
