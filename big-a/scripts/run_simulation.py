#!/usr/bin/env python3
"""AI-driven simulated trading CLI."""
from __future__ import annotations

import random
import sys
from datetime import date, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import typer
from loguru import logger

from big_a.simulation.config import load_simulation_config
from big_a.simulation.engine import SimulationEngine
from big_a.simulation.storage import SimulationStorage
from big_a.simulation.types import (
    OrderSide,
    Portfolio,
    SignalSource,
    SignalStrength,
    SimulationConfig,
    StockSignal,
    TradingDecision,
)

app = typer.Typer(help="AI-driven simulated trading system")


def _try_rich():
    """Import Rich for formatted output if available."""
    try:
        from rich.console import Console
        from rich.table import Table
        return Console()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init(
    config: str = typer.Option(
        "configs/simulation/default.yaml",
        help="Simulation config path (relative to project root)",
    ),
    project_root: Path = typer.Option(
        None,
        help="Project root (default: script parent directory)",
    ),
) -> None:
    """Initialize simulation account."""
    root = project_root or SCRIPT_DIR.parent
    config_path = root / config

    try:
        sim_config = load_simulation_config(config_path)
    except FileNotFoundError:
        logger.warning(f"Config not found at {config_path}, using defaults")
        sim_config = SimulationConfig()

    storage = SimulationStorage(
        base_dir=str(root / sim_config.storage_base_dir),
        trades_dir=str(root / sim_config.storage_trades_dir),
        decisions_dir=str(root / sim_config.storage_decisions_dir),
        snapshots_dir=str(root / sim_config.storage_snapshots_dir),
    )
    storage.ensure_dirs()

    initial_portfolio = Portfolio(
        cash=sim_config.initial_capital,
        positions={},
        total_value=sim_config.initial_capital,
        daily_pnl=0.0,
        updated_at=datetime.now(),
    )
    today = date.today().isoformat()
    storage.save_snapshot(initial_portfolio, today)

    typer.echo(
        f"Initialized simulation account: {sim_config.account} "
        f"with capital {sim_config.initial_capital}"
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status(
    config: str = typer.Option(
        "configs/simulation/default.yaml",
        help="Simulation config path (relative to project root)",
    ),
    project_root: Path = typer.Option(
        None,
        help="Project root (default: script parent directory)",
    ),
) -> None:
    """Show current portfolio state."""
    root = project_root or SCRIPT_DIR.parent
    config_path = root / config

    try:
        sim_config = load_simulation_config(config_path)
    except FileNotFoundError:
        sim_config = SimulationConfig()

    storage = SimulationStorage(
        base_dir=str(root / sim_config.storage_base_dir),
        trades_dir=str(root / sim_config.storage_trades_dir),
        decisions_dir=str(root / sim_config.storage_decisions_dir),
        snapshots_dir=str(root / sim_config.storage_snapshots_dir),
    )

    portfolio = storage.load_latest_snapshot()
    if portfolio is None:
        typer.echo("No simulation data found. Run 'init' first.")
        raise typer.Exit(1)

    console = _try_rich()
    if console:
        _print_status_rich(console, sim_config, portfolio)
    else:
        _print_status_plain(sim_config, portfolio)


def _print_status_rich(console, sim_config: SimulationConfig, portfolio: Portfolio) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n[bold]Simulation Account:[/bold] {sim_config.account}")
    console.print(f"[bold]Initial Capital:[/bold] {sim_config.initial_capital:,.2f}")
    console.print(f"[bold]Current Cash:[/bold] {portfolio.cash:,.2f}")
    console.print(f"[bold]Total Portfolio Value:[/bold] {portfolio.total_value:,.2f}")

    total_pnl = portfolio.total_value - sim_config.initial_capital
    total_pnl_pct = (total_pnl / sim_config.initial_capital) * 100
    console.print(f"[bold]Total P&L:[/bold] {total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")

    if portfolio.positions:
        table = Table(title="Positions")
        table.add_column("Stock", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Price", justify="right", style="yellow")
        table.add_column("Current Price", justify="right", style="yellow")
        table.add_column("P&L", justify="right", style="green")
        table.add_column("P&L%", justify="right", style="green")

        for code, pos in portfolio.positions.items():
            pnl = (pos.current_price - pos.avg_price) * pos.quantity
            pnl_pct = pos.pnl_pct * 100
            table.add_row(
                code,
                str(pos.quantity),
                f"{pos.avg_price:.2f}",
                f"{pos.current_price:.2f}",
                f"{pnl:+,.2f}",
                f"{pnl_pct:+.2f}%",
            )
        console.print(table)
    else:
        console.print("[dim]No open positions[/dim]")


def _print_status_plain(sim_config: SimulationConfig, portfolio: Portfolio) -> None:
    print(f"\nSimulation Account: {sim_config.account}")
    print(f"Initial Capital: {sim_config.initial_capital:,.2f}")
    print(f"Current Cash: {portfolio.cash:,.2f}")
    print(f"Total Portfolio Value: {portfolio.total_value:,.2f}")

    total_pnl = portfolio.total_value - sim_config.initial_capital
    total_pnl_pct = (total_pnl / sim_config.initial_capital) * 100
    print(f"Total P&L: {total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")

    if portfolio.positions:
        print("\nPositions:")
        print(f"{'Stock':<10} {'Qty':>6} {'Avg Price':>12} {'Current':>12} {'P&L':>14} {'P&L%':>10}")
        print("-" * 70)
        for code, pos in portfolio.positions.items():
            pnl = (pos.current_price - pos.avg_price) * pos.quantity
            pnl_pct = pos.pnl_pct * 100
            print(
                f"{code:<10} {pos.quantity:>6} {pos.avg_price:>12.2f} "
                f"{pos.current_price:>12.2f} {pnl:>+14.2f} {pnl_pct:>+10.2f}%"
            )
    else:
        print("\nNo open positions")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    once: bool = typer.Option(
        True,
        "--once/--continuous",
        help="Run a single day and exit (default: True)",
    ),
    config: str = typer.Option(
        "configs/simulation/default.yaml",
        help="Simulation config path (relative to project root)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would happen without executing trades",
    ),
    project_root: Path = typer.Option(
        None,
        help="Project root (default: script parent directory)",
    ),
) -> None:
    """Run one or more simulation trading days."""
    root = project_root or SCRIPT_DIR.parent
    config_path = root / config

    try:
        sim_config = load_simulation_config(config_path)
    except FileNotFoundError:
        logger.warning(f"Config not found at {config_path}, using defaults")
        sim_config = SimulationConfig()

    storage = SimulationStorage(
        base_dir=str(root / sim_config.storage_base_dir),
        trades_dir=str(root / sim_config.storage_trades_dir),
        decisions_dir=str(root / sim_config.storage_decisions_dir),
        snapshots_dir=str(root / sim_config.storage_snapshots_dir),
    )

    today = date.today().isoformat()

    # Load latest snapshot or initialize if none exists
    portfolio = storage.load_latest_snapshot()
    if portfolio is None:
        logger.info("No snapshot found, initializing fresh account")
        initial_portfolio = Portfolio(
            cash=sim_config.initial_capital,
            positions={},
            total_value=sim_config.initial_capital,
            daily_pnl=0.0,
            updated_at=datetime.now(),
        )
        storage.save_snapshot(initial_portfolio, today)
        portfolio = initial_portfolio

    try:
        from big_a.broker.in_memory import InMemoryBroker

        broker = InMemoryBroker(
            initial_cash=portfolio.cash,
            open_cost=sim_config.open_cost,
            close_cost=sim_config.close_cost,
            min_commission=sim_config.min_commission,
            limit_threshold=sim_config.limit_threshold,
        )

        # Restore positions to broker
        for code, pos in portfolio.positions.items():
            from big_a.simulation.types import Position
            from datetime import date as date_type

            broker._positions[code] = Position(
                stock_code=pos.stock_code,
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                realized_pnl=pos.realized_pnl,
                entry_date=pos.entry_date or date_type.today().isoformat(),
            )
            broker._prices[code] = pos.current_price

        engine = SimulationEngine(sim_config, broker)

        if dry_run:
            _run_dry_run(engine, sim_config, storage, today)
        else:
            _run_once(engine, sim_config, storage, today)

    except Exception as e:
        logger.error(f"Simulation run failed: {e}")
        raise typer.Exit(1)


def _generate_mock_signals() -> list[StockSignal]:
    """Generate mock stock signals for self-contained testing."""
    stock_codes = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH", "000858.SZ"]
    selected = random.sample(stock_codes, min(3, len(stock_codes)))

    signals = []
    for code in selected:
        score = random.uniform(-0.3, 0.9)
        if score > 0.6:
            strength = SignalStrength.STRONG_BUY
        elif score > 0.2:
            strength = SignalStrength.BUY
        elif score > -0.2:
            strength = SignalStrength.HOLD
        elif score > -0.6:
            strength = SignalStrength.SELL
        else:
            strength = SignalStrength.STRONG_SELL

        signals.append(
            StockSignal(
                stock_code=code,
                score=score,
                signal=strength,
                source=SignalSource.fused,
                reasoning="Mock signal for CLI testing",
            )
        )
    return signals


def _run_once(
    engine: SimulationEngine,
    sim_config: SimulationConfig,
    storage: SimulationStorage,
    trading_date: str,
) -> None:
    """Execute a single simulation day with mock data."""
    signals = _generate_mock_signals()

    # Mock prices: all stocks at 100.0
    prices = {s.stock_code: 100.0 for s in signals}

    if not signals:
        typer.echo("No signals generated, nothing to do.")
        return

    portfolio_before = engine.get_portfolio()
    initial_capital = sim_config.initial_capital

    portfolio = engine.run_daily(trading_date, signals, prices)

    # Save snapshot
    storage.save_snapshot(portfolio, trading_date)

    # Save decision
    decision = TradingDecision(
        date=trading_date,
        signals=signals,
        orders=[],
        reasoning="Self-contained mock simulation",
    )
    storage.save_decision(decision, trading_date)

    # Collect trade records from broker
    from big_a.simulation.types import TradeRecord

    trade_records: list[TradeRecord] = []
    for order in engine.broker._orders:
        if order.status.value == "FILLED":
            trade_records.append(
                TradeRecord(
                    order_id=order.id,
                    stock_code=order.stock_code,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price=order.price,
                    commission=order.commission,
                    timestamp=order.filled_at or datetime.now(),
                )
            )
    if trade_records:
        storage.save_trades(trade_records, trading_date)

    # Print summary
    daily_pnl = portfolio.total_value - portfolio_before.total_value
    daily_pnl_pct = (daily_pnl / portfolio_before.total_value) * 100 if portfolio_before.total_value else 0

    console = _try_rich()
    if console:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"\n[bold]Simulation Run — {trading_date}[/bold]")
        console.print(f"Trades executed: {len(trade_records)}")
        console.print(f"Portfolio value: {portfolio.total_value:,.2f}")
        console.print(f"Daily P&L: {daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)")
    else:
        print(f"\nSimulation Run — {trading_date}")
        print(f"Trades executed: {len(trade_records)}")
        print(f"Portfolio value: {portfolio.total_value:,.2f}")
        print(f"Daily P&L: {daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)")


def _run_dry_run(
    engine: SimulationEngine,
    sim_config: SimulationConfig,
    storage: SimulationStorage,
    trading_date: str,
) -> None:
    """Show what would happen without executing trades."""
    signals = _generate_mock_signals()
    prices = {s.stock_code: 100.0 for s in signals}

    typer.echo(f"\n[DRY RUN] Simulation — {trading_date}")
    typer.echo(f"Config: {sim_config.account}, capital={sim_config.initial_capital:,}")
    typer.echo(f"\nGenerated {len(signals)} mock signals:")
    for s in signals:
        typer.echo(f"  {s.stock_code}: score={s.score:.4f} ({s.signal.value})")

    typer.echo(f"\nMock prices: {prices}")
    typer.echo("\n[DRY RUN] No trades executed (--dry-run mode)")


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

@app.command()
def history(
    config: str = typer.Option(
        "configs/simulation/default.yaml",
        help="Simulation config path (relative to project root)",
    ),
    limit: int = typer.Option(50, help="Maximum number of trades to show"),
    project_root: Path = typer.Option(
        None,
        help="Project root (default: script parent directory)",
    ),
) -> None:
    """Show trade history."""
    root = project_root or SCRIPT_DIR.parent
    config_path = root / config

    try:
        sim_config = load_simulation_config(config_path)
    except FileNotFoundError:
        sim_config = SimulationConfig()

    storage = SimulationStorage(
        base_dir=str(root / sim_config.storage_base_dir),
        trades_dir=str(root / sim_config.storage_trades_dir),
        decisions_dir=str(root / sim_config.storage_decisions_dir),
        snapshots_dir=str(root / sim_config.storage_snapshots_dir),
    )

    trades = storage.load_trades()
    if not trades:
        typer.echo("No trade history found.")
        return

    trades = trades[:limit]

    console = _try_rich()
    if console:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"Trade History (last {len(trades)})")
        table.add_column("Date", style="cyan")
        table.add_column("Stock", style="yellow")
        table.add_column("Side", style="green" if any(t.side == OrderSide.BUY for t in trades) else "red")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Commission", justify="right")

        for trade in trades:
            table.add_row(
                trade.timestamp.strftime("%Y-%m-%d"),
                trade.stock_code,
                trade.side.value,
                str(trade.quantity),
                f"{trade.fill_price:.2f}",
                f"{trade.commission:.2f}",
            )
        console.print(table)
    else:
        print(f"\nTrade History (last {len(trades)}):")
        print(f"{'Date':<12} {'Stock':<12} {'Side':<6} {'Qty':>6} {'Price':>10} {'Commission':>12}")
        print("-" * 62)
        for trade in trades:
            print(
                f"{trade.timestamp.strftime('%Y-%m-%d'):<12} "
                f"{trade.stock_code:<12} {trade.side.value:<6} "
                f"{trade.quantity:>6} {trade.fill_price:>10.2f} {trade.commission:>12.2f}"
            )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()