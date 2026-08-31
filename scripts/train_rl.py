"""DRL training for carry-trade allocator (synthetic + heuristic fallback).

- Generates synthetic carry data (or real rates via providers if online)
- Runs CarryTradeEnv episodes with random / equal baselines via TradeOutcomeLogger
- Trains DRLAllocator heuristic (hit_rate per currency -> bias)
- Evaluates net_sharpe improvement via VectorBTBacktest fallback
- Saves logs/rl_training_report.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch  # optional
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from tradingagents.dataflows.vectorbt_backtest import VectorBTBacktest
from tradingagents.learning.outcomes import TradeOutcomeLogger
from tradingagents.rl.env import CarryTradeEnv, EnvConfig
from tradingagents.rl.weight_allocator import DRLAllocator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Patch TimesFM to avoid HF 401 hangs - SMA fallback instantly
try:
    from tradingagents.chains import forecasting as _fc

    def _fast(self, hist, horizon=1):
        h = np.asarray(hist, float).ravel()
        sma = float(np.mean(h[-5:])) if len(h) else 0.0
        return type("R", (), {"forecast": np.full(horizon, sma)})()

    _fc.TimesFMForecaster.predict = _fast  # type: ignore
except Exception:
    pass


def synthetic_data(n: int = 60, n_assets: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0, 0.5, size=(n, n_assets)), axis=0)
    symbols = [f"FX{i}" for i in range(n_assets)]
    price = pd.DataFrame(base, columns=symbols)
    spread = rng.normal(0.03, 0.02, size=(n, n_assets))  # rate differential
    return price, spread, symbols


def try_real_data(n_assets=3):
    try:
        from tradingagents.dataflows.providers.fx_multi_currency import MultiCurrencyFXProvider
        from tradingagents.dataflows.providers.interest_rates import GlobalInterestRatesProvider

        rp = GlobalInterestRatesProvider()
        rates = rp.get_all_rates()
        rp.close()
        # pick top spreads as synthetic proxy - offline fallback below
        if len(rates) >= n_assets:
            price, spread, symbols = synthetic_data(n_assets=n_assets)
            # inject real rate levels into spread mean
            vals = sorted([r.rate for r in rates.values()])[-n_assets:]
            spread = np.tile(np.array(vals) / 100.0, (len(price), 1)) + np.random.normal(0, 0.01, size=(len(price), n_assets))
            fp = MultiCurrencyFXProvider()
            fp.close()
            log.info("Real rates fetched, using hybrid synthetic prices")
            return price, spread, symbols
    except Exception as e:
        log.warning("Real data fetch failed (%s) -> synthetic fallback", e)
    return synthetic_data(n_assets=n_assets)


def run_episodes(price, spread, symbols, weights_fn, episodes=3, tag="baseline"):
    # Use isolated logger per run to avoid polluting global file
    tmp_path = Path(f"logs/_tmp_{tag}.jsonl")
    if tmp_path.exists():
        tmp_path.unlink()
    logger = TradeOutcomeLogger(path=tmp_path)
    all_outcomes = []
    for ep in range(episodes):
        env = CarryTradeEnv(price_data=price, rate_spread=spread, config=EnvConfig(n_assets=len(symbols), max_steps=len(price)))
        obs, _ = env.reset(seed=42 + ep)
        done = False
        while not done:
            w = weights_fn(price, ep)
            obs, reward, terminated, truncated, info = env.step(w)
            done = terminated or truncated
            # log per-step outcome
            rec = logger.log_outcome(
                gross_spread=float(np.mean(spread[env.current_step - 1])),
                net_expected=float(reward + info.get("cost", 0)),
                forecast=float(np.mean(obs[: len(symbols)])) if len(obs) else 0.0,
                real_fx_move=float(info.get("pnl", 0)),
                pnl_net=float(reward),
                cost_breakdown={"cost": float(info.get("cost", 0))},
                symbol=symbols[ep % len(symbols)],
            )
            all_outcomes.append(rec)
    stats = logger.compute_stats()
    # keep file for debugging, but also return outcomes
    return all_outcomes, stats, tmp_path


def backtest_metrics(price: pd.DataFrame, outcomes: list):
    bt = VectorBTBacktest(fees=0.0025)
    px = price.iloc[:, 0]
    fast = px.rolling(10, min_periods=10).mean()
    slow = px.rolling(30, min_periods=30).mean()
    entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    res = bt.run(px, entries.fillna(False), exits.fillna(False))
    real = bt.real_sharpe_from_outcomes(outcomes)
    pnls = np.array([float(o.get("pnl_net", 0)) for o in outcomes], dtype=float)
    net_return = float(pnls.sum()) if len(pnls) else 0.0
    # net_sharpe from outcomes is more relevant than SMA backtest
    return {
        "net_return": net_return,
        "sharpe": float(res.sharpe),
        "net_sharpe": float(real.get("net_sharpe_real", res.net_sharpe)),
        "win_rate": float(stats_win(pnls)),
        "backtest_return": float(res.total_return),
    }


def stats_win(pnls):
    return float(np.mean(pnls > 0)) if len(pnls) else 0.0


def main():
    price, spread, symbols = try_real_data(n_assets=3)
    log.info("Data: %s x %s via %s", price.shape, spread.shape, "hybrid" if len(price) else "synthetic")
    if _HAS_TORCH:
        log.info("torch %s available (optional)", torch.__version__)
    else:
        log.info("torch not installed - heuristic only")

    # baselines
    def rand_w(_price, _ep):
        w = np.random.dirichlet(np.ones(len(symbols)))
        return w

    def equal_w(_price, _ep):
        return np.full(len(symbols), 1.0 / len(symbols))

    rand_out, rand_stats, _ = run_episodes(price, spread, symbols, rand_w, episodes=2, tag="rand")
    eq_out, eq_stats, _ = run_episodes(price, spread, symbols, equal_w, episodes=2, tag="equal")
    before_out = rand_out + eq_out
    # aggregate hit_rate before
    pnls_before = np.array([float(o["pnl_net"]) for o in before_out])
    hit_before = float(np.mean(pnls_before > 0)) if len(pnls_before) else 0.0
    bm_before = backtest_metrics(price, before_out)

    # heuristic training: DRLAllocator bias from hit_rate per currency
    alloc = DRLAllocator(n_assets=len(symbols))
    bias = alloc.update_from_outcomes(before_out)
    log.info("Heuristic bias after training: %s", bias)

    # evaluate after: run with biased allocator
    def drl_w(feat, _ep):
        # feat is price df; alloc expects DataFrame
        return alloc.allocate(feat)

    after_out, after_stats, _ = run_episodes(price, spread, symbols, drl_w, episodes=2, tag="drl")
    pnls_after = np.array([float(o["pnl_net"]) for o in after_out])
    hit_after = float(np.mean(pnls_after > 0)) if len(pnls_after) else 0.0
    bm_after = backtest_metrics(price, after_out)

    report = {
        "before": {"hit_rate": hit_before, "net_return": bm_before["net_return"], "sharpe": bm_before["sharpe"], "net_sharpe": bm_before["net_sharpe"], "win_rate": bm_before["win_rate"], "count": len(before_out), "stats": rand_stats},
        "after": {"hit_rate": hit_after, "net_return": bm_after["net_return"], "sharpe": bm_after["sharpe"], "net_sharpe": bm_after["net_sharpe"], "win_rate": bm_after["win_rate"], "count": len(after_out), "stats": after_stats},
        "bias": bias,
        "baselines": {"random_hit": float(np.mean([float(o["pnl_net"]) > 0 for o in rand_out])) if rand_out else 0.0, "equal_hit": float(np.mean([float(o["pnl_net"]) > 0 for o in eq_out])) if eq_out else 0.0},
        "symbols": symbols,
        "has_torch": _HAS_TORCH,
        "price_shape": list(price.shape),
    }
    out = Path("logs/rl_training_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Report saved to %s", out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
