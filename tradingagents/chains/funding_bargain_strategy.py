"""Funding-Bargain strategy: fund cheap in one market to hunt bargain in another."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

_FALLBACK_RATES = {"JPY": 0.10, "CHF": 1.00, "THB": 2.50, "USD": 5.50, "EUR": 4.50, "GBP": 5.00, "BRL": 10.50, "MXN": 11.00, "TRY": 50.00, "ZAR": 8.25, "PLN": 5.75, "COP": 12.00, "IDR": 6.25, "PHP": 6.50, "INR": 6.50, "CLP": 6.50, "AUD": 4.35, "CAD": 4.50, "CNY": 3.45, "ARS": 110.00}
_CCY_TO_COUNTRY = {"USD": "US", "JPY": "JP", "CHF": "CH", "EUR": "EU", "GBP": "GB", "BRL": "BR", "MXN": "MX", "TRY": "TR", "ZAR": "ZA", "PLN": "PL", "COP": "CO", "IDR": "ID", "THB": "TH", "PHP": "PH", "INR": "IN", "CLP": "CL", "ARS": "AR", "AUD": "AU", "CAD": "CA", "CNY": "CN"}
_FALLBACK_FX = {("USD", "BRL"): 5.16, ("USD", "JPY"): 155.0, ("USD", "CHF"): 0.91, ("USD", "THB"): 36.5, ("USD", "MXN"): 17.5, ("USD", "TRY"): 48.0, ("USD", "ZAR"): 18.5, ("JPY", "BRL"): 0.033, ("CHF", "BRL"): 5.6}

try:
    from tradingagents.dataflows.cost_model import CostModel
    _HAS_COST = True
except Exception:
    _HAS_COST = False
    CostModel = None  # type: ignore


class FundingBargainStrategy:
    """Borrow cheap funding, hunt bargain, unwind."""

    def __init__(self, fx_provider=None, rates_provider=None, scanner=None, cost_model=None, logger=None):
        if fx_provider is None:
            try:
                from tradingagents.dataflows.providers.fx_multi_currency import MultiCurrencyFXProvider
                fx_provider = MultiCurrencyFXProvider()
            except Exception:
                fx_provider = None
        if rates_provider is None:
            try:
                from tradingagents.dataflows.providers.interest_rates import GlobalInterestRatesProvider
                rates_provider = GlobalInterestRatesProvider()
            except Exception:
                rates_provider = None
        if scanner is None:
            try:
                from tradingagents.dataflows.arbitrage_scanner import ArbitrageScanner
                scanner = ArbitrageScanner(fx_provider=fx_provider, rates_provider=rates_provider, cost_model=cost_model)
            except Exception:
                scanner = None
        if cost_model is None:
            try:
                cost_model = CostModel() if _HAS_COST and CostModel else None
            except Exception:
                cost_model = None
        self.fx = fx_provider
        self.rates = rates_provider
        self.scanner = scanner
        self.cost_model = cost_model
        self._outcome_logger = logger
        self._lifecycle: Dict[str, Dict] = {}

    def _rate_for(self, ccy: str) -> float:
        ccy = ccy.upper()
        # try provider with country mapping
        for key in (_CCY_TO_COUNTRY.get(ccy, ccy), ccy):
            try:
                if self.rates:
                    r = self.rates.get_rate(key)
                    if r and r.rate is not None:
                        return float(r.rate)
            except Exception:
                continue
        return float(_FALLBACK_RATES.get(ccy, 3.0))

    def _fx_rate(self, base: str, quote: str) -> float:
        base, quote = base.upper(), quote.upper()
        if base == quote:
            return 1.0
        try:
            if self.fx:
                r = self.fx.get_rate(base, quote)
                if r and r.rate:
                    return float(r.rate)
        except Exception:
            pass
        # fallback direct or via USD cross
        if (base, quote) in _FALLBACK_FX:
            return float(_FALLBACK_FX[(base, quote)])
        if (quote, base) in _FALLBACK_FX and _FALLBACK_FX[(quote, base)]:
            return 1.0 / float(_FALLBACK_FX[(quote, base)])
        # cross via USD
        try:
            usd_base = _FALLBACK_FX.get(("USD", base))
            usd_quote = _FALLBACK_FX.get(("USD", quote))
            if usd_base and usd_quote:
                return float(usd_quote) / float(usd_base)
        except Exception:
            pass
        return 1.0

    def find_funding_market(self) -> Dict:
        """Cheapest funding: lowest central-bank rate."""
        candidates: List[Dict] = []
        for ccy in _FALLBACK_RATES:
            rate = self._rate_for(ccy)
            candidates.append({"ccy": ccy, "rate": rate, "country": _CCY_TO_COUNTRY.get(ccy, ccy)})
        candidates.sort(key=lambda x: x["rate"])
        cheapest = candidates[0]
        return {"funding_ccy": cheapest["ccy"], "rate": cheapest["rate"], "country": cheapest["country"], "all_ranked": candidates, "timestamp": datetime.now().isoformat()}

    def find_bargain_market(self, threshold_bps: int = 50) -> Dict:
        """Cheap assets from ArbitrageScanner (ETF disconnect or arbitrage edge)."""
        if self.scanner:
            try:
                return self.scanner.find_cheap_assets(threshold_bps=threshold_bps)
            except Exception as e:
                logger.debug("scanner find_cheap_assets failed: %s", e)
        return {"arbitrage": [], "etf_disconnects": [], "threshold_bps": threshold_bps, "scanned_pairs": 0, "cheap_count": 0, "fallback": True}

    def generate_trade(self, funding_ccy: str, bargain_ccy: str, notional_usd: float = 10000, tenor_m: int = 3) -> Dict:
        funding_ccy, bargain_ccy = funding_ccy.upper(), bargain_ccy.upper()
        r_fund = self._rate_for(funding_ccy)
        r_bargain = self._rate_for(bargain_ccy)
        fx = self._fx_rate(funding_ccy, bargain_ccy)
        spread = r_bargain - r_fund
        # cost
        if self.cost_model:
            try:
                cm = self.cost_model.for_currency(bargain_ccy)
                cost_bps = float(cm.total_bps())
                breakdown = cm.cost_breakdown()
            except Exception:
                cost_bps, breakdown = 22.0, {"total_bps": 22.0}
        else:
            cost_bps, breakdown = 22.0, {"total_bps": 22.0, "fee_bps": 10, "slippage_bps": 5, "fx_spread_bps": 7}
        # check scanner edge for bargain
        edge_bps = None
        try:
            bargains = self.find_bargain_market(threshold_bps=50)
            for a in bargains.get("arbitrage", []):
                if a.get("quote") == bargain_ccy or bargain_ccy in a.get("pair", ""):
                    edge_bps = float(a.get("edge_bps", 0))
                    break
            for d in bargains.get("etf_disconnects", []):
                if d.get("ccy") == bargain_ccy and edge_bps is None:
                    edge_bps = float(d.get("disconnect_bps", 0))
        except Exception:
            pass
        gross_pct = (edge_bps / 100) if edge_bps is not None and abs(edge_bps) > abs(spread * 100) else spread
        # net via CostModel helper if available
        try:
            from tradingagents.dataflows.cost_model import net_expected_return as _ner
            net_pct, extra = _ner(spread, bargain_ccy, cost_model=self.cost_model)
        except Exception:
            net_pct, extra = spread - cost_bps / 100, {}
        tenor_y = tenor_m / 12.0
        funding_cost = notional_usd * r_fund / 100 * tenor_y
        expected_pnl = notional_usd * net_pct / 100 * tenor_y
        if edge_bps is not None:
            expected_pnl += notional_usd * edge_bps / 10000 * 0.5  # half edge as alpha
        bargain_notional = notional_usd * fx if funding_ccy == "USD" else notional_usd
        steps = [
            {"n": 1, "action": "BORROW", "desc": f"Borrow {notional_usd:.2f} {funding_ccy} at {r_fund:.2f}% ({tenor_m}M)", "ccy": funding_ccy, "notional": notional_usd, "rate": r_fund},
            {"n": 2, "action": "CONVERT", "desc": f"Convert {funding_ccy}->{bargain_ccy} @ {fx:.4f}", "from": funding_ccy, "to": bargain_ccy, "fx_rate": fx, "out_notional": bargain_notional},
            {"n": 3, "action": "INVEST", "desc": f"Invest in {bargain_ccy} bargain (ETF/carry) @ {r_bargain:.2f}% edge={edge_bps}", "ccy": bargain_ccy, "notional": bargain_notional, "expected_return_pct": round(gross_pct, 2)},
        ]
        unwind = [
            {"n": 1, "action": "SELL", "desc": f"Sell {bargain_ccy} bargain asset"},
            {"n": 2, "action": "CONVERT_BACK", "desc": f"Convert {bargain_ccy}->{funding_ccy} @ {fx:.4f}"},
            {"n": 3, "action": "REPAY", "desc": f"Repay {funding_ccy} {notional_usd:.2f} + interest {funding_cost:.2f}"},
        ]
        return {
            "funding_ccy": funding_ccy, "bargain_ccy": bargain_ccy, "notional_usd": notional_usd, "tenor_m": tenor_m,
            "funding_rate": r_fund, "bargain_rate": r_bargain, "spread_pct": round(spread, 2), "spread_bps": round(spread * 100, 2),
            "fx_rate": fx, "edge_bps": edge_bps, "cost_bps": cost_bps, "cost_breakdown": {**breakdown, **extra},
            "expected_gross_pct": round(float(gross_pct), 2), "expected_net_pct": round(float(net_pct), 2),
            "expected_pnl_usd": round(float(expected_pnl), 2), "funding_cost_usd": round(float(funding_cost), 2),
            "steps": steps, "unwind": unwind, "timestamp": datetime.now().isoformat(),
        }

    def execute_dry_run(self, trade: Dict, fx_shock_bps: float = 0) -> Dict:
        """Simulate pnl with costs and FX moves, log via TradeOutcomeLogger."""
        notional = float(trade.get("notional_usd", 10000))
        net_pct = float(trade.get("expected_net_pct", 0))
        tenor_y = float(trade.get("tenor_m", 3)) / 12.0
        gross_spread = float(trade.get("spread_pct", 0))
        # fx shock from volatility if not supplied
        if fx_shock_bps == 0:
            try:
                vol = self.fx.get_fx_volatility(trade["funding_ccy"], trade["bargain_ccy"], days=30) if self.fx else None
                fx_shock_bps = float(vol or 0) * 10000 * 0.5  # half vol as shock
            except Exception:
                fx_shock_bps = 0
        fx_drag = notional * fx_shock_bps / 10000
        simulated_pnl = notional * net_pct / 100 * tenor_y - fx_drag
        cost_breakdown = trade.get("cost_breakdown", {})
        # log
        try:
            from tradingagents.learning.outcomes import TradeOutcomeLogger
            lg = self._outcome_logger or TradeOutcomeLogger()
            lg.log_outcome(gross_spread=gross_spread, net_expected=net_pct, forecast=net_pct, real_fx_move=-fx_shock_bps / 100, pnl_net=simulated_pnl, cost_breakdown=cost_breakdown, symbol=f"{trade.get('funding_ccy')}/{trade.get('bargain_ccy')}")
        except Exception as e:
            logger.debug("TradeOutcomeLogger failed: %s", e)
        return {"simulated_pnl": round(simulated_pnl, 2), "fx_shock_bps": round(fx_shock_bps, 2), "net_pct": net_pct, "tenor_y": tenor_y, "cost_breakdown": cost_breakdown, "trade": trade}

    def manage_lifecycle(self, trade: Dict) -> Dict:
        """Track funding leg and bargain leg separately for unwind."""
        tid = f"{trade.get('funding_ccy')}/{trade.get('bargain_ccy')}/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        funding_leg = {"leg": "funding", "ccy": trade.get("funding_ccy"), "notional": trade.get("notional_usd"), "rate": trade.get("funding_rate"), "tenor_m": trade.get("tenor_m"), "status": "open", "due": (datetime.now() + timedelta(days=int(trade.get("tenor_m", 3) * 30))).isoformat(), "repay_amount": round(float(trade.get("notional_usd", 0)) + float(trade.get("funding_cost_usd", 0)), 2)}
        bargain_leg = {"leg": "bargain", "ccy": trade.get("bargain_ccy"), "notional": trade.get("notional_usd"), "fx_rate": trade.get("fx_rate"), "expected_net_pct": trade.get("expected_net_pct"), "status": "open", "unwind": trade.get("unwind", [])}
        lifecycle = {"trade_id": tid, "funding_leg": funding_leg, "bargain_leg": bargain_leg, "steps": trade.get("steps", []), "status": "active"}
        self._lifecycle[tid] = lifecycle
        return lifecycle
