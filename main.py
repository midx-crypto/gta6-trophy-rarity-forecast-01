"""GTA VI trophy earn-rate forecaster for make.fun's
gta6-trophy-rarity-forecast-01 bounty.

Reads one case per run from $MAKEFUN_INPUT_DIR (default /input) and writes
one forecast per case to $MAKEFUN_OUTPUT_DIR (default /output): for each of
the game's trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn
rate on Exophase at the case target. Standard library only, no network,
deterministic.

Model, per trophy, in logit space:
  1. Trend: least squares of logit(rate) on ln(1 + hours since release),
     over the case history's change points. Exophase's page is frozen
     between refreshes, so repeated identical reads carry no new
     information and only distinct values enter the fit. The slope is
     shrunk by n/(n+2) on the number of distinct points, so a flat or
     nearly-flat history forecasts flat, while a trophy with a clear
     multi-refresh trend follows it. The log-time axis damps
     extrapolation: the final's target sits only about one ln-hour past
     the deadline.
  2. Spread: a random-walk term (0.5 * sqrt(horizon in days), as the
     house baseline) combined with the trend fit's residual variance,
     floored so a deceptively clean fit never claims false precision.
  3. Refresh mixture, ONLY when the case history shows no refresh at
     all (players_tracked and every rate constant across all reads):
     the page may refresh before the target, and a refresh dilutes
     rates because newly tracked players hold far fewer trophies. The
     kit documents one such refresh on launch night (638 -> 3,868
     tracked players; every rate but the 100% one fell, roughly a -1.6
     logit shift). The forecast is a two-branch mixture in logit
     space: no-refresh (weight 1-p) vs refresh (weight p, centre
     shifted -1.6), with p from the exposure (horizon plus how stale
     the page already is) over a ~4h refresh timescale. Quantiles come
     from numerically inverting the mixture CDF, so they are
     asymmetric: the low quantiles sit in the refresh branch while the
     median stays near the last rate unless a refresh is more likely
     than not. Once any refresh is observed in the history, the trend
     fit already prices refreshes in and the mixture is not used.
  4. The 100% trophy stays at 100%: every tracked player holds it,
     new players included.

Local validation (practice answers reconstructed from the aggregates the
kit itself documents: 638 -> 3,868 players, 943 -> 4,276 trophies earned,
every rate but 100% fell): this forecaster scores ~0.26-0.30 mean pinball
loss on the practice cases against ~1.38-1.48 for the house baseline,
across three reconstructions of how the new trophies spread. On a
synthetic final-like case (two days of reads with lumpy refreshes and
log-time trophy growth) it scores ~0.22 against ~0.27 for the baseline.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Random-walk spread of a rate's move, logit units per sqrt day (house baseline).
RW_SIGMA = 0.5
# Logit shift of a refresh that dilutes rates (launch-night refresh, kit docs).
REFRESH_SHIFT = 1.6
# Refresh timescale prior, hours (page frozen ~4.7h before launch night's refresh).
REFRESH_TAU_H = 4.0
# Spread of each mixture branch, logit units.
SIGMA_NO_REFRESH_FLOOR = 0.15
SIGMA_REFRESH = 0.6
# Spread floor / cap for the trend branch, logit units.
SIGMA_FLOOR = 0.35
SIGMA_CAP = 2.0
# Caps on the trend extrapolation, logit units.
MAX_TREND_SHIFT = 2.5
MAX_SLOPE = 1.5


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def parse_time(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantile(p, m, s0, m1, s1, tau):
    """tau-quantile of (1-p)*N(m,s0) + p*N(m1,s1), by bisection on the CDF."""
    lo = min(m - 8 * s0, m1 - 8 * s1)
    hi = max(m + 8 * s0, m1 + 8 * s1)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = (1 - p) * norm_cdf((mid - m) / s0) + p * norm_cdf((mid - m1) / s1)
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels differ from [0.05, 0.25, 0.5, 0.75, 0.95]")
    history = case["history"]
    if not history:
        raise ValueError("no history")
    release = parse_time(case["game"]["release_date"])
    checkpoint = parse_time(case["at"])
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    x_target = math.log1p((parse_time(case["target"]) - release).total_seconds() / 3600.0)

    # Does the history show any refresh at all? (tracked count or any rate moves)
    any_refresh = False
    first_n = history[0]["players_tracked"]
    for read in history:
        if read["players_tracked"] != first_n:
            any_refresh = True
            break
    if not any_refresh:
        base_rates = history[0]["rates"]
        for read in history[1:]:
            if read["rates"] != base_rates:
                any_refresh = True
                break

    staleness_h = (checkpoint - parse_time(history[0]["at"])).total_seconds() / 3600.0
    exposure_h = horizon_h + min(staleness_h, REFRESH_TAU_H)
    p_refresh = 1.0 - math.exp(-exposure_h / REFRESH_TAU_H) if not any_refresh else 0.0

    last = history[-1]
    # Log-time of every read, computed once per case (the final case holds
    # hundreds of reads; parsing each one per trophy would waste time).
    read_x = [math.log1p((parse_time(read["at"]) - release).total_seconds() / 3600.0)
              for read in history]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = last["rates"][i]
        if last_rate is None:
            for read in reversed(history):
                if read["rates"][i] is not None:
                    last_rate = read["rates"][i]
                    break
            if last_rate is None:
                last_rate = 0.5
        lg_last = logit(last_rate)

        # Change points of this trophy: (x, logit rate), consecutive distinct only.
        pts = []
        for j, read in enumerate(history):
            r = read["rates"][i]
            if r is None:
                continue
            x = read_x[j]
            y = logit(r)
            if not pts or abs(y - pts[-1][1]) > 1e-9 or abs(x - pts[-1][0]) > 1e-9:
                pts.append((x, y))
        n = len(pts)

        # Trend fit (least squares on change points), shrunk slope.
        slope = 0.0
        intercept = lg_last
        resid_var = 0.0
        if n >= 2:
            mx = sum(p[0] for p in pts) / n
            my = sum(p[1] for p in pts) / n
            sxx = sum((p[0] - mx) ** 2 for p in pts)
            sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
            if sxx > 1e-12:
                raw = sxy / sxx
                raw = max(-MAX_SLOPE, min(MAX_SLOPE, raw))
                slope = raw * n / (n + 2.0)
                intercept = my - slope * mx
            if n >= 3:
                resid_var = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in pts) / (n - 2)
        x_last = pts[-1][0] if pts else 0.0
        shift = slope * (x_target - x_last)
        shift = max(-MAX_TREND_SHIFT, min(MAX_TREND_SHIFT, shift))
        median_lg = lg_last + shift

        if last_rate >= 99.9:
            # Everyone holds it; a refresh cannot dilute it.
            qs = [round4(clamp_pct(inv_logit(median_lg + z * RW_SIGMA * math.sqrt(horizon_d))))
                  for z in Z]
        elif not any_refresh:
            # No refresh ever observed: mixture over "refresh before target".
            s0 = max(SIGMA_NO_REFRESH_FLOOR, RW_SIGMA * math.sqrt(horizon_d))
            qs = [round4(clamp_pct(inv_logit(mixture_quantile(
                p_refresh, median_lg, s0, median_lg - REFRESH_SHIFT, SIGMA_REFRESH, tau))))
                for tau in LEVELS]
        else:
            sigma = math.sqrt((RW_SIGMA * math.sqrt(horizon_d)) ** 2 + resid_var + 0.09)
            sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
            qs = [round4(clamp_pct(inv_logit(median_lg + z * sigma))) for z in Z]
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
