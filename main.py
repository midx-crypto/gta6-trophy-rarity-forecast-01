"""GTA VI trophy rarity forecaster: earners vs players, refresh-aware.

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles. The core idea is that a rate is a ratio, earners / players, and
the two sides move for different reasons: earners only grow as players
progress through the game, while the tracked-player count jumps whenever
Exophase refreshes its counts and floods in new players. So the program
fits the two counts separately in log time and divides at the target,
instead of extrapolating the ratio itself.

Two regimes, picked from what the history actually shows:

* Flat history (every read identical: same players, same rates). The page
  has not refreshed since launch, so the next refresh is the whole story.
  The program assumes one refresh lands before the target and dilutes
  every rate: earners stay where they are, players grow to a post-refresh
  level given by a saturating launch-adoption curve in the hours since the
  first read. A trophy at 100% stays at 100%: every tracked player has it,
  and a refresh that only adds players who also have it cannot move it.
* Moving history (reads change). Per trophy, log earners (or log
  non-earners for trophies above 50%, where the complement is the small,
  well-behaved count) is fitted against log(1 + hours since release) by
  least squares over the recent window, with each slope shrunk toward the
  cross-trophy median slope, and log players is fitted the same way. Both
  counts are extrapolated from the last read, but the trophy's own excess
  over the players trend is damped: the further the target lies past the
  data, the less of the recent rate trend is trusted, so the forecast
  glides toward persistence instead of overshooting. The spread combines
  the fit residuals, slope uncertainty scaled by how far the target is
  past the data, and a random-walk term that grows with the horizon.

Standard library only, no network, deterministic: the same input always
writes the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Midnight UTC on release day; the time axis for the trend fits.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Flat-history regime: log of the post-refresh player multiple, saturating.
# ln G = G_AMP * (1 - exp(-H / G_TAU)), H = hours from first read to target.
# Launch night the tracked count multiplies several-fold at the first
# refresh and then the wave flattens; the curve below reaches ~6x after
# about four and a half hours and asymptotes near e**2.
G_AMP = 2.0
G_TAU = 2.0
FLAT_SIGMA = 0.15

# Moving-history regime: spread pieces, in logit units.
RW_SIGMA = 0.15      # random-walk sigma per sqrt(day) of horizon
BASE_SIGMA = 0.15    # refresh lag / staleness floor
SIGMA_MIN = 0.20
SIGMA_MAX = 2.0

# Moving-history regime: slope estimation and extrapolation.
TREND_WINDOW_H = 36.0  # hours of recent reads the slopes are fitted on
TREND_DAMP_P = 0.9      # damp on the players log-increment extrapolation
TREND_DAMP_MODE = "adaptive"  # "adaptive": 1/(1+dx); else a float damp


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime (tolerates a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def quantiles_from_logit(center, sigma):
    """Five quantiles around a logit centre with a logit spread."""
    out = []
    for z in Z:
        q = inv_logit(center + z * sigma)
        q = min(100.0, max(0.01, q))
        out.append(round4(q))
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, se_b, xbar, sxx, n)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return ybar, 0.0, 0.0, 0.0, xbar, sxx, n
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    if n > 2:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        resid_sd = math.sqrt(rss / (n - 2))
    else:
        resid_sd = 0.0
    se_b = resid_sd / math.sqrt(sxx) if sxx > 0 else 0.0
    return a, b, resid_sd, se_b, xbar, sxx, n


def median(values):
    vals = sorted(values)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def last_known(values):
    """Last non-None entry, or None."""
    for v in reversed(values):
        if v is not None:
            return v
    return None


def history_flat(reads):
    """True when the page never moved: one distinct player count and rates
    that never change beyond rounding noise."""
    if len(reads) <= 1:
        return True
    players = [r["players_tracked"] for r in reads if r.get("players_tracked") is not None]
    if players and max(players) != min(players):
        return False
    n_trophies = len(reads[0]["rates"])
    for j in range(n_trophies):
        col = [r["rates"][j] for r in reads
               if j < len(r["rates"]) and r["rates"][j] is not None]
        if col and max(col) - min(col) > 0.005:
            return False
    return True


def forecast_flat(case, reads, target):
    """Flat history: one refresh before the target dilutes every rate."""
    first = parse_time(reads[0]["at"])
    hours = max(0.0, (target - first).total_seconds() / 3600.0)
    growth = math.exp(G_AMP * (1.0 - math.exp(-hours / G_TAU)))
    last = reads[-1]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if i < len(last["rates"]) else None
        if rate is None:
            rate = last_known([r["rates"][i] for r in reads
                               if i < len(r["rates"])]) or 0.5
        if rate >= 99.99:
            # A trophy every tracked player has stays at 100% through a
            # refresh that only adds players who have it too.
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        diluted = clamp_rate(rate / growth)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(logit(diluted), FLAT_SIGMA)})
    return {"trophies": rows}


def trend_damp(dx):
    """Damp on a trophy's own excess trend: 1 at the last read, fading as
    the target moves past the data, so a far target trusts the recent
    rate trend less and glides toward persistence."""
    if TREND_DAMP_MODE == "adaptive":
        return 1.0 / (1.0 + max(0.0, dx))
    return float(TREND_DAMP_MODE)


def forecast_trend(case, reads, target):
    """Moving history: extrapolate counts from the last read using recent
    log-time slopes, damping each trophy's excess over the players trend.

    log P grows with its recent slope; log E grows with the players slope
    plus a damped excess (its own slope minus the players slope), so the
    rate moves by the damped excess only. Trophies above 50% are handled
    through the non-earner fraction 1 - rate, which is the small,
    well-behaved count near 100%.
    """
    times = [parse_time(r["at"]) for r in reads]
    hours = [(t - RELEASE).total_seconds() / 3600.0 for t in times]
    xs = [math.log1p(max(0.0, h)) for h in hours]
    x_target = math.log1p(max(0.0, (target - RELEASE).total_seconds() / 3600.0))
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)

    players = []
    for r in reads:
        p = r.get("players_tracked")
        players.append(p if p is not None else (players[-1] if players else 1))
    last_players = players[-1]
    x_last = xs[-1]
    dx = max(0.0, x_target - x_last)
    damp = trend_damp(dx)

    # Recent window for slope estimation (all reads if the case is short).
    t_last = hours[-1]
    win = [i for i, h in enumerate(hours) if h >= t_last - TREND_WINDOW_H]
    if len(win) < 3 or max(xs[i] for i in win) - min(xs[i] for i in win) <= 1e-9:
        win = list(range(len(reads)))

    # Players trend from the window.
    wx = [xs[i] for i in win if players[i] and players[i] > 0]
    wy = [math.log(players[i]) for i in win if players[i] and players[i] > 0]
    if len(wx) >= 3 and max(wx) - min(wx) > 1e-9:
        _, pd, presid, pse_b, pxbar, psxx, _ = ols(wx, wy)
    else:
        pd, presid, pse_b, pxbar, psxx = 0.0, 0.0, 0.0, x_last, 0.0
    pd = min(max(pd, 0.0), 2.0)
    players_target = last_players * math.exp(pd * dx * TREND_DAMP_P)
    players_target = min(players_target, float(last_players) * 200.0)
    se_players = presid * math.sqrt(max(0.0, 1.0 / max(len(wx), 1) +
                                        (x_target - pxbar) ** 2 / psxx)) if psxx > 0 else presid

    n_trophies = len(case["trophies"])
    # Per-trophy window series: log earners, or log non-earner fraction.
    series = []
    for j in range(n_trophies):
        col = []
        for r, p in zip(reads, players):
            rate = r["rates"][j] if j < len(r["rates"]) else None
            col.append(None if rate is None else rate * p / 100.0)
        last_rate = last_known([r["rates"][j] for r in reads
                                if j < len(r["rates"])])
        use_complement = last_rate is not None and last_rate > 50.0
        pts = []
        for i in win:
            r = reads[i]
            rate = r["rates"][j] if j < len(r["rates"]) else None
            if rate is None:
                continue
            if use_complement:
                frac = max(1.0 - rate / 100.0, 1e-4)
                pts.append((xs[i], math.log(frac)))
            else:
                e = col[i]
                if e is not None:
                    pts.append((xs[i], math.log(max(e, 0.5))))
        series.append({"use_complement": use_complement, "pts": pts,
                       "last_rate": last_rate,
                       "last_earners": last_known(col) or 0.0})

    raw = []
    for s in series:
        if len(s["pts"]) >= 3 and max(p[0] for p in s["pts"]) - min(p[0] for p in s["pts"]) > 1e-9:
            a, b, sd, se_b, xbar, sxx, n = ols([p[0] for p in s["pts"]],
                                               [p[1] for p in s["pts"]])
        else:
            b, sd, se_b, xbar, sxx, n = 0.0, 0.0, 0.0, x_last, 0.0, 0
        raw.append({"b": b, "sd": sd, "se_b": se_b, "xbar": xbar,
                    "sxx": sxx, "n": len(s["pts"])})
    med_e = median([r["b"] for r, s in zip(raw, series) if not s["use_complement"]]) or pd
    med_u = median([r["b"] for r, s in zip(raw, series) if s["use_complement"]]) or 0.0
    med_sd = median([r["sd"] for r in raw if r["n"] >= 3]) or 0.1

    rows = []
    for j, trophy in enumerate(case["trophies"]):
        s = series[j]
        r = raw[j]
        centre = med_u if s["use_complement"] else med_e
        # Shrink the slope toward the group median; noisy fits shrink more.
        w = r["n"] / (r["n"] + 8.0) if r["n"] else 0.0
        if r["sd"] > 0.3:
            w *= 0.5
        b = w * r["b"] + (1.0 - w) * centre

        if s["use_complement"]:
            # Non-earner fraction moves by its own damped slope.
            b = min(max(b, -2.0), 2.0)
            u_last = max(1.0 - (s["last_rate"] or 0.0) / 100.0, 1e-4)
            u_target = u_last * math.exp(b * dx * damp)
            u_target = min(max(u_target, 1e-4), 0.999)
            med_rate = clamp_rate(100.0 * (1.0 - u_target))
            slope_var = (r["se_b"] * dx * damp) ** 2 if r["sxx"] > 0 else (0.3 * dx * damp) ** 2
        else:
            # Earners ride the players trend plus a damped excess.
            b = min(max(b, 0.0), 3.0)
            earners_target = s["last_earners"] * math.exp(
                (pd * TREND_DAMP_P + damp * (b - pd)) * dx)
            earners_target = min(max(earners_target, s["last_earners"]),
                                 players_target)
            med_rate = clamp_rate(100.0 * earners_target / players_target)
            slope_var = (r["se_b"] * dx * damp) ** 2 if r["sxx"] > 0 else (0.3 * dx * damp) ** 2

        sd = max(r["sd"], 0.05)
        sd = 0.5 * sd + 0.5 * med_sd
        var = sd * sd + slope_var + se_players ** 2 \
            + (RW_SIGMA * math.sqrt(horizon_days)) ** 2 + BASE_SIGMA ** 2
        sigma = min(max(math.sqrt(var), SIGMA_MIN), SIGMA_MAX)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(logit(med_rate), sigma)})
    return {"trophies": rows}


def forecast_case(case):
    reads = sorted(case.get("history") or [], key=lambda r: r.get("at", ""))
    target = parse_time(case["target"])
    if not reads:
        return {"trophies": [
            {"id": t["id"], "q": quantiles_from_logit(logit(1.0), 1.0)}
            for t in case["trophies"]]}
    if history_flat(reads):
        return forecast_flat(case, reads, target)
    return forecast_trend(case, reads, target)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
