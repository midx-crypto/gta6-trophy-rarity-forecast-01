"""GTA VI trophy rarity forecast, round 2.

Forecasts each trophy's earn rate on Exophase at the case target, as five
quantiles (levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95) on the logit scale.

Two regimes, picked from the case history:

* Flat history (every read shows the same page: players tracked constant
  and every rate unchanged). The page is stale: Exophase refreshes on its
  own schedule, so by the target the page will have refreshed at least
  once. A refresh brings many new players (dilution) but the earners of
  fast, early trophies grow with them, while slow grind trophies barely
  gain earners. So the median forecast dilutes each trophy's last rate by
  the expected player growth G, then scales its earners back up by a tier
  multiplier: the 6 fastest trophies (highest last rate) grow most, the
  next 17 grow a little, the rest not at all.

* Moving history (rates/players change across reads). Fit, per trophy,
  a least-squares line of log earners (or of the log non-earner fraction
  for trophies above 50%) against log time since release, over the last
  36 hours of reads, with the slope shrunk toward its group median; fit
  log players the same way for the denominator. The spread combines the
  fit residuals, the slope uncertainty, the players uncertainty and a
  horizon term.

Standard library only, no network, deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Flat regime: player growth G(H) = exp(G_AMP * (1 - exp(-H / G_TAU))),
# H = hours from the first read to the target. G_AMP=1.82, G_TAU=1.0
# saturates fast: G ~ 6.07 at the practice targets, matching the observed
# refresh (3868 / 638 = 6.06); the slower r1 curve (2.0, 2.0) overshoots.
G_AMP = 1.82
G_TAU = 1.0
# Flat regime: earner-growth tiers by last-rate rank (ties: id ascending).
K1 = 6
M1 = 2.25
K2 = 17
M2 = 1.25
# Flat regime: logit spread of the five quantiles around the median.
# Swept {0.08..0.20} on truth v2: narrower wins on every measured
# criterion, but below 0.08 the gain only exploits truth v2 being
# deterministic; 0.08 is the bottom of the swept grid, kept as the
# robust choice rather than chasing the knife edge.
FLAT_SIGMA = 0.08


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l >= 0.0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def quantiles_from_logit(center, sigma):
    out = []
    prev = None
    for z in Z:
        q = round4(min(100.0, max(0.01, inv_logit(center + z * sigma))))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def ols(xs, ys):
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (ybar, 0.0, 0.0, 0.0, xbar, sxx, n)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    resid_sd = math.sqrt(rss / (n - 2)) if n > 2 else 0.0
    se_b = resid_sd / math.sqrt(sxx) if sxx > 0.0 else 0.0
    return (a, b, resid_sd, se_b, xbar, sxx, n)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def last_known(values):
    for v in reversed(values):
        if v is not None:
            return v
    return None


def history_flat(reads):
    if len(reads) <= 1:
        return True
    players = [r["players_tracked"] for r in reads]
    if any(p != players[0] for p in players):
        return False
    ntr = len(reads[0]["rates"])
    for i in range(ntr):
        vals = [r["rates"][i] for r in reads if r["rates"][i] is not None]
        if vals and max(vals) - min(vals) > 0.005:
            return False
    return True


def forecast_case(case):
    reads = sorted(case.get("history") or [], key=lambda r: r["at"])
    target = parse_time(case["target"])
    trophies = case["trophies"]
    if not reads:
        return {"trophies": [
            {"id": t["id"], "q": quantiles_from_logit(logit(1.0), 1.0)}
            for t in trophies]}
    if history_flat(reads):
        return forecast_flat(case, reads, target)
    return forecast_trend(case, reads, target)


def forecast_flat(case, reads, target):
    first_at = parse_time(reads[0]["at"])
    hours = (target - first_at).total_seconds() / 3600.0
    growth = math.exp(G_AMP * (1.0 - math.exp(-hours / G_TAU)))
    last = reads[-1]
    players_last = last["players_tracked"]
    last_rates = []
    for i in range(len(case["trophies"])):
        col = [r["rates"][i] for r in reads if r["rates"][i] is not None]
        last_rates.append(col[-1] if col else 1.0)
    order = sorted(range(len(last_rates)),
                   key=lambda i: (-last_rates[i], case["trophies"][i]["id"]))
    mult = [1.0] * len(last_rates)
    for rank, i in enumerate(order):
        if rank < K1:
            mult[i] = M1
        elif rank < K1 + K2:
            mult[i] = M2
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i]
        if rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        earners = round(rate * players_last / 100.0)
        median_rate = 100.0 * earners * mult[i] / (players_last * growth)
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(logit(median_rate), FLAT_SIGMA)})
    return {"trophies": rows}


def forecast_trend(case, reads, target):
    trophies = case["trophies"]
    ntr = len(trophies)
    times = [parse_time(r["at"]) for r in reads]
    hours_list = [(t - RELEASE).total_seconds() / 3600.0 for t in times]
    xs = [math.log(1.0 + h) for h in hours_list]
    x_target = math.log(1.0 + (target - RELEASE).total_seconds() / 3600.0)
    horizon_days = case["horizon_minutes"] / 1440.0
    last_hour = hours_list[-1]
    players = []
    prev = None
    for r in reads:
        p = r["players_tracked"]
        if p is None:
            p = prev
        players.append(p)
        if p is not None:
            prev = p
    first_known = next((p for p in players if p is not None), None)
    players = [p if p is not None else first_known for p in players]
    last_players = players[-1]
    x_last = xs[-1]
    dx = max(0.0, x_target - x_last)
    damp = 1.0 / (1.0 + dx)
    win_idx = [i for i, h in enumerate(hours_list) if h >= last_hour - 36.0]
    wx_probe = [xs[i] for i in win_idx]
    if len(win_idx) < 3 or (wx_probe and max(wx_probe) - min(wx_probe) <= 1e-9):
        win_idx = list(range(len(reads)))
    wx = [xs[i] for i in win_idx]
    wlogp = [math.log(players[i]) for i in win_idx]
    _pa, pd, presid, _pse, pxbar, psxx, _pn = ols(wx, wlogp)
    pd = min(2.0, max(0.0, pd))
    players_target = last_players * math.exp(pd * dx * 0.9)
    players_target = min(players_target, 200.0 * last_players)
    if psxx > 0.0:
        se_players = presid * math.sqrt(
            1.0 / max(len(wx), 1) + (x_target - pxbar) ** 2 / psxx)
    else:
        se_players = presid
    fits = []
    for i in range(ntr):
        col_all = [r["rates"][i] for r in reads]
        last_rate = last_known(col_all)
        if last_rate is None:
            last_rate = 1.0
        complement = last_rate > 50.0
        pts_x = []
        pts_y = []
        for j in win_idx:
            rate = reads[j]["rates"][i]
            if rate is None:
                continue
            if complement:
                pts_y.append(math.log(max(1.0 - rate / 100.0, 1e-4)))
            else:
                earners = max(rate * players[j] / 100.0, 0.5)
                pts_y.append(math.log(earners))
            pts_x.append(xs[j])
        if pts_x:
            _a, b, sd, se_b, _xb, sxx, n = ols(pts_x, pts_y)
        else:
            b, sd, se_b, sxx, n = 0.0, 0.1, 0.0, 0.0, 0
        fits.append({"last_rate": last_rate, "complement": complement,
                     "b": b, "sd": sd, "se_b": se_b, "sxx": sxx, "n": n})
    med_e = median([f["b"] for f in fits if not f["complement"]]) \
        if any(not f["complement"] for f in fits) else 0.0
    med_u = median([f["b"] for f in fits if f["complement"]]) \
        if any(f["complement"] for f in fits) else 0.0
    med_sd_e = median([f["sd"] for f in fits if not f["complement"]]) \
        if any(not f["complement"] for f in fits) else 0.1
    med_sd_u = median([f["sd"] for f in fits if f["complement"]]) \
        if any(f["complement"] for f in fits) else 0.1
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        f = fits[i]
        last_rate = f["last_rate"]
        if f["complement"]:
            med_b = med_u
            med_sd = med_sd_u
        else:
            med_b = med_e
            med_sd = med_sd_e
        w = f["n"] / (f["n"] + 8.0) if f["n"] > 0 else 0.0
        if f["sd"] > 0.3:
            w *= 0.5
        b = w * f["b"] + (1.0 - w) * med_b
        if f["complement"]:
            b = min(2.0, max(-2.0, b))
            u_last = max(1.0 - last_rate / 100.0, 1e-4)
            u_target = u_last * math.exp(b * dx * damp)
            u_target = min(0.999, max(1e-4, u_target))
            med_rate = clamp_rate(100.0 * (1.0 - u_target))
        else:
            b = min(3.0, max(0.0, b))
            last_earners = max(last_rate * last_players / 100.0, 0.5)
            earners_target = last_earners * math.exp(
                (pd * 0.9 + damp * (b - pd)) * dx)
            earners_target = min(max(earners_target, last_earners),
                                 players_target)
            med_rate = clamp_rate(100.0 * earners_target / players_target)
        if f["sxx"] > 0.0:
            slope_var = (f["se_b"] * dx * damp) ** 2
        else:
            slope_var = (0.3 * dx * damp) ** 2
        sd = max(f["sd"], 0.05)
        sd = 0.5 * sd + 0.5 * med_sd
        var = (sd ** 2 + slope_var + se_players ** 2
               + (0.15 * math.sqrt(horizon_days)) ** 2 + 0.15 ** 2)
        sigma = min(2.0, max(0.20, math.sqrt(var)))
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(logit(med_rate), sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
