#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case file per run (schema makefun/gta6-trophies-case/v1) from
$MAKEFUN_INPUT_DIR (else /input/) and writes /output/<id>.out per case:
five quantiles of each trophy's Exophase earn rate at the case's target.

Model, per trophy (Python 3.13, stdlib only, deterministic, no network):
  1. Logit trend: least-squares line of logit(rate) on ln(1 + hours since
     release), read at the target.
  2. Cohort model: holders (= rate * players_tracked / 100) and players
     tracked as power laws in ln(1 + hours since release); the cohort
     forecast is holders(target) / players(target).
  3. Blend: logit median = 0.65 * trend + 0.35 * cohort (weighted to trend).
  4. Archetype ceilings on the week-one forecast, from the trophy's
     description: Chapter 1 45%, other chapters 30%, finale/missions 15%,
     grind/collectibles 3%, platinum 0.6%.
  5. Short-horizon refresh prior: for horizons under 12 h, Exophase has
     usually not refreshed the game's counts yet, so the forecast shrinks
     toward a refresh-adjusted rate. The longer the page has sat unchanged,
     the bigger the pending player influx: players multiply by
     (1 + 1.5 * staleness_hours) while holders only grow by
     (1 + 0.12 * staleness_hours), so the rate is multiplied by their
     ratio. Calibrated on the practice cut: at the 03:00 UTC checkpoint
     the page had been unchanged for ~3.2 h, giving a factor of ~0.24,
     so the 55 practice medians sum to ~11.4% against the known 04:30
     answer aggregate of 10.55% (persistence sums to 47.77%).
  6. Spread on the logit scale: random-walk 0.5*sqrt(horizon_days) plus a
     0.45 base, plus disagreement between trend and cohort, plus a 0.8
     flat-history penalty when a trophy's rates never moved in the case
     history (a stale page hides the next refresh); capped at 2.0.
  7. The 100% first-mission trophy is pinned at 99.9 (all quantiles).

Quantiles use the standard normal z-values at the case's levels.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

# --- constants ------------------------------------------------------------

Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

TREND_WEIGHT = 0.65          # blend weight on the logit trend vs cohort
BASE_SPREAD = 0.45           # logit spread floor
RANDOM_WALK = 0.5            # random-walk coefficient on sqrt(days)
FLAT_PENALTY = 0.8           # extra spread when a trophy's history is flat
MAX_SPREAD = 2.0             # logit spread cap
REFRESH_HORIZON_MIN = 720    # refresh prior only for horizons under 12 h
PLAYER_JUMP_PER_HOUR = 1.5   # pending player influx per stale hour
HOLDER_GROWTH_PER_HOUR = 0.12  # holder growth per stale hour
PINNED_RATE = 99.9           # the 100% first-mission trophy

CEILING_PLATINUM = 0.6
CEILING_CHAPTER_1 = 45.0
CEILING_CHAPTER = 30.0
CEILING_MISSION = 15.0
CEILING_GRIND = 3.0

RATE_MIN = 0.01
RATE_MAX = 99.99


# --- small helpers --------------------------------------------------------

def parse_time(text):
    """Parse an ISO 8601 timestamp (with Z or offset) or a plain date."""
    if text is None:
        return None
    text = str(text).strip()
    if "T" not in text:
        text += "T00:00:00Z"
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(rate):
    return min(RATE_MAX, max(RATE_MIN, float(rate)))


def logit(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    if value > 30.0:
        return RATE_MAX
    if value < -30.0:
        return RATE_MIN
    return 100.0 / (1.0 + math.exp(-value))


def least_squares(xs, ys):
    """Return (slope, intercept) of the least-squares line through xs, ys."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, ys[0]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def archetype_ceiling(trophy):
    """Week-one ceiling (percent) from a trophy's tier and description."""
    if trophy.get("tier") == "platinum":
        return CEILING_PLATINUM
    desc = (trophy.get("description") or "").strip()
    low = desc.lower()
    if "chapter 1" in low:
        return CEILING_CHAPTER_1
    if "chapter" in low:
        return CEILING_CHAPTER
    if "mission strand" in low or "final mission" in low:
        return CEILING_MISSION
    if desc.startswith("Complete ") and not _starts_with_number(desc):
        return CEILING_MISSION
    return CEILING_GRIND


def _starts_with_number(desc):
    rest = desc[len("Complete "):].strip()
    return rest[:1].isdigit()


def page_staleness_hours(history, checkpoint):
    """Hours the page has sat unchanged: checkpoint minus the last read
    where any rate moved (or the first read, if nothing ever moved)."""
    if not history:
        return 0.0
    last_change = parse_time(history[0]["at"])
    prev = None
    for read in history:
        rates = read.get("rates") or []
        if prev is not None and rates != prev:
            last_change = parse_time(read["at"])
        prev = rates
    first = parse_time(history[0]["at"])
    anchor = last_change if last_change is not None else first
    delta = (checkpoint - anchor).total_seconds() / 3600.0
    return max(0.0, delta)


# --- per-trophy forecast --------------------------------------------------

def forecast_trophy(trophy, idx, history, release, target, horizon_minutes,
                   checkpoint):
    """Return the five quantiles for one trophy (percent, non-decreasing)."""
    last_rate = None
    for read in reversed(history):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            last_rate = float(rates[idx])
            break
    if last_rate is None:
        last_rate = 0.01

    # The 100% first-mission trophy is pinned at 99.9.
    if last_rate >= 99.99:
        return [PINNED_RATE] * 5

    xs, ys = [], []
    pxs, pys = [], []   # players power law: ln players on ln(1 + hours)
    hxs, hys = [], []   # holders power law: ln holders on ln(1 + hours)
    flat = True
    seen = None
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        rate = float(rates[idx])
        if seen is None:
            seen = rate
        elif rate != seen:
            flat = False
        at = parse_time(read.get("at"))
        if at is None:
            continue
        hours = max(0.0, (at - release).total_seconds() / 3600.0)
        x = math.log1p(hours)
        xs.append(x)
        ys.append(logit(rate))
        players = read.get("players_tracked")
        if players:
            holders = max(rate * float(players) / 100.0, 0.01)
            pxs.append(x)
            pys.append(math.log(float(players)))
            hxs.append(x)
            hys.append(math.log(holders))

    target_hours = max(0.0, (target - release).total_seconds() / 3600.0)
    x_target = math.log1p(target_hours)

    # 1. Logit trend on ln(1 + hours since release).
    if len(xs) >= 2:
        slope, intercept = least_squares(xs, ys)
        trend_logit = intercept + slope * x_target
    elif xs:
        trend_logit = ys[-1]
    else:
        trend_logit = logit(last_rate)

    # 2. Cohort model: holders and players as power laws.
    cohort_logit = None
    if len(pxs) >= 2 and len(hxs) >= 2:
        slope_p, int_p = least_squares(pxs, pys)
        slope_h, int_h = least_squares(hxs, hys)
        players_tgt = math.exp(int_p + slope_p * x_target)
        holders_tgt = math.exp(int_h + slope_h * x_target)
        if players_tgt > 0:
            cohort_logit = logit(100.0 * holders_tgt / players_tgt)
    if cohort_logit is None:
        cohort_logit = logit(last_rate)

    # 3. Blend, weighted toward the trend.
    median_logit = TREND_WEIGHT * trend_logit + (1.0 - TREND_WEIGHT) * cohort_logit

    # 5. Short-horizon refresh prior: shrink toward the refresh-adjusted
    #    rate when the target is under 12 h away.
    if horizon_minutes < REFRESH_HORIZON_MIN and history:
        stale = page_staleness_hours(history, checkpoint) if checkpoint else 0.0
        players_mult = 1.0 + PLAYER_JUMP_PER_HOUR * stale
        holders_mult = 1.0 + HOLDER_GROWTH_PER_HOUR * stale
        median_logit += math.log(holders_mult / players_mult)

    # 4. Archetype ceiling, applied to the median before the spread.
    ceiling = archetype_ceiling(trophy)
    median = min(inv_logit(median_logit), ceiling)
    median_logit = logit(median)

    # 6. Spread: random walk + base + trend/cohort disagreement +
    #    flat-history penalty, capped.
    horizon_days = max(horizon_minutes, 0) / 1440.0
    disagreement = min(0.5, 0.5 * abs(trend_logit - cohort_logit))
    spread = RANDOM_WALK * math.sqrt(horizon_days) + BASE_SPREAD + disagreement
    if flat:
        spread += FLAT_PENALTY
    spread = min(spread, MAX_SPREAD)

    quantiles = []
    for z in Z_VALUES:
        q = inv_logit(median_logit + z * spread)
        q = min(q, ceiling)
        quantiles.append(round(max(RATE_MIN, min(RATE_MAX, q)), 4))
    # Keep the quantiles non-decreasing after rounding.
    for i in range(1, len(quantiles)):
        quantiles[i] = max(quantiles[i], quantiles[i - 1])
    return quantiles


# --- case driver ----------------------------------------------------------

def run_case(case):
    """Forecast one case dict, return the output dict."""
    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    if target is None:
        target = release
    checkpoint = parse_time(case.get("at"))
    horizon = int(case.get("horizon_minutes") or 0)
    history = case.get("history") or []
    trophies = case.get("trophies") or []

    rows = []
    for idx, trophy in enumerate(trophies):
        rows.append({
            "id": trophy.get("id"),
            "q": forecast_trophy(trophy, idx, history, release,
                                 target, horizon, checkpoint),
        })
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    case_files = sorted(glob.glob(os.path.join(input_dir, "*.in")))
    if not case_files:
        print("no case files in %s" % input_dir, file=sys.stderr)
        return 1
    for path in case_files:
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = run_case(case)
        out_path = os.path.join(output_dir, case.get("id", "case") + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
