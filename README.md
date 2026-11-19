# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/qienzdork/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/qienzdork/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z — a week after launch — as five quantiles (5/25/50/75/95%) per trophy, scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only) reads a case's history of Exophase reads and, per trophy:

1. **Trend fit** — least squares of logit(rate) on ln(1 + hours since release). The honest signal once the history already moves.
2. **Cohort model** — holders (rate × players tracked) and players tracked are each fitted as power laws in (1 + hours since release) and extrapolated separately, then rate = holders / players. This separates the two forces moving a rate: players progressing (holders grow) vs new players joining (tracked grows, diluting the rate). Early on dilution wins and rates fall even while holders grow; later, progression wins for story trophies.
3. **Blend** — the median averages the trend and cohort forecasts in logit space, weighted toward the trend when the history shows real movement, then clamped to a week-one ceiling per trophy archetype (story chapters climb, grind/collectible trophies barely move in a week, 100% completion and the Platinum stay near zero).
4. **Refresh prior (short horizons only)** — Exophase refreshes its counts every few hours and each refresh brings a big cohort of new tracked players, so a 90-minute horizon straddling a refresh drops the rates (launch night: 638 → 3,868 tracked, every rate but the 100% one down). The median is shifted down by an archetype-dependent amount that fades with horizon; for the multi-day final horizon it is zero.
5. **Spread** — a random-walk term growing with √days, base model uncertainty, the disagreement between the trend and cohort forecasts, and a penalty when the history is flat (no signal).

Priors are calibrated on the live Exophase page (fetched 2026-11-19 ~08:51 UTC: Chapter 1 at 0.49%, most trophies 0.10–0.23%, the first mission at 100%) and on the practice cases' known aggregate: the 04:30 refresh read had 3,868 players and 4,276 trophies earned, so the 54 non-100% trophies sum to 10.55% — the forecast's medians sum to 11.37% on the practice cases, vs 47.77% for the persistence baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic, no network, ~0.1s per case. Reads every `*.in` in the input directory and writes one `*.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
