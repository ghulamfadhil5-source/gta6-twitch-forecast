[![GTA VI Twitch CCU Forecast](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ghulamfadhil5-source/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ghulamfadhil5-source/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast

A single Python program (`main.py`, standard library only) that forecasts
how many people watch Grand Theft Auto VI live on Twitch in each hour of
its release day, 2026-11-19 (UTC), after the first hour — five quantiles
(5/25/50/75/95%) per hour, hours 2–24.

## The model

Work on the log scale. The program fits everything fresh from the case's
own comparables at run time — nothing is hardcoded about any game:

- **Anchor:** hour 1's print. A launch whose first hour already counts
  hundreds of thousands of viewers has most of its audience in hand and
  ramps little; a launch whose first hour counts a few thousand ramps
  many times over as the world wakes up and unlocks.
- **Pre-release level:** the category's mean hourly viewers over the 7
  days before release day (the case's own history, extended by the
  matching comparable's hourly history when the case's own history is
  shorter). A category that was already busy before launch has less
  headroom left than one that was quiet.
- **Two halves, blended in log space:**
  - *SHIFT* — the median comparable launch-day curve, plus one scalar
    level shift regressed on (log hour-1 print, log pre-release level).
  - *OLS2* — per hour, an OLS fit of log viewers on (log hour-1 print,
    log pre-release level) over the comparables.
- **Spreads:** per hour, the residual standard deviation of the
  comparables nearest the case in log hour-1 print (12 nearest) —
  small launches are far wilder than big ones, so a single global
  spread would be wrong for both.

## Local practice score

Scored against the four practice launches' true release-day prints
(fetched from SullyGnome's public hourly data, hour-1 values verified
to match each case exactly), mean pinball loss on the log scale:

| case | game | loss |
|---|---|---|
| p00 | Marvel's Wolverine | 0.172 |
| p01 | Red Dead Redemption 2 | 0.143 |
| p02 | God of War Ragnarök | 0.172 |
| p03 | Marvel's Spider-Man 2 | 0.191 |
| **mean** | | **0.170** |

House persistence baseline on the same cases: **0.387** (verified with
the kit starter). The practice pays nothing and is no guide to the
final — it only validates the program.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs write byte-identical outputs. Runs in ~0.6 s
for all four practice cases (limit is 8 s per case).

## make.fun bounty

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
on [make.fun](https://make.fun).
