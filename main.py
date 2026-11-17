"""GTA VI launch-day Twitch forecast, version 2.

Reads one case file per run (``$MAKEFUN_INPUT_DIR`` or ``/input``) and
writes one forecast per case (``$MAKEFUN_OUTPUT_DIR`` or ``/output``):
for each of hours 2..24 of the release day (UTC), five quantiles of the
concurrent-viewer count in the game's Twitch category, at levels
0.05 / 0.25 / 0.5 / 0.75 / 0.95.

The model is an ensemble fitted fresh from the case's own comparables
at run time -- nothing is hardcoded about any game, and nothing is
carried over from the practice cases.  Everything runs on the log
scale, because a miss of 20% costs the same at 10,000 viewers as at
1,000,000.

Two signals describe a launch's scale:

* ``p1`` -- the hour-1 print.  A launch whose first hour already counts
  hundreds of thousands of viewers has most of its audience in hand
  and ramps little; a launch whose first hour counts a few thousand
  ramps many times over as the world wakes up and unlocks.
* ``pre7`` -- the category's mean hourly viewers over the 7 days
  before release.  A category that was already busy has less headroom
  left than one that was quiet.  For the final case the case's own
  history only reaches back to 2026-11-17, so the matching
  comparable's hourly history (same Twitch category) extends it.

Two forecasts are built from the comparables and averaged in log
space (geometric mean of the quantiles):

* LEVEL -- the median comparable curve (per hour, the median log
  ratio to hour 1) plus one scalar level shift.  The shift is each
  comparable's mean residual against the median curve, regressed
  (ordinary least squares) on ``[1, ln p1, ln pre7]``.  The shape
  stays the comparables' own shape; only its level moves.
* HOURLY -- per hour, an OLS fit of ``ln(y_h)`` on
  ``[1, ln p1, ln pre7]`` over the comparables, evaluated at the
  case's own ``p1`` and ``pre7``.

Spreads: for each hour, the residual standard deviation of the
comparables nearest the case in ``ln p1`` (12 nearest).  Residual
spread is heteroscedastic -- small launches are far wilder than big
ones -- so one global spread would be far too wide for a big launch
and far too narrow for a small one.

Version 2 changes one thing against version 1: the spread is
calibrated.  The raw local spread is an over-estimate of the
predictive spread -- it measures how far comparable curves sit from
the fitted median, but the case-specific fit already absorbs much of
that variation, so the intervals came out too wide (90% coverage 1.00
on three of the four practice cases, when an honest forecast covers
about 0.90).  Shrinking the spread by 0.7 brings coverage to
0.96 / 0.91 / 1.00 / 0.83 and cuts the leave-one-out pinball loss
from 0.1697 to 0.1539, improving every practice case except the one
whose loss is dominated by median error.  The factor 0.7 sits in the
middle of a flat optimum (0.65 gives 0.1525, 0.75 gives 0.1558), so
it is not a knife-edge tuning.  Spreads stay clamped to
[0.20, 1.00] log units after shrinking.

What version 2 deliberately does NOT change: the pre7 coefficient.
Shrinking it even to 0.75 destroys the model (leave-one-out loss
0.37), because the pre-release level carries most of the scale
signal; the coefficient has to stay as fitted.  The medians are
therefore identical to version 1, including the shape at very large
hour-1 prints (peak around hour 18, ratio about 1.4x at p1 = 1M).

If there are too few comparables to fit anything, the entry falls
back to the persistence forecast (every hour stays at hour 1 with a
widening random-walk spread), so the output is always valid.

Run as ``python3 -P -s main.py``.  Standard library only, no
network, no randomness, fully deterministic.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0, 0.674, 1.645]
HOURS = list(range(2, 25))
NEAR = 12            # comparables used for a local (per-hour) spread
SPREAD_SHRINK = 0.7  # v2 calibration: raw local spread is too wide (see docstring)
SPREAD_MIN = 0.20    # log units; narrower than this is not credible
SPREAD_MAX = 1.0     # log units; wider than this is not credible
FLAT_SIGMA = 0.15    # fallback only: random-walk spread per sqrt hour


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day(hourly, day_start):
    """A launch's release-day hourly counts as {hour: viewers} (1..24)."""
    start = parse_time(day_start)
    counts = {}
    for row in hourly or []:
        viewers = row.get("viewers")
        if not viewers:
            continue
        at = parse_time(row["from"])
        if start <= at < start + timedelta(days=1):
            counts[at.hour + 1] = float(viewers)
    return counts


def pre_release_avg(hourly, day_start, days=7):
    """Mean hourly viewers over the ``days`` days before the release day."""
    start = parse_time(day_start)
    values = []
    for row in hourly or []:
        viewers = row.get("viewers")
        if not viewers:
            continue
        at = parse_time(row["from"])
        if start - timedelta(days=days) <= at < start:
            values.append(float(viewers))
    return statistics.mean(values) if values else None


def solve_linear(design, target):
    """Least squares by Gauss-Jordan elimination with partial pivoting.

    Returns the coefficient list, or None when the system is singular.
    """
    n = len(design[0])
    ata = [[sum(row[i] * row[j] for row in design) for j in range(n)]
           for i in range(n)]
    atb = [sum(row[i] * y for row, y in zip(design, target))
           for i in range(n)]
    aug = [ata[i] + [atb[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(n):
            if r != col:
                f = aug[r][col] / aug[col][col]
                for k in range(col, n + 1):
                    aug[r][k] -= f * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def half_up(value):
    """Round to 2 decimals, halves up (as the starter rounds them)."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if rounded == int(rounded) else rounded


def make_quantiles(p1, med_log, spread):
    """Five quantiles from a log median and a log spread, rounded, valid."""
    out = []
    for z in Z_SCORES:
        q = p1 * math.exp(med_log + z * spread)
        q = min(max(q, 1.0), 1e9)
        out.append(half_up(q))
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


class Comparable:
    """One comparable launch: release-day curve, hour-1 print, pre7."""

    def __init__(self, comp):
        self.name = comp.get("name", "?")
        self.curve = release_day(comp.get("hourly"), comp["day_start"])
        self.p1 = self.curve.get(1)
        self.pre7 = pre_release_avg(comp.get("hourly"), comp["day_start"])
        self.log_p1 = math.log(self.p1) if self.p1 else None
        self.log_pre7 = math.log(self.pre7) if self.pre7 else None


def case_pre_release(case):
    """The case game's 7-day pre-release mean hourly viewers.

    From the case's own history_hourly, extended by the matching
    comparable's hourly history (same Twitch category) when the case's
    own history does not reach 7 days back -- the final case's
    recorder only started on 2026-11-17, while the GTA VI
    comparable's history reaches back to its 2026-08-27 premiere.
    Points are deduplicated by timestamp, the case's own history
    winning ties.
    """
    day_start = parse_time(case["day_start"])
    by_time = {}
    game_id = (case.get("game") or {}).get("twitch_id")
    for comp in case.get("comparables") or []:
        if game_id is not None and str(comp.get("twitch_id")) == str(game_id):
            for row in comp.get("hourly") or []:
                if row.get("viewers"):
                    by_time[parse_time(row["from"])] = float(row["viewers"])
    for row in case.get("history_hourly") or []:
        if row.get("viewers"):
            by_time[parse_time(row["from"])] = float(row["viewers"])
    vals = [v for t, v in by_time.items()
            if day_start - timedelta(days=7) <= t < day_start]
    return statistics.mean(vals) if vals else None


def nearby_scale(residuals, log_p1s, log_p1, k=NEAR):
    """Residual std of the k comparables nearest the case in ln p1."""
    order = sorted(range(len(log_p1s)), key=lambda i: abs(log_p1s[i] - log_p1))
    near = [residuals[i] for i in order[:k]]
    if len(near) > 1:
        s = statistics.pstdev(near)
    elif residuals:
        s = statistics.pstdev(residuals)
    else:
        s = 0.5
    return s


def fit_level(comps, med_curve, use_pre7):
    """Regress each comparable's mean residual vs the median curve on
    [1, ln p1 (, ln pre7)].  Returns (beta, rows) or None."""
    rows = []
    for comp in comps:
        vals = [math.log(comp.curve[h] / comp.p1) - med_curve[h - 2]
                for h in HOURS if h in comp.curve]
        if not vals:
            continue
        delta = statistics.mean(vals)
        if use_pre7 and comp.log_pre7 is not None:
            rows.append((comp.log_p1, comp.log_pre7, delta))
        elif not use_pre7:
            rows.append((comp.log_p1, delta))
    if len(rows) < 5:
        return None
    if use_pre7:
        beta = solve_linear([[1.0, a, b] for a, b, _ in rows],
                            [r[2] for r in rows])
    else:
        beta = solve_linear([[1.0, a] for a, _ in rows],
                            [r[1] for r in rows])
    return (beta, rows) if beta is not None else None


def predict_level(comps, med_curve, fit, log_p1, log_pre7):
    """Median-curve + scalar shift forecast.  Returns (med_log, spread)."""
    beta, rows = fit
    if log_pre7 is not None and len(beta) == 3:
        delta = beta[0] + beta[1] * log_p1 + beta[2] * log_pre7
    else:
        delta = beta[0] + beta[1] * log_p1
    med_log = [m + delta for m in med_curve]
    spread = []
    for h in HOURS:
        resid = []
        xs = []
        for comp in comps:
            if h not in comp.curve:
                continue
            if (log_pre7 is not None and len(beta) == 3
                    and comp.log_pre7 is not None):
                dp = (beta[0] + beta[1] * comp.log_p1
                      + beta[2] * comp.log_pre7)
            else:
                dp = beta[0] + beta[1] * comp.log_p1
            resid.append(math.log(comp.curve[h] / comp.p1)
                         - med_curve[h - 2] - dp)
            xs.append(comp.log_p1)
        spread.append(nearby_scale(resid, xs, log_p1))
    return med_log, spread


def fit_hourly(comps, use_pre7):
    """Per hour, OLS of ln(y_h) on [1, ln p1 (, ln pre7)].
    Returns {hour: (beta, residuals, log_p1s)}."""
    fits = {}
    for h in HOURS:
        rows = []
        for comp in comps:
            if h not in comp.curve:
                continue
            if use_pre7 and comp.log_pre7 is not None:
                rows.append((comp.log_p1, comp.log_pre7,
                             math.log(comp.curve[h])))
            elif not use_pre7:
                rows.append((comp.log_p1, math.log(comp.curve[h])))
        if len(rows) < 5:
            continue
        if use_pre7:
            beta = solve_linear([[1.0, a, b] for a, b, _ in rows],
                                [r[2] for r in rows])
        else:
            beta = solve_linear([[1.0, a] for a, _ in rows],
                                [r[1] for r in rows])
        if beta is None:
            continue
        if use_pre7:
            design = [[1.0, a, b] for a, b, _ in rows]
            target = [r[2] for r in rows]
        else:
            design = [[1.0, a] for a, _ in rows]
            target = [r[1] for r in rows]
        resid = [y - sum(b * x for b, x in zip(beta, row))
                 for row, y in zip(design, target)]
        fits[h] = (beta, resid, [r[0] for r in rows])
    return fits


def predict_hourly(fits, log_p1, log_pre7):
    """Per-hour OLS forecast.  Returns (med_log, spread), med in log-ratio."""
    med_log = []
    spread = []
    for h in HOURS:
        beta, resid, xs = fits[h]
        if log_pre7 is not None and len(beta) == 3:
            pred = beta[0] + beta[1] * log_p1 + beta[2] * log_pre7
        else:
            pred = beta[0] + beta[1] * log_p1
        med_log.append(pred - log_p1)
        spread.append(nearby_scale(resid, xs, log_p1))
    return med_log, spread


def flat_forecast(p1):
    """Fallback: every hour stays at hour 1, spread widening like a
    random walk (the house baseline's shape)."""
    return {"hours": [{"hour": h,
                       "q": [half_up(p1 * math.exp(z * FLAT_SIGMA
                                                   * math.sqrt(h - 1)))
                             for z in Z_SCORES]}
                      for h in HOURS]}


def forecast(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, "
                         f"not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    comps = []
    for comp in case.get("comparables") or []:
        launch = Comparable(comp)
        if launch.p1 and len(launch.curve) >= 20:
            comps.append(launch)
    comps.sort(key=lambda c: c.name)
    if len(comps) < 5:
        return flat_forecast(p1)

    med_curve = [statistics.median(
        [math.log(c.curve[h] / c.p1) for c in comps if h in c.curve])
        for h in HOURS]

    pre7 = case_pre_release(case)
    log_pre7 = math.log(pre7) if pre7 else None
    use_pre7 = log_pre7 is not None

    forecasts = []
    level_fit = fit_level(comps, med_curve, use_pre7)
    if level_fit is not None:
        forecasts.append(predict_level(comps, med_curve, level_fit,
                                       log_p1, log_pre7))
    hourly_fits = fit_hourly(comps, use_pre7)
    if len(hourly_fits) == len(HOURS):
        forecasts.append(predict_hourly(hourly_fits, log_p1, log_pre7))
    if not forecasts:
        return flat_forecast(p1)

    # Blend the halves in log space: geometric mean of the quantiles,
    # i.e. the average log median and the average log spread -- then
    # calibrate the spread (v2): the raw local spread is too wide, so
    # shrink it and clamp it to a credible band.
    med_log = [statistics.mean([f[0][i] for f in forecasts])
               for i in range(len(HOURS))]
    spread = [max(SPREAD_MIN, min(SPREAD_MAX,
                  SPREAD_SHRINK * statistics.mean(
                      [f[1][i] for f in forecasts])))
              for i in range(len(HOURS))]
    return {"hours": [{"hour": h, "q": make_quantiles(p1, med_log[i], spread[i])}
                      for i, h in enumerate(HOURS)]}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast(case, path.name)
        (OUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
