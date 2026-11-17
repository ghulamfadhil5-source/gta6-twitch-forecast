"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

One program, standard library only, no network, no randomness, fully
deterministic: it reads one case file (/input/<id>.in) and writes one
forecast (/output/<id>.out) with five quantiles of the concurrent-viewer
count for each of hours 2..24 of the release day (UTC).

The model (fit fresh from the case's own comparables at run time, nothing
hardcoded about any game):

  Work on the log scale.  Every comparable launch gives its release-day
  curve as log ratios to its own hour-1 print, ln(y_h / y_1).  Two
  predictors carry the information about a launch's scale:

    * p1  -- hour 1's print.  A launch whose first hour already counts
      hundreds of thousands of viewers has most of its audience in hand
      and ramps little; a launch whose first hour counts a few thousand
      ramps many times over as the world wakes up and unlocks.
    * pre7 -- the category's mean hourly viewers over the 7 days before
      the release day (the case's own history, extended by the matching
      comparable's hourly history when the case's own history is shorter,
      e.g. the final case's recorder only started on 2026-11-17).
      A category that was already busy before launch has less headroom
      left than one that was quiet.

  Two forecasts are built from the comparables and blended in log space
  (geometric mean of the quantiles, i.e. the average of the two log
  medians and the average of the two log spreads):

    A. SHIFT -- the median comparable curve (per hour, the median log
       ratio over all comparables) plus one scalar level shift delta,
       where delta is the mean residual of a launch's curve against the
       median curve, regressed (OLS) on [1, ln p1, ln pre7].  The shape
       stays the comparables' own shape; only its level moves.

    B. OLS2 -- per hour, an OLS fit of ln(y_h) on [1, ln p1, ln pre7]
       over the comparables, evaluated at the case's p1 and pre7.

  Spreads: for each hour, the residual standard deviation of the
  comparables nearest the case in ln p1 (12 nearest), floored at 0.25
  and capped at 1.2 log units.  Residual spread is heteroscedastic --
  small launches are far wilder than big ones -- so a global spread
  would be far too wide for a big launch and far too narrow for a
  small one.  Quantiles are q = p1 * exp(median_log + z_tau * s_h)
  with the standard normal quantiles at the five scored levels.

  If the case carries no usable pre-release history, both halves fall
  back to p1-only fits.  If there are too few comparables to fit at
  all, the entry falls back to the persistence forecast (every hour
  stays at hour 1 with a widening random-walk spread), so the output
  is always valid.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only.  It reads $MAKEFUN_INPUT_DIR
when set, else /input/, and writes to $MAKEFUN_OUTPUT_DIR when set,
else /output/.  Every *.in file found is forecast, one .out each.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
HOURS = list(range(2, 25))
NEAREST = 12          # comparables used for a local (per-hour) spread
SPREAD_FLOOR = 0.25   # log units; narrower than this is not credible
SPREAD_CAP = 1.2      # log units; wider than this is not credible
PERSIST_SIGMA = 0.15  # fallback only: random-walk spread per sqrt hour


def parse_ts(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_curve(hourly, day_start):
    """A launch's release-day hourly counts as {hour: viewers} (hour 1..24)."""
    start = parse_ts(day_start)
    out = {}
    for row in hourly or []:
        viewers = row.get("viewers")
        if not viewers:
            continue
        at = parse_ts(row["from"])
        if start <= at < start + timedelta(days=1):
            out[at.hour + 1] = float(viewers)
    return out


def pre_release_mean(hourly, day_start, days=7):
    """Mean hourly viewers over the `days` days before the release day."""
    start = parse_ts(day_start)
    vals = []
    for row in hourly or []:
        viewers = row.get("viewers")
        if not viewers:
            continue
        at = parse_ts(row["from"])
        if start - timedelta(days=days) <= at < start:
            vals.append(float(viewers))
    return statistics.mean(vals) if vals else None


def ols(design, target):
    """Least squares by Gaussian elimination with partial pivoting.

    Returns the coefficient list, or None when the system is singular.
    """
    n = len(design[0])
    ata = [[sum(r[i] * r[j] for r in design) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] * y for r, y in zip(design, target)) for i in range(n)]
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


def round2(x):
    """x rounded to 2 decimals, half up (as the starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def quantiles(p1, med_log, spread):
    """Five quantiles from a log median and a log spread, rounded, valid."""
    out = []
    for z in Z:
        q = p1 * math.exp(med_log + z * spread)
        q = min(max(q, 1.0), 1e9)
        out.append(round2(q))
    for i in range(1, 5):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


class Launch:
    """One comparable launch: release-day curve, hour-1 print, pre7."""

    def __init__(self, comp):
        self.name = comp.get("name", "?")
        self.curve = day_curve(comp.get("hourly"), comp["day_start"])
        self.p1 = self.curve.get(1)
        self.pre7 = pre_release_mean(comp.get("hourly"), comp["day_start"])
        self.log_p1 = math.log(self.p1) if self.p1 else None
        self.log_pre7 = math.log(self.pre7) if self.pre7 else None


def case_pre7(case):
    """The case game's 7-day pre-release mean hourly viewers.

    From the case's own history_hourly, extended by the matching
    comparable's hourly history (same Twitch category) when the case's
    own history does not reach 7 days back -- the final case's recorder
    only started on 2026-11-17, while the GTA VI comparable's history
    reaches back to its 2026-08-27 premiere.  Points are deduplicated
    by timestamp, the case's own history winning ties.
    """
    day_start = parse_ts(case["day_start"])
    by_time = {}
    game_id = (case.get("game") or {}).get("twitch_id")
    for comp in case.get("comparables") or []:
        if game_id is not None and str(comp.get("twitch_id")) == str(game_id):
            for row in comp.get("hourly") or []:
                if row.get("viewers"):
                    by_time[parse_ts(row["from"])] = float(row["viewers"])
    for row in case.get("history_hourly") or []:
        if row.get("viewers"):
            by_time[parse_ts(row["from"])] = float(row["viewers"])
    vals = [v for t, v in by_time.items()
            if day_start - timedelta(days=7) <= t < day_start]
    return statistics.mean(vals) if vals else None


def local_spread(residuals, log_p1s, log_p1, k=NEAREST):
    """Residual std of the k comparables nearest the case in ln p1."""
    order = sorted(range(len(log_p1s)), key=lambda i: abs(log_p1s[i] - log_p1))
    near = [residuals[i] for i in order[:k]]
    if len(near) > 1:
        s = statistics.pstdev(near)
    elif residuals:
        s = statistics.pstdev(residuals)
    else:
        s = 0.5
    return max(SPREAD_FLOOR, min(SPREAD_CAP, s))


def fit_shift(comps, med_curve, use_pre7):
    """Regress each comparable's mean residual vs the median curve on
    [1, ln p1 (, ln pre7)].  Returns (beta, deltas) or None."""
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
        beta = ols([[1.0, a, b] for a, b, _ in rows], [r[2] for r in rows])
    else:
        beta = ols([[1.0, a] for a, _ in rows], [r[1] for r in rows])
    return (beta, rows) if beta is not None else None


def predict_shift(comps, med_curve, fit, log_p1, log_pre7):
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
            if log_pre7 is not None and len(beta) == 3 and comp.log_pre7 is not None:
                dp = beta[0] + beta[1] * comp.log_p1 + beta[2] * comp.log_pre7
            else:
                dp = beta[0] + beta[1] * comp.log_p1
            resid.append(math.log(comp.curve[h] / comp.p1) - med_curve[h - 2] - dp)
            xs.append(comp.log_p1)
        spread.append(local_spread(resid, xs, log_p1))
    return med_log, spread


def fit_ols2(comps, use_pre7):
    """Per hour, OLS of ln(y_h) on [1, ln p1 (, ln pre7)].
    Returns {hour: (beta, residuals, log_p1s)}."""
    fits = {}
    for h in HOURS:
        rows = []
        for comp in comps:
            if h not in comp.curve:
                continue
            if use_pre7 and comp.log_pre7 is not None:
                rows.append((comp.log_p1, comp.log_pre7, math.log(comp.curve[h])))
            elif not use_pre7:
                rows.append((comp.log_p1, math.log(comp.curve[h])))
        if len(rows) < 5:
            continue
        if use_pre7:
            beta = ols([[1.0, a, b] for a, b, _ in rows], [r[2] for r in rows])
        else:
            beta = ols([[1.0, a] for a, _ in rows], [r[1] for r in rows])
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


def predict_ols2(fits, log_p1, log_pre7):
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
        spread.append(local_spread(resid, xs, log_p1))
    return med_log, spread


def persistence(p1):
    """Fallback: every hour stays at hour 1, spread widening like a
    random walk (the house baseline's shape)."""
    return {"hours": [{"hour": h,
                       "q": [round2(p1 * math.exp(z * PERSIST_SIGMA * math.sqrt(h - 1)))
                             for z in Z]}
                      for h in HOURS]}


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    comps = []
    for comp in case.get("comparables") or []:
        launch = Launch(comp)
        if launch.p1 and len(launch.curve) >= 20:
            comps.append(launch)
    comps.sort(key=lambda c: c.name)
    if len(comps) < 5:
        return persistence(p1)

    med_curve = [statistics.median(
        [math.log(c.curve[h] / c.p1) for c in comps if h in c.curve])
        for h in HOURS]

    pre7 = case_pre7(case)
    log_pre7 = math.log(pre7) if pre7 else None
    use_pre7 = log_pre7 is not None

    forecasts = []
    shift_fit = fit_shift(comps, med_curve, use_pre7)
    if shift_fit is not None:
        forecasts.append(predict_shift(comps, med_curve, shift_fit, log_p1, log_pre7))
    ols_fits = fit_ols2(comps, use_pre7)
    if len(ols_fits) == len(HOURS):
        forecasts.append(predict_ols2(ols_fits, log_p1, log_pre7))
    if not forecasts:
        return persistence(p1)

    # Blend the halves in log space: geometric mean of the quantiles,
    # i.e. the average log median and the average log spread.
    med_log = [statistics.mean([f[0][i] for f in forecasts]) for i in range(len(HOURS))]
    spread = [statistics.mean([f[1][i] for f in forecasts]) for i in range(len(HOURS))]
    return {"hours": [{"hour": h, "q": quantiles(p1, med_log[i], spread[i])}
                      for i, h in enumerate(HOURS)]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
