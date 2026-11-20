"""Data preparation, hypothesis tests, and Ridge forecasting for ClimateSense.

No Streamlit dependency, so the scientific calculations can be tested separately.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SECONDS_PER_DAY = 86400
INTERVAL_MIN = 15
HORIZON_STEPS = 16
DHT11_TEMP_ERR_C = 2.0  # Assumed sensor margin supplied by project team.
DHT11_HUM_ERR_PCT = 5.0  # Percentage points, not percent of the RH reading.


def make_tod_features(tod_seconds: Any) -> pd.DataFrame:
    """Recreate the two cyclical inputs used in the original Colab model."""
    secs = np.asarray(tod_seconds, dtype=float) % SECONDS_PER_DAY
    angle = 2 * np.pi * secs / SECONDS_PER_DAY
    return pd.DataFrame({"sin_tod": np.sin(angle), "cos_tod": np.cos(angle)})


def parse_time_to_sec(raw_str: str) -> int:
    s = raw_str.strip()
    if not s:
        raise ValueError("Enter a start time, e.g. 22:15 or 10:15 PM.")
    if s.isdigit():
        return int(s) % SECONDS_PER_DAY
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M%p", "%I:%M:%S %p"):
        try:
            t = datetime.strptime(s.upper(), fmt)
            return t.hour * 3600 + t.minute * 60 + t.second
        except ValueError:
            pass
    raise ValueError("Invalid time. Try 22:15 or 10:15 PM.")


def generate_demo_data() -> pd.DataFrame:
    """Synthetic measurements for UI demonstration, NEVER presented as sensor readings."""
    rng = np.random.default_rng(52)
    rows = []
    for day in range(3):
        date = pd.Timestamp("2026-11-17") + pd.Timedelta(days=day)
        for offset in range(0, 61):  # 08:00 to 23:00 inclusive, 15 min cadence
            tod = 8 * 3600 + offset * INTERVAL_MIN * 60
            h = tod / 3600
            temp = (28.1 + 2.7 * np.sin((h - 8) / 15 * np.pi)
                    + day * .25 + rng.normal(0, .24))
            hum = 75 - (3.15 + .18 * day) * (temp - 28) + rng.normal(0, 1.35)
            rows.append({"Date": str(date.date()), "Time_sec": tod,
                         "Temperature_C": round(temp, 2),
                         "Humidity_percent": round(float(np.clip(hum, 0, 100)), 2)})
    return pd.DataFrame(rows)


def prepare_data(raw: pd.DataFrame, split_counts: tuple[int, ...] = ()) -> tuple[pd.DataFrame, list[str]]:
    """Validate readings, preserve chronological order, and add _day/_time columns.

    Supports a Timestamp/Datetime column, Date + Time_sec, Day + Time_sec,
    or Time_sec alone. If dates are absent, optional split_counts can assign
    user-confirmed recording sessions by original row count. Unverified date
    boundaries are never silently invented.
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Match known names case-insensitively while preserving user-provided content.
    canonical = {"temperature_c": "Temperature_C", "humidity_percent": "Humidity_percent",
                 "time_sec": "Time_sec", "timestamp": "Timestamp", "datetime": "Timestamp",
                 "date": "Date", "day": "Day"}
    df = df.rename(columns={c: canonical[c.lower()] for c in df.columns if c.lower() in canonical})
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names were found after normalizing names.")
    required = {"Temperature_C", "Humidity_percent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("CSV is missing: " + ", ".join(sorted(missing)))
    warnings = []
    original_len = len(df)
    for field in ("Temperature_C", "Humidity_percent"):
        df[field] = pd.to_numeric(df[field], errors="coerce")
    if "Timestamp" in df:
        df["_timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        if df["_timestamp"].isna().any():
            warnings.append("Rows with invalid timestamps were removed.")
        df = df.dropna(subset=["_timestamp"]).sort_values("_timestamp", kind="stable")
        df["Time_sec"] = (df["_timestamp"].dt.hour * 3600 +
                          df["_timestamp"].dt.minute * 60 +
                          df["_timestamp"].dt.second)
        df["_day"] = df["_timestamp"].dt.strftime("%d %b %Y")
    elif "Time_sec" in df:
        df["Time_sec"] = pd.to_numeric(df["Time_sec"], errors="coerce")
        df = df.dropna(subset=["Time_sec"])
        if "Date" in df:
            dates = pd.to_datetime(df["Date"], errors="coerce")
            if dates.isna().any():
                raise ValueError("Date column contains invalid dates. Use YYYY-MM-DD.")
            df["_timestamp"] = dates + pd.to_timedelta(df["Time_sec"], unit="s")
            df = df.sort_values("_timestamp", kind="stable")
            df["_day"] = df["_timestamp"].dt.strftime("%d %b %Y")
        elif "Day" in df:
            # Day numbers or labels are accepted; row order is assumed chronological.
            df["_day"] = df["Day"].astype(str)
            warnings.append("No dates supplied; rows are assumed to be in recording order within each Day.")
        elif split_counts:
            counts = tuple(int(v) for v in split_counts)
            if any(c <= 0 for c in counts) or sum(counts) != len(raw):
                raise ValueError(f"The day counts must be positive and sum to {len(raw)} raw rows.")
            # Counts refer to the original CSV row order; we do not fabricate dates.
            labels = np.repeat([f"Day {i}" for i in range(1, len(counts) + 1)], counts)
            df["_day"] = pd.Series(labels, index=raw.index).reindex(df.index)
            warnings.append("Recording days were assigned from your entered row counts, "
                            "not verified calendar dates. Confirm the boundaries match your collection log.")
        else:
            # Time_sec values may be a running elapsed counter instead of seconds
            # since midnight. That is detectable when values exceed one day.
            t = df["Time_sec"].astype(float)
            if (t >= SECONDS_PER_DAY).any() and t.is_monotonic_increasing:
                elapsed_day = np.floor(t / SECONDS_PER_DAY).astype(int)
                df["_day"] = (elapsed_day - elapsed_day.iloc[0] + 1).map(lambda x: f"Session {x}")
                warnings.append("Sessions were inferred from an elapsed-seconds counter, "
                                "not verified calendar dates.")
            else:
                rollovers = t.diff().fillna(0) < 0
                df["_day"] = (rollovers.cumsum() + 1).map(lambda x: f"Session {x}")
                warnings.append("No Date/Day column: sessions inferred from clock resets. "
                                "If these 288 readings span three days without resets, "
                                "use 'Split into recording days' in the sidebar and verify counts.")
    else:
        raise ValueError("Provide Time_sec (seconds since midnight), or a Timestamp column.")
    df = df.dropna(subset=["Temperature_C", "Humidity_percent"]).copy()
    if len(df) != original_len:
        warnings.append(f"Removed {original_len - len(df)} row(s) with missing/invalid readings or time.")
    if df.empty:
        raise ValueError("No valid readings were found in the CSV.")
    if ((df["Humidity_percent"] < 0) | (df["Humidity_percent"] > 100)).any():
        raise ValueError("Humidity_percent has values outside 0–100. Correct the readings first.")
    # No smoothing, interpolation or cross-day forward fill: report real valid readings.
    df["_tod"] = df["Time_sec"].astype(float) % SECONDS_PER_DAY
    if df.duplicated(subset=["_day", "_tod"]).any():
        warnings.append("Some day/time pairs repeat. Check whether readings were recorded twice.")
    df["_order"] = np.arange(len(df))
    return df.reset_index(drop=True), warnings


def correlation_stats(frame: pd.DataFrame) -> dict[str, float | int | None]:
    clean = frame[["Temperature_C", "Humidity_percent"]].dropna()
    n = len(clean)
    result: dict[str, float | int | None] = {"n": n, "r": None, "r2": None,
                                            "slope": None, "intercept": None}
    if n < 3:
        return result
    x = clean["Temperature_C"].to_numpy(dtype=float)
    y = clean["Humidity_percent"].to_numpy(dtype=float)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return result
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, deg=1)
    result.update(r=r, r2=r * r, slope=float(slope), intercept=float(intercept))
    return result


def daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for day, subset in df.groupby("_day", sort=False):
        metrics = correlation_stats(subset)
        records.append({"Day": day, "Readings": metrics["n"], "Pearson r": metrics["r"],
                        "R²": metrics["r2"], "Slope (RH pp/°C)": metrics["slope"],
                        "Mean temp (°C)": subset["Temperature_C"].mean(),
                        "Mean RH (%)": subset["Humidity_percent"].mean()})
    return pd.DataFrame(records)


def train_models(df: pd.DataFrame) -> dict[str, Any]:
    """Train the 0..240 min time-of-day Ridge model, with 4 purged time splits.

    Targets are kept within a day/session. Only anchors with a complete 15 min
    spaced 4-hour future history are eligible; thus a row shift = 15 minutes.
    """
    features = make_tod_features(df["_tod"])
    grouped = df.groupby("_day", sort=False)
    targets = {}
    is_regular = pd.Series(True, index=df.index)
    for step in range(HORIZON_STEPS + 1):
        minutes = step * INTERVAL_MIN
        targets[f"temp_+{minutes}m"] = grouped["Temperature_C"].shift(-step)
        targets[f"hum_+{minutes}m"] = grouped["Humidity_percent"].shift(-step)
        if step:
            future_t = grouped["_tod"].shift(-step)
            expected = step * INTERVAL_MIN * 60
            is_regular &= (future_t - df["_tod"] - expected).abs().le(90)
    y_all = pd.DataFrame(targets, index=df.index)
    mask = y_all.notna().all(axis=1) & is_regular
    x = features.loc[mask].reset_index(drop=True)
    y = y_all.loc[mask].reset_index(drop=True)
    n = len(x)
    if n < 20:
        raise ValueError(
            f"Only {n} complete 4-hour training windows found. "
            "Need at least 20 windows with uninterrupted 15-minute readings "
            "within individual days; the monitoring and hypotheses still work."
        )
    # Prefer validating on future *whole days*: no shifted targets can
    # overlap the validation day. This also tests generalization across days.
    days = df.loc[mask, "_day"].reset_index(drop=True).astype(str).to_numpy()
    unique_days = list(dict.fromkeys(days))
    folds = []
    validation_description = "Expanding train-on-earlier-days validation"
    if len(unique_days) >= 2:
        for idx in range(1, len(unique_days)):
            train_days = set(unique_days[:idx])
            tr = np.where(np.isin(days, list(train_days)))[0]
            va = np.where(days == unique_days[idx])[0]
            if len(tr) >= 10 and len(va) >= 5:
                folds.append((tr, va))
        folds = folds[-4:]
    else:
        # With one day only, purge 16 anchors between training and validation
        # because training targets can reach as far as 4 hours into the future.
        if n >= 72:
            folds = list(TimeSeriesSplit(n_splits=2, gap=HORIZON_STEPS).split(x))
            validation_description = "Purged within-day TimeSeriesSplit (gap=16)"
    if not folds:
        raise ValueError(
            "Not enough complete day-wise windows for validation. "
            "Please collect longer, uninterrupted 15-minute sequences; "
            "the monitoring and hypothesis analysis still work."
        )
    scores = {col: [] for col in y.columns}
    for tr, va in folds:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        pipe.fit(x.iloc[tr], y.iloc[tr])
        predictions = pipe.predict(x.iloc[va])
        for j, col in enumerate(y.columns):
            scores[col].append(float(mean_absolute_error(y.iloc[va, j], predictions[:, j])))
    final = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    final.fit(x, y)
    return {"model": final, "columns": list(y.columns),
            "mae": {key: float(np.mean(val)) for key, val in scores.items()},
            "windows": n, "folds": len(folds), "validation": validation_description}


def generate_forecast(model_data: dict[str, Any], start_tod: int,
                      temp_error: float = DHT11_TEMP_ERR_C,
                      hum_error: float = DHT11_HUM_ERR_PCT) -> pd.DataFrame:
    all_steps = list(range(HORIZON_STEPS + 1))
    x_query = make_tod_features([start_tod])
    values = model_data["model"].predict(x_query)[0]
    lookup = dict(zip(model_data["columns"], values))
    rows = []
    for step in all_steps:
        minutes = step * INTERVAL_MIN
        pred_temp = float(lookup[f"temp_+{minutes}m"])
        pred_hum = float(lookup[f"hum_+{minutes}m"])
        mae_t = model_data["mae"][f"temp_+{minutes}m"]
        mae_h = model_data["mae"][f"hum_+{minutes}m"]
        future_tod = (start_tod + minutes * 60) % SECONDS_PER_DAY
        rows.append({
            "Offset": f"+{minutes}m", "Minutes": minutes,
            "Clock time": f"{int(future_tod // 3600):02d}:{int((future_tod % 3600) // 60):02d}",
            "Temperature (°C)": pred_temp,
            "Humidity (%)": pred_hum,
            "Temp combined margin (°C)": float(np.hypot(mae_t, temp_error)),
            "RH combined margin (pp)": float(np.hypot(mae_h, hum_error)),
            "Temp validation MAE (°C)": mae_t,
            "RH validation MAE (pp)": mae_h,
        })
    return pd.DataFrame(rows)
