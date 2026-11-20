# ClimateSense — Arduino UNO + DHT11 research dashboard

A Streamlit website converted from the Google Colab forecasting program, with a polished interface and two hypothesis-analysis sections.

## What it includes

- **Overview:** recorded temperature and relative humidity, time chart, latest uploaded measurement.
- **Hypothesis 1:** Pearson correlation, least-squares slope, R², scatter plot and clear interpretation.
- **Hypothesis 2:** compare those statistics day by day to assess consistency.
- **4-hour forecast:** +0m, +15m, ..., +240m using time-of-day cyclical features, StandardScaler, Ridge alpha=10. Full table and interactive graphs.
- **Methods & Data:** validation MAE, assumed DHT11 margins, data inspection and CSV export.

**The app does NOT collect live Arduino data.** It reads CSV files. The built-in 3-day demo data are **synthetic** and always clearly labeled, so no one mistakes them for collected experiment results.

## Files to upload to GitHub

Upload **all four files** into the *root of the repository* (not the ZIP archive):

1. `app.py` — website design and user interactions
2. `engine.py` — cleaned data, statistics and Ridge forecasting
3. `requirements.txt` — packages needed by Streamlit
4. `README.md` — these instructions

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free

1. Create/use your GitHub repository (`ClimateSense` is already created).
2. Click **uploading an existing file** or **Add file → Upload files** and upload the four files above.
3. **Commit changes**.
4. Go to https://share.streamlit.io and sign in with GitHub.
5. Create an app from your repository, branch `main`, entry file `app.py`.
6. Click **Deploy**. Open the URL and upload your *actual* sensor CSV.

## CSV format

For best results provide:

```csv
Date,Time_sec,Temperature_C,Humidity_percent
2026-11-17,28800,28.1,75.3
2026-11-17,29700,28.3,74.6
2026-11-17,30600,28.7,73.9
2026-11-18,28800,28.4,74.9
```

- `Time_sec`: seconds since midnight (e.g., 08:00 = 28800, 08:15 = 29700).
- `Date`: `YYYY-MM-DD`, recommended for accurate multi-day comparisons.
- `Temperature_C`: temperature in degrees Celsius.
- `Humidity_percent`: relative humidity on a 0–100 scale.
- Alternatively, one `Timestamp` column may replace Date+Time_sec, e.g. `2026-11-17 08:00:00`.
- `Day` plus `Time_sec` also works. If only `Time_sec` is present, the app assumes the uploaded row order is chronological and infers sessions when times reset; that is less reliable.

The forecast model needs continuous readings approximately every 15 minutes and a complete 4-hour future window within a day. Incomplete or irregular samples may still be analyzed in the two hypothesis sections, but cannot always support forecasting.

## Scientific notes

- **At +0 minutes:** a model **estimate**, not the latest live DHT11 reading.
- **No live-sensor input:** forecast uses only time-of-day `sin`/`cos`; it does not update from the current sensor reading.
- **Hypothesis comparison:** regression results describe association, not causation.
- **Uncertainty display:** the displayed margin `sqrt(CV_MAE² + SENSOR_MARGIN²)` is a **heuristic**, not a calibrated probabilistic interval.
- **Better evaluation:** models are validated on later *whole days* when at least two days have enough training windows; the output indicates how many validation folds were used. For a sufficiently long one-day dataset, purged time-series splits are used.
- **Two fixed sensor assumptions (from the supplied Colab code):** ±2 °C and ±5 percentage points relative humidity.
- **Change from the Colab implementation:** avoid sorting all rows by `Time_sec` across days, because that can blend different days. Targets also cannot jump between unrelated measurement sessions.

This release is ready for presentation as a **CSV-based research dashboard**. Use the real measurements for scientific conclusions.
