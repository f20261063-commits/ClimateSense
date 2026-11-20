# ClimateSense — corrected three-day dashboard (v2)

This version fixes **the three issues** reported after deployment:

1. **One day instead of three**: if the CSV lacks a Date, Day or Timestamp column, use the sidebar's **Fix recording days** control. For 288 sequential rows from three equally sized sessions, choose **Split into equally sized days → 3** to label Day 1 / 2 / 3 (96 rows each). **Only choose this if the actual collection log confirms those boundaries.** If days have different counts, enter the real row counts instead. For the most reliable result, add actual dates to your source measurements.
2. **Confusing graph**: temperature and humidity now have separate, readable graphs, using time of day on the horizontal axis. Each day's data is a **separate trace**, so points from different days are never joined into zig-zags.
3. **Data disappears on refresh**: Streamlit uploads are temporary. Download **corrected CSV** from the sidebar and commit it into your GitHub repository named exactly `final_data.csv` alongside `app.py`. The website then loads your real CSV by default on every visit, including visitors who never uploaded anything. Do not put personal or private information in a public repository.

## How to update your existing deployed site

Your repository: `f20261063-commits/ClimateSense`.

1. Open your GitHub ClimateSense repository.
2. Update **both `app.py` and `engine.py`** with this version (use the file edit pencil and paste replacement, or GitHub's **Add file → Upload files** and replace existing file names).
3. Keep your existing `requirements.txt`; this package also contains a matching copy.
4. Commit the changes. Streamlit Community Cloud should redeploy automatically.
5. Open the site and upload your original `final_data.csv`. In the sidebar under **Fix recording days**, choose **Split into equally sized days** with `3` only if the 288 rows truly are 96 per recording day. Otherwise choose **Enter row counts** and enter the correct values.
6. Check the three day totals, both graphs, and the Day 1/Day 2/Day 3 comparison.
7. Click **Download corrected CSV**, and upload that CSV to GitHub as **`final_data.csv`** in the root of your repository. GitHub public repositories expose their data to anyone. Do not do this for sensitive information.
8. Reload in an incognito window. Your saved real readings, not demo data, should appear automatically.

## CSV columns

Recommended (known dates):

```csv
Date,Time_sec,Temperature_C,Humidity_percent
2026-11-17,28800,28.1,75.3
2026-11-17,29700,28.3,74.6
2026-11-18,28800,28.4,74.9
```

Or explicit sessions (when actual dates aren't known):

```csv
Day,Time_sec,Temperature_C,Humidity_percent
Day 1,28800,28.1,75.3
Day 1,29700,28.3,74.6
Day 2,28800,28.4,74.9
```

`Time_sec` is seconds since midnight; `Temperature_C` is Celsius; `Humidity_percent` is RH in percentage points.

## Forecast caveat

The time-of-day Ridge forecast is preserved (+0 to +240 min every 15 min). It requires **consecutive 15-minute readings within each day** to form training targets. If data intervals are not 15 minutes, monitoring and correlation will work, but forecasting will report that there aren't enough suitable windows. Estimated ± margins are heuristic, not calibrated probability intervals. Don't describe them as guaranteed error limits.

The site does not connect live to Arduino. Visitors can upload a temporary CSV themselves; only a committed dataset is automatic on reload.
