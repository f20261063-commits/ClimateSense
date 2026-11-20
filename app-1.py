"""ClimateSense: Streamlit dashboard for Arduino UNO + DHT11.

Launch: streamlit run app.py
The demo dataset is simulated and is prominently identified as such.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from engine import (
    DHT11_HUM_ERR_PCT,
    DHT11_TEMP_ERR_C,
    HORIZON_STEPS,
    INTERVAL_MIN,
    correlation_stats,
    daily_stats,
    generate_demo_data,
    generate_forecast,
    parse_time_to_sec,
    prepare_data,
    train_models,
)

st.set_page_config(
    page_title="ClimateSense | Environmental Research Dashboard",
    page_icon="🌤️", layout="wide", initial_sidebar_state="expanded"
)

COLORS = {"ink": "#11253B", "muted": "#63758A", "blue": "#2578D8",
          "red": "#F0775E", "teal": "#078D9C", "green": "#26A781"}

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1420px; }
    .hero {
        background: linear-gradient(113deg,#102C45 0%,#124965 53%,#147C80 100%);
        color: white; padding: 30px 34px 29px; border-radius: 20px;
        margin-bottom: 18px; box-shadow: 0 12px 34px rgba(27,62,94,.16);
    }
    .hero h1 { color: #fff; font-size: clamp(2rem,4vw,3.1rem); letter-spacing: -1.4px;
               margin: 8px 0 8px; line-height: 1.08; font-weight: 760; }
    .hero p { color: #D9EBF1; margin: 0; font-size: 1rem; max-width: 850px; }
    .hero-top { color: #AAEADE; font-size: .7rem; letter-spacing: 2.1px;
                font-weight: 800; text-transform: uppercase; }
    .source-tag { background: #FDF3D5; border: 1px solid #F0D79B; color: #795912;
                  border-radius: 10px; padding: 12px 16px; margin-bottom: 14px; line-height: 1.5; }
    .source-real { background: #E7F7F3; border-color: #9AD6C6; color: #15584D; }
    .section-intro { color: #667D8E; font-size: .96rem; margin-top: -7px; margin-bottom: 16px; }
    .metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 16px 0 18px; }
    .metric-tile { background: #fff; border: 1px solid #E0E9F0; border-radius: 16px;
                   padding: 17px 20px; box-shadow: 0 4px 13px rgba(16,46,70,.035); }
    .metric-label { font-size: .74rem; color: #607587; font-weight: 800;
                    letter-spacing: .55px; text-transform: uppercase; }
    .metric-value { color: #16344D; margin: 7px 0 3px; font-size: clamp(1.28rem,2.2vw,1.9rem);
                    font-weight: 800; letter-spacing: -.5px; }
    .metric-foot { color: #7C90A0; font-size: .75rem; }
    .science-note { background:#F1F7FD; border-left: 4px solid #2D7CCA; border-radius: 8px;
                    color:#314E68; padding: 13px 16px; margin: 12px 0 20px; line-height: 1.55; }
    .method-card { background:#fff; border:1px solid #E1E8EE; padding: 16px 18px;
                   border-radius: 14px; color: #25445A; margin-bottom: 10px; }
    .method-card b { color: #15374E; }
    @media (max-width: 800px) {
      .metric-grid { grid-template-columns: repeat(2,1fr); }
      .hero { padding: 24px 18px; }
      .metric-tile { padding: 13px; }
    }
</style>
""", unsafe_allow_html=True)


def colored_metrics(items: list[tuple[str, str, str]]) -> None:
    """Small, mobile-friendly and presentation-friendly KPI cards."""
    cards = []
    for label, value, foot in items:
        cards.append(
            '<div class="metric-tile">'
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-value">{escape(value)}</div>'
            f'<div class="metric-foot">{escape(foot)}</div>'
            '</div>'
        )
    st.markdown('<div class="metric-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def fmt(value: object, decimals: int = 2) -> str:
    return "Not available" if value is None or pd.isna(value) else f"{float(value):.{decimals}f}"


def figure_style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=20, r=20, b=20, t=40),
        font=dict(family="Arial, sans-serif", color=COLORS["muted"]),
        plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EDF2F6", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EDF2F6", zeroline=False)
    return fig


def readings_plot(df: pd.DataFrame, measurement: str, title: str, unit: str) -> go.Figure:
    """One physical quantity per plot, one disconnected trace per recording day.

    Plot numeric hours rather than repeated categorical labels; this is essential
    for a three-day dataset with identical clock times across days.
    """
    fig = go.Figure()
    palette = ["#DC755A", "#0E8797", "#6659B8", "#D59539", "#2B936B", "#C05286"]
    for idx, (day, frame) in enumerate(df.groupby("_day", sort=False)):
        frame = frame.sort_values("_tod", kind="stable")
        fig.add_trace(go.Scatter(
            x=frame["_tod"] / 3600.0, y=frame[measurement],
            mode="lines+markers", name=str(day),
            line=dict(color=palette[idx % len(palette)], width=2.8),
            marker=dict(size=5),
            customdata=frame["_tod"].map(lambda t: f"{int(t//3600):02d}:{int((t%3600)//60):02d}"),
            hovertemplate="<b>%{fullData.name}</b> · %{customdata}<br>" +
                          title + ": %{y:.2f} " + unit + "<extra></extra>",
        ))
    fig.update_xaxes(title_text="Time of day", tickvals=list(range(0,25,3)),
                     ticktext=[f"{hour:02d}:00" for hour in range(0,25,3)],
                     range=[0,24], type="linear")
    fig.update_yaxes(title_text=f"{title} ({unit})")
    if measurement == "Humidity_percent":
        fig.update_yaxes(range=[0,100])
    return figure_style(fig, 355)



def correlation_plot(df: pd.DataFrame, metrics: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Temperature_C"], y=df["Humidity_percent"],
                             mode="markers", name="Recorded readings",
                             marker=dict(size=9, color=COLORS["teal"], opacity=.8,
                                         line=dict(width=1, color="white"))))
    if metrics["slope"] is not None:
        xs = np.linspace(df["Temperature_C"].min(), df["Temperature_C"].max(), 100)
        ys = metrics["slope"] * xs + metrics["intercept"]
        fig.add_trace(go.Scatter(x=xs, y=ys, name="Least-squares regression",
                                 mode="lines", line=dict(color=COLORS["red"], width=3)))
    fig.update_xaxes(title_text="Temperature (°C)")
    fig.update_yaxes(title_text="Relative humidity (%)")
    return figure_style(fig, 420)


def forecast_plot(preds: pd.DataFrame, measurement: str, margin: str,
                  color: str, shade: str, label: str) -> go.Figure:
    fig = go.Figure()
    x = preds["Minutes"]
    val = preds[measurement]
    unc = preds[margin]
    fig.add_trace(go.Scatter(x=x, y=val+unc, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=val-unc, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=shade,
                             name="Combined margin (illustrative)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=val, mode="lines+markers", name="Model prediction",
                             line=dict(color=color, width=3), marker=dict(size=6)))
    fig.update_xaxes(title_text="Minutes after entered time", dtick=30)
    fig.update_yaxes(title_text=label)
    return figure_style(fig, 370)


@st.cache_data(show_spinner=False)
def cached_prepare(content: bytes, split_counts: tuple[int, ...] = ()) -> tuple[pd.DataFrame, list[str]]:
    return prepare_data(pd.read_csv(BytesIO(content)), split_counts=split_counts)


@st.cache_resource(show_spinner=False)
def cached_train(content: bytes, split_counts: tuple[int, ...] = ()) -> dict:
    df, _ = cached_prepare(content, split_counts)
    return train_models(df)


def columns_in_csv(raw: pd.DataFrame) -> set[str]:
    return {str(c).strip().casefold() for c in raw.columns}


# A CSV committed to the GitHub repository becomes the default for every visitor.
# A browser upload overrides it only for the current session; it cannot modify GitHub.
default_path = Path(__file__).resolve().parent / "final_data.csv"
with st.sidebar:
    st.markdown("## 🌤️ ClimateSense")
    st.caption("Arduino UNO · DHT11 · Research Dashboard")
    st.divider()
    uploaded = st.file_uploader(
        "Upload your Arduino readings", type=["csv"],
        help="Use Date + Time_sec, Day + Time_sec, or Timestamp. "
             "Without dates, manually assign the correct recording days below.")
    if uploaded is not None:
        data_bytes = uploaded.getvalue()
        source_name = uploaded.name
        st.success(f"Uploaded for this browser session: {source_name}")
    elif default_path.is_file():
        data_bytes = default_path.read_bytes()
        source_name = "final_data.csv"
        st.success("Using saved experimental dataset: final_data.csv")
        st.caption("This CSV is stored in the project repository and loads for everyone.")
    else:
        demo = generate_demo_data()
        data_bytes = demo.to_csv(index=False).encode("utf-8")
        source_name = "Simulated demonstration data"
        st.info("Demo mode: data are simulated, not sensor measurements.")
        st.caption("Upload a real CSV. To keep it after refreshing, commit a labeled CSV named final_data.csv to GitHub.")
    is_demo = uploaded is None and not default_path.is_file()

    try:
        raw_df = pd.read_csv(BytesIO(data_bytes))
        raw_cols = columns_in_csv(raw_df)
    except Exception as err:
        st.error(f"Can't read the CSV: {err}")
        st.stop()

    split_counts: tuple[int, ...] = ()
    has_day_info = bool(raw_cols & {"date", "day", "datetime", "timestamp"})
    if not has_day_info and not is_demo:
        st.divider()
        st.markdown("### 📅 Fix recording days")
        st.caption(f"Your CSV has {len(raw_df)} rows but no Date or Day column. "
                   "Select how its original rows belong to recording days.")
        mode = st.radio("Day grouping", [
            "Automatic: detect time resets",
            "Split into equally sized days",
            "Enter row counts for each day",
        ], index=0, key="day_grouping")
        if mode == "Split into equally sized days":
            day_number = st.number_input("Number of recording days", min_value=2,
                                          max_value=20, value=3, step=1)
            if len(raw_df) % int(day_number):
                st.error("Rows do not divide equally into this many days. "
                         "Use custom row counts instead.")
                st.stop()
            per_day = len(raw_df) // int(day_number)
            st.info(f"Assumes {per_day} consecutive rows per day "
                    f"({len(raw_df)} rows ÷ {day_number} days).")
            split_counts = tuple([per_day] * int(day_number))
        elif mode == "Enter row counts for each day":
            counts_text = st.text_input("Rows per day (in CSV order)", value="96,96,96",
                                        help="For example, 80,104,104 means Day 1 has 80 rows.")
            try:
                split_counts = tuple(int(v.strip()) for v in counts_text.split(","))
                if len(split_counts) < 2 or any(v <= 0 for v in split_counts) or sum(split_counts) != len(raw_df):
                    raise ValueError
            except ValueError:
                st.error(f"Enter positive counts adding to {len(raw_df)}, such as 96,96,96 for 288 rows.")
                st.stop()
        if split_counts:
            st.warning("Only use these day boundaries if they match your actual collection log. "
                       "Equal splitting cannot prove where one day ended.")
            st.caption("After checking, download the labeled CSV below and commit it to GitHub as final_data.csv.")

    st.divider()
    st.markdown("**Prediction settings**")
    st.caption("Ridge regression (α = 10) · 15-minute steps · 4-hour horizon")
    st.markdown("**Assumed sensor margins**")
    st.write(f"🌡️ ±{DHT11_TEMP_ERR_C:g} °C")
    st.write(f"💧 ±{DHT11_HUM_ERR_PCT:g} percentage points RH")
    st.caption("CSV-based website, not a live Arduino connection.")

try:
    data, notes = cached_prepare(data_bytes, split_counts)
except Exception as err:
    st.error(f"Unable to process the dataset: {err}")
    st.code("Day,Time_sec,Temperature_C,Humidity_percent\nDay 1,28800,28.3,72.5")
    st.stop()

# Normalize the loaded measurements into a durable file with explicit Day labels.
# Users can download this file and commit it to GitHub as final_data.csv.
labeled_data = data[["_day", "Time_sec", "Temperature_C", "Humidity_percent"]].rename(
    columns={"_day": "Day"})
with st.sidebar:
    if not is_demo:
        st.download_button("⬇️ Download corrected CSV", labeled_data.to_csv(index=False),
                           file_name="final_data.csv", mime="text/csv",
                           help="Contains a Day column so the same 3-day grouping loads on every visit.")
        st.caption("Permanent for all visitors: upload this file to the GitHub repository. "
                   "Do not publish personal/sensitive information.")


st.markdown("""
<div class="hero">
    <div class="hero-top">Arduino UNO × DHT11 · Data Science Experiment</div>
    <h1>ClimateSense</h1>
    <p>Monitor temperature and relative humidity, test two scientific hypotheses,
    and explore time-of-day-based 4-hour machine-learning forecasts.</p>
</div>
""", unsafe_allow_html=True)

if is_demo:
    st.markdown('<div class="source-tag"><b>SIMULATED DEMO DATA</b> — These readings were generated to preview the interface. They are <b>not</b> Arduino measurements or experimental evidence. Upload the collected CSV for your presentation.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="source-tag source-real"><b>EXPERIMENTAL CSV</b> — Showing readings from your loaded CSV. Confirm that day labels match your recording log before presenting findings.</div>', unsafe_allow_html=True)

for note in notes:
    st.warning(note)

first, second, third, fourth, fifth = st.tabs([
    "📊 Overview", "🔎 Hypothesis 1", "📅 Hypothesis 2",
    "🌦️ 4-hour Forecast", "🧪 Methods & Data"
])

with first:
    st.subheader("Environmental monitoring")
    st.markdown('<div class="section-intro">Explore recorded temperature and humidity measurements.</div>', unsafe_allow_html=True)
    total_metrics = correlation_stats(data)
    colored_metrics([
        ("Total readings", f"{len(data):,}", "Valid rows"),
        ("Mean temperature", f"{data['Temperature_C'].mean():.2f} °C", "Across uploaded dataset"),
        ("Mean humidity", f"{data['Humidity_percent'].mean():.2f}%", "Relative humidity"),
        ("Recording days", str(data["_day"].nunique()), "Days or inferred sessions"),
    ])
    days = list(data["_day"].drop_duplicates())
    if len(days) > 1:
        counts = data.groupby("_day", sort=False).size()
        st.caption("Readings by day: " + " · ".join(f"{label}: {count}" for label, count in counts.items()))
    which = st.selectbox("View readings from", ["All days"] + days)
    plot_data = data if which == "All days" else data.loc[data["_day"] == which]
    st.caption("Each day's line is separate; horizontal axis shows clock time. "
               "Temperature and humidity use different charts and units.")
    st.markdown("#### Temperature over the recording days")
    st.plotly_chart(readings_plot(plot_data, "Temperature_C", "Temperature", "°C"),
                    use_container_width=True)
    st.markdown("#### Relative humidity over the recording days")
    st.plotly_chart(readings_plot(plot_data, "Humidity_percent", "Relative humidity", "%"),
                    use_container_width=True)
    st.markdown('<div class="science-note"><b>How to read the graphs:</b> Each color represents a different recording day. '
                'The lines compare values at the same time of day; they are <b>not</b> connected across days. '
                'Use the selector above to see an individual day.</div>', unsafe_allow_html=True)
    st.subheader("Latest reading in the selected data")
    latest = plot_data.iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric("Temperature", f"{latest['Temperature_C']:.2f} °C")
    c2.metric("Relative humidity", f"{latest['Humidity_percent']:.2f}%")
    st.caption("'Latest' means the last reading in the uploaded or demo dataset — not necessarily the current live sensor value.")

with second:
    st.subheader("Hypothesis 1 · Temperature versus relative humidity")
    st.markdown('<div class="section-intro">Test whether higher temperatures are associated with lower relative humidity.</div>', unsafe_allow_html=True)
    study_day = st.selectbox("Analyze", ["All days combined"] + list(data["_day"].drop_duplicates()), key="hyp1_day")
    subset = data if study_day == "All days combined" else data.loc[data["_day"] == study_day]
    result = correlation_stats(subset)
    colored_metrics([
        ("Pearson correlation (r)", fmt(result["r"], 3), "Direction + strength"),
        ("R-squared (R²)", fmt(result["r2"], 3), "Linear fit association"),
        ("Regression slope", fmt(result["slope"], 3), "RH percentage points / °C"),
        ("Data points", str(result["n"]), "Readings analyzed"),
    ])
    if result["r"] is None:
        st.warning("Pearson r and linear slope need at least three readings with variable temperature and humidity.")
    else:
        st.plotly_chart(correlation_plot(subset, result), use_container_width=True)
        direction = "negative" if result["r"] < 0 else "positive" if result["r"] > 0 else "approximately zero"
        if result["r"] < 0:
            summary = "As temperature increases, relative humidity tends to decrease in these readings."
        elif result["r"] > 0:
            summary = "As temperature increases, relative humidity also tends to increase in these readings."
        else:
            summary = "There is little linear association in these readings."
        st.markdown(f'<div class="science-note"><b>Interpretation:</b> The correlation is <b>{direction}</b>. {summary} '
                    f'The fitted slope is <b>{fmt(result["slope"])} percentage points RH per 1 °C</b>. '
                    f'Here, R² ≈ <b>{fmt(result["r2"]*100, 1)}%</b> is the proportion of variation in RH explained by this simple line within the selected sample, '
                    'not the accuracy of the weather forecast.</div>', unsafe_allow_html=True)
    st.caption("Correlation describes association, not proof that temperature causes humidity changes. Sensor uncertainty is not included in this simple regression fit.")

with third:
    st.subheader("Hypothesis 2 · Does the relationship persist across days?")
    st.markdown('<div class="section-intro">Compare the direction and strength of the temperature–humidity relationship across recorded days.</div>', unsafe_allow_html=True)
    day_table = daily_stats(data)
    if len(day_table) < 2:
        st.warning("Multi-day comparisons need data from at least two days. Add a Date or Day column and more measurements.")
    else:
        available = day_table.dropna(subset=["Pearson r"])
        if len(available) >= 2:
            negative_days = int((available["Pearson r"] < 0).sum())
            slope_range = available["Slope (RH pp/°C)"].max() - available["Slope (RH pp/°C)"].min()
            colored_metrics([
                ("Valid days", str(len(available)), "Days with calculable r"),
                ("Negative-correlation days", str(negative_days), "Days with r below zero"),
                ("Mean daily r", fmt(available["Pearson r"].mean(), 3), "Average of daily r values"),
                ("Slope range", fmt(slope_range, 2), "Max − min (RH pp/°C)"),
            ])
            fig = go.Figure(go.Bar(
                x=available["Day"], y=available["Pearson r"],
                marker_color=[COLORS["teal"] if val < 0 else COLORS["red"] for val in available["Pearson r"]],
                text=available["Pearson r"].map(lambda x: f"{x:.3f}"), textposition="outside",
            ))
            fig.update_yaxes(title_text="Pearson correlation (r)", range=[-1.15, 1.15])
            fig.update_xaxes(title_text="Recording day")
            st.plotly_chart(figure_style(fig, 340), use_container_width=True)
            if negative_days == len(available):
                interpretation = "The direction is negative on every evaluable day, but this does not guarantee equal strength or establish causality."
            else:
                interpretation = f"Only {negative_days} of {len(available)} evaluable days have negative correlation; the direction is not consistent across all days."
            st.markdown(f'<div class="science-note"><b>Daily comparison:</b> {interpretation} Compare r and the fitted slope separately: r measures linear association strength, while slope measures the RH change per °C.</div>', unsafe_allow_html=True)
        else:
            st.warning("Too few days have sufficient variation to calculate correlation.")
    st.dataframe(
        day_table.style.format({"Pearson r":"{:.3f}", "R²":"{:.3f}",
                               "Slope (RH pp/°C)":"{:.3f}",
                               "Mean temp (°C)":"{:.2f}", "Mean RH (%)":"{:.2f}"}, na_rep="N/A"),
        use_container_width=True, hide_index=True
    )
    st.caption("Average of daily correlations is descriptive, not a pooled Pearson r. Inferred sessions are not verified calendar dates.")

# Training is intentionally lazy: only run if the forecast/methods tab is used.
# Note: Streamlit executes the code in all tab blocks on each rerun, so caching
# ensures fitting happens once per uploaded dataset rather than every interaction.
forecast_model = None
forecast_problem = None
try:
    forecast_model = cached_train(data_bytes, split_counts)
except Exception as err:
    forecast_problem = str(err)

with fourth:
    st.subheader("Time-of-day · 4-hour atmospheric forecast")
    st.markdown('<div class="section-intro">Predict at +0 minutes and at 15-minute intervals to +240 minutes.</div>', unsafe_allow_html=True)
    if forecast_problem:
        st.warning(f"Forecast unavailable with this dataset: {forecast_problem}")
        st.info("You can still use the Overview, Hypothesis 1, Hypothesis 2, and Data sections.")
    else:
        with st.form("forecast_form"):
            a, b = st.columns([3, 1])
            with a:
                time_text = st.text_input("Enter starting time", value=st.session_state.get("saved_start", "22:15"),
                                          placeholder="22:15 or 10:15 PM",
                                          help="Accepts 24-hour time, 12-hour time, or seconds after midnight.")
            with b:
                st.write(" ")
                submitted = st.form_submit_button("Generate forecast", type="primary", use_container_width=True)
        if submitted or "saved_start" not in st.session_state:
            try:
                parse_time_to_sec(time_text)
                st.session_state["saved_start"] = time_text
                st.session_state.pop("time_error", None)
            except ValueError as err:
                st.session_state["time_error"] = str(err)
        if "time_error" in st.session_state:
            st.error(st.session_state["time_error"])
        elif "saved_start" in st.session_state:
            start_tod = parse_time_to_sec(st.session_state["saved_start"])
            pred = generate_forecast(forecast_model, start_tod)
            row0, row15 = pred.iloc[0], pred.iloc[1]
            colored_metrics([
                ("At entered time · +0m", f"{row0['Temperature (°C)']:.2f} °C", f"±{row0['Temp combined margin (°C)']:.2f} °C combined margin"),
                ("At entered time · +0m", f"{row0['Humidity (%)']:.2f}%", f"±{row0['RH combined margin (pp)']:.2f} pp combined margin"),
                ("Next · +15m", f"{row15['Temperature (°C)']:.2f} °C", "Temperature prediction"),
                ("Next · +15m", f"{row15['Humidity (%)']:.2f}%", "RH prediction"),
            ])
            left, right = st.columns(2)
            with left:
                st.markdown("#### Predicted temperature")
                st.plotly_chart(forecast_plot(pred, "Temperature (°C)",
                    "Temp combined margin (°C)", COLORS["red"], "rgba(240,119,94,.15)",
                    "Predicted temperature (°C)"), use_container_width=True)
            with right:
                st.markdown("#### Predicted relative humidity")
                st.plotly_chart(forecast_plot(pred, "Humidity (%)",
                    "RH combined margin (pp)", COLORS["teal"], "rgba(7,141,156,.13)",
                    "Predicted humidity (%)"), use_container_width=True)
            st.markdown('<div class="science-note"><b>Model note:</b> This predicts from the <b>entered time of day only</b> (sin/cos features), not from a live sensor reading. +0m is also a <b>model estimate</b>, not a measured value. The shaded margin is sqrt(validation MAE² + assumed sensor margin²); it is a <b>heuristic display</b>, not a statistical confidence interval or guaranteed bound.</div>', unsafe_allow_html=True)
            table = pred[["Offset", "Clock time", "Temperature (°C)", "Temp combined margin (°C)",
                          "Humidity (%)", "RH combined margin (pp)"]].copy()
            st.markdown("#### Full prediction table")
            st.dataframe(table.round(2), use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download forecast CSV", table.to_csv(index=False),
                               file_name="climatesense_forecast.csv", mime="text/csv", type="secondary")
            if ((pred["Humidity (%)"] < 0) | (pred["Humidity (%)"] > 100)).any():
                st.warning("Some raw model predictions fall outside the physical 0–100% RH range. Treat these as model limitations.")

with fifth:
    st.subheader("Data quality & research methods")
    st.markdown('<div class="section-intro">Scientific definitions and model validation, ready to explain during your demonstration.</div>', unsafe_allow_html=True)
    x1, x2 = st.columns(2)
    with x1:
        st.markdown('<div class="method-card"><b>Hypothesis 1</b><div>Temperature and relative humidity may be negatively correlated. Tested with Pearson r and an ordinary least-squares regression slope.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="method-card"><b>Hypothesis 2</b><div>Negative association may persist over multiple days. Tested by comparing each day’s r, R², and slope.</div></div>', unsafe_allow_html=True)
    with x2:
        st.markdown('<div class="method-card"><b>Forecast model</b><div>Cyclical time-of-day inputs (sin/cos) with StandardScaler and Ridge (alpha = 10), producing +0m to +240m predictions at 15-minute increments.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="method-card"><b>Sensor + error</b><div>Assumed DHT11 margins: ±2 °C and ±5 RH percentage points. Combine each horizon’s held-out MAE with the assumed margin using root-sum-of-squares for an illustrative display.</div></div>', unsafe_allow_html=True)
    with st.expander("How are correlation, slope, and R² different?", expanded=False):
        st.markdown("**Pearson r:** direction and strength of *linear association* (−1 to +1).  "
                    "\n**Slope:** fitted change in relative humidity in percentage points for an additional 1 °C.  "
                    "\n**R²:** r² for this simple one-variable regression with intercept; it describes fit within the data, not forecast accuracy.")
    with st.expander("How is the forecast tested?", expanded=False):
        st.write("Forecast targets use later readings within the **same day/session**. Rows must be close to a 15-minute cadence. "
                 "Model evaluation favors training on earlier complete days and measuring MAE on later days. "
                 "For a sufficiently long single day, a time-series split with a 16-step gap is used.")
        st.write("Sensor margin and MAE measure different sources of variation. Adding their squares is a chosen heuristic, "
                 "not a calibrated 95% prediction interval.")
    if forecast_model is None:
        st.warning(f"Validation metrics unavailable: {forecast_problem}")
    else:
        colored_metrics([
            ("Complete training windows", str(forecast_model["windows"]), "15-minute, within day"),
            ("Validation folds", str(forecast_model["folds"]), forecast_model["validation"]),
            ("Temp MAE at +60m", f"{forecast_model['mae']['temp_+60m']:.2f} °C", "Held-out forecast error"),
            ("RH MAE at +60m", f"{forecast_model['mae']['hum_+60m']:.2f} pp", "Held-out forecast error"),
        ])
        error_rows = [{"Minutes": i * INTERVAL_MIN,
                       "Temperature MAE (°C)": forecast_model["mae"][f"temp_+{i*INTERVAL_MIN}m"],
                       "RH MAE (pp)": forecast_model["mae"][f"hum_+{i*INTERVAL_MIN}m"]}
                      for i in range(HORIZON_STEPS + 1)]
        errors = pd.DataFrame(error_rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=errors["Minutes"], y=errors["Temperature MAE (°C)"],
                                 mode="lines+markers", name="Temperature MAE (°C)",
                                 line=dict(color=COLORS["red"], width=3)))
        fig.add_trace(go.Scatter(x=errors["Minutes"], y=errors["RH MAE (pp)"],
                                 mode="lines+markers", name="RH MAE (percentage points)",
                                 line=dict(color=COLORS["teal"], width=3)))
        fig.update_xaxes(title_text="Forecast horizon (minutes)")
        fig.update_yaxes(title_text="Validation mean absolute error (different units)")
        st.plotly_chart(figure_style(fig, 340), use_container_width=True)
        st.caption("Curves use different units and should not be compared numerically as if on a common scale.")
    st.markdown("#### Dataset preview")
    columns = ["_day", "Time_sec", "Temperature_C", "Humidity_percent"]
    if "_timestamp" in data:
        columns = ["_timestamp"] + columns
    st.dataframe(data[columns].head(150), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download cleaned measurements CSV", labeled_data.to_csv(index=False),
                       file_name="final_data.csv", mime="text/csv")
    if is_demo:
        st.caption("Downloaded demo readings are synthetic and must not be submitted as experimental observations.")

st.divider()
st.caption("ClimateSense · Arduino UNO + DHT11 · Environmental research and model demonstration · Not a live weather service")
