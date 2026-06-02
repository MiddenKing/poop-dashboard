from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("data/poop_data.csv")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

time_col = [c for c in df.columns if c.lower() == "time"][0]
value_col = [c for c in df.columns if c.lower() == "value"][0]

df["datetime"] = pd.to_datetime(df[time_col])
df["date"] = df["datetime"].dt.floor("D")
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.day_name()
df["is_poor"] = df[value_col].eq(0)
df["is_good"] = df[value_col].eq(1)

date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

daily = (
    df.groupby("date")
    .size()
    .reindex(date_range, fill_value=0)
    .rename("count")
    .reset_index()
    .rename(columns={"index": "date"})
)

summary = {
    "observations": int(len(df)),
    "days": int(len(daily)),
    "average_per_day": round(len(df) / len(daily), 2),
    "median_per_day": float(daily["count"].median()),
    "good": int(df["is_good"].sum()),
    "poor": int(df["is_poor"].sum()),
    "good_rate": round(df["is_good"].mean() * 100, 1),
    "poor_rate": round(df["is_poor"].mean() * 100, 1),
    "max_per_day": int(daily["count"].max()),
    "zero_poop_days": int((daily["count"] == 0).sum()),
    "days_with_poop": int((daily["count"] > 0).sum()),
    "earliest_hour": int(df["hour"].min()),
    "latest_hour": int(df["hour"].max()),
    "most_common_hour": int(df["hour"].mode().iloc[0]),
}

Path("summary.json").write_text(json.dumps(summary, indent=2))


# Plot 1: daily frequency split by good and poor
daily_quality = (
    df.groupby(["date", value_col])
    .size()
    .unstack(fill_value=0)
    .reindex(date_range, fill_value=0)
)

for v in [0, 1]:
    if v not in daily_quality.columns:
        daily_quality[v] = 0

daily_quality = daily_quality[[1, 0]]

plt.figure(figsize=(12, 5.5))
plt.bar(daily_quality.index, daily_quality[1], width=0.9, label="Good")
plt.bar(
    daily_quality.index,
    daily_quality[0],
    width=0.9,
    bottom=daily_quality[1],
    label="Poor"
)
plt.title("Daily poop frequency")
plt.xlabel("Date")
plt.ylabel("Poops per day")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "daily_frequency.png", dpi=200)
plt.close()


# Plot 2: poops by hour split by good and poor
hourly_quality = (
    df.groupby(["hour", value_col])
    .size()
    .unstack(fill_value=0)
    .reindex(range(24), fill_value=0)
)

for v in [0, 1]:
    if v not in hourly_quality.columns:
        hourly_quality[v] = 0

hourly_quality = hourly_quality[[1, 0]]

plt.figure(figsize=(12, 5.5))
plt.bar(hourly_quality.index, hourly_quality[1], label="Good")
plt.bar(
    hourly_quality.index,
    hourly_quality[0],
    bottom=hourly_quality[1],
    label="Poor"
)
plt.title("Poops by hour of day")
plt.xlabel("Hour of day")
plt.ylabel("Number of poops")
plt.xticks(range(0, 24, 2))
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "hour_of_day.png", dpi=200)
plt.close()


# Plot 3: poops by weekday split by good and poor
weekday_quality = (
    df.groupby(["weekday", value_col])
    .size()
    .unstack(fill_value=0)
    .reindex(weekday_order, fill_value=0)
)

for v in [0, 1]:
    if v not in weekday_quality.columns:
        weekday_quality[v] = 0

weekday_quality = weekday_quality[[1, 0]]

plt.figure(figsize=(12, 5.5))
plt.bar(weekday_quality.index, weekday_quality[1], label="Good")
plt.bar(
    weekday_quality.index,
    weekday_quality[0],
    bottom=weekday_quality[1],
    label="Poor"
)
plt.title("Poops by weekday")
plt.xlabel("Weekday")
plt.ylabel("Number of poops")
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "poops_by_weekday.png", dpi=200)
plt.close()

# Plot 4: cumulative poops over time
daily_cumulative = daily.copy()
daily_cumulative["cumulative_count"] = daily_cumulative["count"].cumsum()

plt.figure(figsize=(12, 5.5))
plt.plot(daily_cumulative["date"], daily_cumulative["cumulative_count"], marker="o")
plt.title("Cumulative poops over time")
plt.xlabel("Date")
plt.ylabel("Cumulative number of poops")
plt.tight_layout()
plt.savefig(FIGURES / "cumulative_poops.png", dpi=200)
plt.close()


# Plot 5: distribution of daily poop counts
daily_count_distribution = (
    daily["count"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(12, 5.5))
plt.bar(daily_count_distribution.index, daily_count_distribution.values)
plt.title("Distribution of daily poop counts")
plt.xlabel("Poops per day")
plt.ylabel("Number of days")
plt.xticks(daily_count_distribution.index)
plt.tight_layout()
plt.savefig(FIGURES / "daily_count_distribution.png", dpi=200)
plt.close()

# Plot 6: rolling average daily poops
daily_rolling = daily.copy()
daily_rolling["rolling_average"] = daily_rolling["count"].rolling(
    window=7,
    min_periods=1
).mean()

plt.figure(figsize=(12, 5.5))
plt.plot(
    daily_rolling["date"],
    daily_rolling["rolling_average"],
    marker="o"
)
plt.title("Rolling average daily poops")
plt.xlabel("Date")
plt.ylabel("7-day rolling average")
plt.tight_layout()
plt.savefig(FIGURES / "rolling_average_daily_poops.png", dpi=200)
plt.close()


# Plot 7: poops by month split by good and poor
df["month"] = df["datetime"].dt.to_period("M").astype(str)

month_order = sorted(df["month"].unique())

monthly_quality = (
    df.groupby(["month", value_col])
    .size()
    .unstack(fill_value=0)
    .reindex(month_order, fill_value=0)
)

for v in [0, 1]:
    if v not in monthly_quality.columns:
        monthly_quality[v] = 0

monthly_quality = monthly_quality[[1, 0]]

plt.figure(figsize=(12, 5.5))
plt.bar(monthly_quality.index, monthly_quality[1], label="Good")
plt.bar(
    monthly_quality.index,
    monthly_quality[0],
    bottom=monthly_quality[1],
    label="Poor"
)
plt.title("Poops by month")
plt.xlabel("Month")
plt.ylabel("Number of poops")
plt.xticks(rotation=30, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "poops_by_month.png", dpi=200)
plt.close()

# List of plots shown on the website
# To add a new plot later:
# 1. Save the figure into the figures/ folder
# 2. Add one new entry to this list
plots = [
    {
        "title": "Cumulative poops over time",
        "file": "cumulative_poops.png",
        "alt": "Cumulative poops over time"
    },
    {
        "title": "Daily frequency",
        "file": "daily_frequency.png",
        "alt": "Daily poop frequency"
    },
    {
        "title": "Rolling average daily poops",
        "file": "rolling_average_daily_poops.png",
        "alt": "Rolling average daily poops"
    },
    {
        "title": "Poops by month",
        "file": "poops_by_month.png",
        "alt": "Poops by month"
    },
    {
        "title": "Hour of day",
        "file": "hour_of_day.png",
        "alt": "Poops by hour of day"
    },
    {
        "title": "Poops by weekday",
        "file": "poops_by_weekday.png",
        "alt": "Poops by weekday"
    },
    {
        "title": "Distribution of daily poop counts",
        "file": "daily_count_distribution.png",
        "alt": "Distribution of daily poop counts"
    }
]

def make_anchor(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


nav_links = "\n".join(
    f"""    <a href="#{make_anchor(plot["title"])}">{plot["title"]}</a>"""
    for plot in plots
)


plot_cards = "\n".join(
    f"""
  <div class="plot-card" id="{make_anchor(plot["title"])}">
    <h2>{plot["title"]}</h2>
    <img src="figures/{plot["file"]}" alt="{plot["alt"]}">
  </div>
"""
    for plot in plots
)

# Build HTML
html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>The Poop Observatory</title>
  <style>
    body {{
      max-width: 1000px;
      margin: auto;
      font-family: Arial, sans-serif;
      line-height: 1.5;
      padding: 20px;
      background: #fafafa;
      color: #222;
    }}

    h1, h2 {{
      text-align: center;
    }}

    .subtitle {{
      text-align: center;
      color: #555;
    }}

    html {{
      scroll-behavior: smooth;
    }}

    .page-layout {{
      display: grid;
      grid-template-columns: 220px minmax(0, 1fr);
      gap: 25px;
      align-items: start;
    }}

    .sidebar {{
      position: sticky;
      top: 20px;
      background: white;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 15px;
    }}

    .sidebar h2 {{
      text-align: left;
      font-size: 1rem;
      margin-top: 0;
      margin-bottom: 10px;
      color: #555;
    }}

    .sidebar a {{
      display: block;
      color: #222;
      text-decoration: none;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 0.95rem;
    }}

    .sidebar a:hover {{
      background: #f0f0f0;
    }}

    .content {{
      min-width: 0;
    }}

    @media (max-width: 850px) {{
      .page-layout {{
        grid-template-columns: 1fr;
      }}

      .sidebar {{
        position: static;
      }}
    }}

    .statbox {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 220px));
      gap: 15px;
      margin: 25px auto;
      justify-content: center;
    }}

    .stat {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 15px;
      text-align: center;
    }}

    .stat h3 {{
      margin-bottom: 5px;
      font-size: 1rem;
      color: #555;
    }}

    .stat p {{
      font-size: 1.8rem;
      font-weight: bold;
      margin: 0;
    }}

    img {{
      width: 100%;
      display: block;
      background: white;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 8px;
      box-sizing: border-box;
    }}

    .plots {{
      max-width: 950px;
      margin: 40px auto;
    }}

    .plot-card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 35px;
    }}

    .plot-card h2 {{
      margin-top: 0;
      margin-bottom: 15px;
      text-align: left;
    }}

    footer {{
      text-align: center;
      color: #777;
      margin-top: 40px;
      font-size: 0.9rem;
    }}
  </style>
</head>
<body>

<h1>The Poop Observatory</h1>
<p class="subtitle">Automatically generated from anonymised raw poop data.</p>

<div class="statbox">

  <div class="stat">
    <h3>Total observations</h3>
    <p>{summary["observations"]}</p>
  </div>

  <div class="stat">
    <h3>Days covered</h3>
    <p>{summary["days"]}</p>
  </div>

  <div class="stat">
    <h3>Days with poop</h3>
    <p>{summary["days_with_poop"]}</p>
  </div>

  <div class="stat">
    <h3>Zero-poop days</h3>
    <p>{summary["zero_poop_days"]}</p>
  </div>

  <div class="stat">
    <h3>Average/day</h3>
    <p>{summary["average_per_day"]}</p>
  </div>

  <div class="stat">
    <h3>Median/day</h3>
    <p>{summary["median_per_day"]}</p>
  </div>

  <div class="stat">
    <h3>Maximum/day</h3>
    <p>{summary["max_per_day"]}</p>
  </div>

  <div class="stat">
    <h3>Good poops</h3>
    <p>{summary["good"]}</p>
  </div>

  <div class="stat">
    <h3>Poor poops</h3>
    <p>{summary["poor"]}</p>
  </div>

  <div class="stat">
    <h3>Good poop rate</h3>
    <p>{summary["good_rate"]}%</p>
  </div>

  <div class="stat">
    <h3>Poor poop rate</h3>
    <p>{summary["poor_rate"]}%</p>
  </div>

  <div class="stat">
    <h3>Most common hour</h3>
    <p>{summary["most_common_hour"]}:00</p>
  </div>

</div>

<div class="plots">
{plot_cards}
</div>

<footer>
  Raw timestamps are not displayed on this site.
</footer>

</body>
</html>
"""

Path("index.html").write_text(html, encoding="utf-8")
