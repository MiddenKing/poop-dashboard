from pathlib import Path
import json
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

# Plot 1: daily frequency
plt.figure(figsize=(9, 4.5))
plt.bar(daily["date"], daily["count"], width=0.9)
plt.title("Daily poop frequency")
plt.xlabel("Date")
plt.ylabel("Poops per day")
plt.tight_layout()
plt.savefig(FIGURES / "daily_frequency.png", dpi=200)
plt.close()

# Plot 2: hour of day
hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)

plt.figure(figsize=(9, 4.5))
plt.bar(hourly.index, hourly.values)
plt.title("Poops by hour of day")
plt.xlabel("Hour of day")
plt.ylabel("Number of poops")
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.savefig(FIGURES / "hour_of_day.png", dpi=200)
plt.close()

# Plot 3: good vs poor by hour
quality_hour = df.groupby(["hour", value_col]).size().unstack(fill_value=0)
for v in [0, 1]:
    if v not in quality_hour.columns:
        quality_hour[v] = 0

quality_hour = quality_hour.reindex(range(24), fill_value=0)

plt.figure(figsize=(9, 4.5))
plt.bar(quality_hour.index, quality_hour[1], label="Good")
plt.bar(quality_hour.index, quality_hour[0], bottom=quality_hour[1], label="Poor")
plt.title("Good vs poor poops by hour")
plt.xlabel("Hour of day")
plt.ylabel("Number of poops")
plt.xticks(range(0, 24, 2))
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "good_vs_poor_by_hour.png", dpi=200)
plt.close()

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
      max-width: 100%;
      display: block;
      margin: 20px auto 45px auto;
      background: white;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 8px;
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
    <h3>Average/day</h3>
    <p>{summary["average_per_day"]}</p>
  </div>

  <div class="stat">
    <h3>Median/day</h3>
    <p>{summary["median_per_day"]}</p>
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
    <h3>Maximum/day</h3>
    <p>{summary["max_per_day"]}</p>
  </div>

  <div class="stat">
    <h3>Most common hour</h3>
    <p>{summary["most_common_hour"]}:00</p>
  </div>
</div>

<h2>Daily frequency</h2>
<img src="figures/daily_frequency.png" alt="Daily poop frequency">

<h2>Hour of day</h2>
<img src="figures/hour_of_day.png" alt="Poops by hour of day">

<h2>Good vs poor by hour</h2>
<img src="figures/good_vs_poor_by_hour.png" alt="Good vs poor poops by hour">

<footer>
  Raw timestamps are not displayed on this site.
</footer>

</body>
</html>
"""

Path("index.html").write_text(html, encoding="utf-8")
