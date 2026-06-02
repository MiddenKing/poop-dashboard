from pathlib import Path
import json
import re
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA = Path("data/poop_data.csv")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)


# Load data
df = pd.read_csv(DATA)

time_col = [c for c in df.columns if c.lower() == "time"][0]
value_col = [c for c in df.columns if c.lower() == "value"][0]


# Prepare data
df["datetime"] = pd.to_datetime(df[time_col])
df["date"] = df["datetime"].dt.floor("D")
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.day_name()
df["month"] = df["datetime"].dt.to_period("M")

df["is_poor"] = df[value_col].eq(0)
df["is_good"] = df[value_col].eq(1)

date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

month_range = pd.period_range(
    df["month"].min(),
    df["month"].max(),
    freq="M"
)


# Daily counts
daily = (
    df.groupby("date")
    .size()
    .reindex(date_range, fill_value=0)
    .rename("count")
    .reset_index()
    .rename(columns={"index": "date"})
)


# Summary statistics
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
plt.plot(
    daily_cumulative["date"],
    daily_cumulative["cumulative_count"],
    marker="o"
)
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
daily_rolling["rolling_average"] = (
    daily_rolling["count"]
    .rolling(window=7, min_periods=1)
    .mean()
)

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
monthly_quality = (
    df.groupby(["month", value_col])
    .size()
    .unstack(fill_value=0)
    .reindex(month_range, fill_value=0)
)

for v in [0, 1]:
    if v not in monthly_quality.columns:
        monthly_quality[v] = 0

monthly_quality = monthly_quality[[1, 0]]
monthly_labels = [str(m) for m in monthly_quality.index]
monthly_x = np.arange(len(monthly_labels))

plt.figure(figsize=(12, 5.5))
plt.bar(monthly_x, monthly_quality[1].values, label="Good")
plt.bar(
    monthly_x,
    monthly_quality[0].values,
    bottom=monthly_quality[1].values,
    label="Poor"
)
plt.title("Poops by month")
plt.xlabel("Month")
plt.ylabel("Number of poops")
plt.xticks(monthly_x, monthly_labels, rotation=30, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES / "poops_by_month.png", dpi=200)
plt.close()


# Plot 8: poop calendar
calendar_data = daily.copy()
calendar_data["weekday_num"] = calendar_data["date"].dt.weekday
calendar_data["week"] = calendar_data["date"].dt.isocalendar().week.astype(int)
calendar_data["year"] = calendar_data["date"].dt.isocalendar().year.astype(int)

calendar_data["year_week"] = (
    calendar_data["year"].astype(str)
    + "-"
    + calendar_data["week"].astype(str).str.zfill(2)
)

week_order = calendar_data["year_week"].drop_duplicates().tolist()
week_lookup = {week: i for i, week in enumerate(week_order)}

calendar_grid = np.full((7, len(week_order)), np.nan)

for _, row in calendar_data.iterrows():
    week_index = week_lookup[row["year_week"]]
    weekday_index = int(row["weekday_num"])
    calendar_grid[weekday_index, week_index] = row["count"]

plt.figure(figsize=(14, 4.8))
plt.imshow(calendar_grid, aspect="auto", interpolation="nearest")
plt.title("Poop calendar")
plt.xlabel("Week")
plt.ylabel("Day of week")
plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

week_labels = []
week_positions = []

for i, week in enumerate(week_order):
    if i == 0 or week.endswith("-01") or i % 4 == 0:
        week_labels.append(week)
        week_positions.append(i)

plt.xticks(week_positions, week_labels, rotation=45, ha="right")
cbar = plt.colorbar()
cbar.set_label("Poops per day")
plt.tight_layout()
plt.savefig(FIGURES / "poop_calendar.png", dpi=200)
plt.close()


# Plot 9: top three longest poop streaks
streak_data = daily.copy()
streak_data["has_poop"] = streak_data["count"] > 0

streaks = []
current_start = None
current_length = 0
previous_date = None

for _, row in streak_data.iterrows():
    if row["has_poop"]:
        if current_start is None:
            current_start = row["date"]
            current_length = 1
        else:
            current_length += 1
    else:
        if current_start is not None:
            streaks.append({
                "start": current_start,
                "end": previous_date,
                "length": current_length
            })
            current_start = None
            current_length = 0

    previous_date = row["date"]

if current_start is not None:
    streaks.append({
        "start": current_start,
        "end": previous_date,
        "length": current_length
    })

streak_df = pd.DataFrame(streaks)

plt.figure(figsize=(12, 5.5))

if not streak_df.empty:
    top_streaks = (
        streak_df
        .sort_values("length", ascending=False)
        .head(3)
        .sort_values("length", ascending=True)
    )

    streak_labels = [
        f"{row['start'].strftime('%Y-%m-%d')} to {row['end'].strftime('%Y-%m-%d')}"
        for _, row in top_streaks.iterrows()
    ]

    plt.barh(streak_labels, top_streaks["length"])
    plt.xlabel("Consecutive days with poop")
    plt.ylabel("Streak period")
else:
    plt.text(
        0.5,
        0.5,
        "No poop streaks found",
        ha="center",
        va="center",
        transform=plt.gca().transAxes
    )
    plt.xticks([])
    plt.yticks([])

plt.title("Top three longest poop streaks")
plt.tight_layout()
plt.savefig(FIGURES / "top_three_poop_streaks.png", dpi=200)
plt.close()

# Plot 10: distribution of time between poops
poop_times = df.sort_values("datetime").copy()
poop_times["hours_since_previous"] = (
    poop_times["datetime"]
    .diff()
    .dt.total_seconds()
    / 3600
)

time_between = poop_times["hours_since_previous"].dropna()

plt.figure(figsize=(12, 5.5))
plt.hist(time_between, bins=20)
plt.title("Distribution of time between poops")
plt.xlabel("Hours since previous poop")
plt.ylabel("Number of intervals")
plt.tight_layout()
plt.savefig(FIGURES / "time_between_poops_distribution.png", dpi=200)
plt.close()


# Plot 11: distribution of poop quality
quality_counts = pd.Series({
    "Good": int(df["is_good"].sum()),
    "Poor": int(df["is_poor"].sum())
})

plt.figure(figsize=(12, 5.5))
plt.bar(quality_counts.index, quality_counts.values)
plt.title("Distribution of poop quality")
plt.xlabel("Poop quality")
plt.ylabel("Number of poops")
plt.tight_layout()
plt.savefig(FIGURES / "poop_quality_distribution.png", dpi=200)
plt.close()

# Website plot list
# Add future plots here. The website cards and sidebar index are built from this list.
plots = [
    {
        "section": "Overview",
        "title": "Cumulative poops over time",
        "file": "cumulative_poops.png",
        "alt": "Cumulative poops over time"
    },
    {
    "section": "Daily patterns",
    "title": "Daily frequency",
    "file": "daily_frequency.png",
    "alt": "Daily poop frequency"
},
{
    "section": "Daily patterns",
    "title": "Poop calendar",
    "file": "poop_calendar.png",
    "alt": "Poop calendar"
},
{
    "section": "Daily patterns",
    "title": "Top three longest poop streaks",
    "file": "top_three_poop_streaks.png",
    "alt": "Top three longest poop streaks"
},
{
    "section": "Daily patterns",
    "title": "Rolling average daily poops",
    "file": "rolling_average_daily_poops.png",
    "alt": "Rolling average daily poops"
},
    {
        "section": "Monthly patterns",
        "title": "Poops by month",
        "file": "poops_by_month.png",
        "alt": "Poops by month"
    },
    {
        "section": "Timing patterns",
        "title": "Hour of day",
        "file": "hour_of_day.png",
        "alt": "Poops by hour of day"
    },
    {
        "section": "Timing patterns",
        "title": "Poops by weekday",
        "file": "poops_by_weekday.png",
        "alt": "Poops by weekday"
    },
{
    "section": "Distributions",
    "title": "Distribution of daily poop counts",
    "file": "daily_count_distribution.png",
    "alt": "Distribution of daily poop counts"
},
{
    "section": "Distributions",
    "title": "Distribution of time between poops",
    "file": "time_between_poops_distribution.png",
    "alt": "Distribution of time between poops"
},
{
    "section": "Distributions",
    "title": "Distribution of poop quality",
    "file": "poop_quality_distribution.png",
    "alt": "Distribution of poop quality"
}
]


def make_anchor(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


sections = []

for plot in plots:
    if plot["section"] not in sections:
        sections.append(plot["section"])


nav_links = ""

for section in sections:
    nav_links += f"""
    <div class="nav-section">
      <h3>{section}</h3>
"""

    for plot in plots:
        if plot["section"] == section:
            nav_links += f"""      <a href="#{make_anchor(plot['title'])}">{plot['title']}</a>
"""

    nav_links += """    </div>
"""


plot_cards = "\n".join(
    f"""
  <div class="plot-card" id="{make_anchor(plot['title'])}">
    <h2>{plot['title']}</h2>
    <img src="figures/{plot['file']}" alt="{plot['alt']}">
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
    html {{
      scroll-behavior: smooth;
    }}

    body {{
      max-width: 1250px;
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

    .page-layout {{
      display: grid;
      grid-template-columns: 230px minmax(0, 1fr);
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

    .nav-section {{
      margin-bottom: 18px;
    }}

    .nav-section h3 {{
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #777;
      margin: 12px 0 6px 0;
    }}

    .nav-section:first-of-type h3 {{
      margin-top: 0;
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
      scroll-margin-top: 20px;
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

    @media (max-width: 850px) {{
      .page-layout {{
        grid-template-columns: 1fr;
      }}

      .sidebar {{
        position: static;
      }}
    }}
  </style>
</head>
<body>

<h1>The Poop Observatory</h1>
<p class="subtitle">Automatically generated from anonymised raw poop data.</p>

<div class="page-layout">

<nav class="sidebar">
  <h2>Visualisations</h2>
{nav_links}
</nav>

<main class="content">

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

</main>

</div>

<footer>
  Raw timestamps are not displayed on this site.
</footer>

</body>
</html>
"""

Path("index.html").write_text(html, encoding="utf-8")
