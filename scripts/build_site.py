"""
The Poop Observatory
====================

Builds a self-contained static site (index.html + figures/*.png) for GitHub
Pages from a raw poop log (columns: Time, Value where Value is 1 = good,
0 = poor). No interactivity, no extra dependencies beyond matplotlib.

What this keeps over the original static version:
  * A 24-hour polar "dial" as the signature reading of the daily rhythm.
  * Circular statistics: a regularity score (mean resultant length) plus a
    Rayleigh test for whether timing is genuinely clustered vs. uniform.
  * Weekday tests: chi-square for frequency (correctly weighted by how many
    of each weekday the log actually covers) and a contingency test for
    whether quality depends on the day.
  * Poisson and exponential models overlaid on the daily-count and
    time-between distributions.
  * A single plot registry, a reusable good/poor split helper, and pure
    analytics functions kept separate from plotting.

Run:  python generate_dashboard.py
Needs: pandas, numpy, scipy, matplotlib
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")   # headless: deterministic output, no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

DATA = Path("data/poop_data.csv")
FIGURES = Path("figures")
OUTPUT = Path("index.html")
SUMMARY = Path("summary.json")

# Palette: cool instrument paper, one warm signal colour for the anomaly.
GOOD_COLOR = "#2f6f8f"   # good = deep instrument blue
POOR_COLOR = "#d98a3d"   # poor = warm amber (the reading you notice)
INK = "#16233b"
MUTED = "#5b6b82"
GRID = "#e4e9ef"

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

FIGSIZE = (11, 5.2)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 12,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "axes.axisbelow": True,
})


# ----------------------------------------------------------------------
# Load + validate
# ----------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_col = next(c for c in df.columns if c.lower() == "time")
    value_col = next(c for c in df.columns if c.lower() == "value")

    # Raw CSV may hold non-zero-padded timestamps (2026-6-16 10:23:3),
    # so parse as mixed format rather than trusting one inferred layout.
    df["datetime"] = pd.to_datetime(
        df[time_col].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )

    bad = df[df["datetime"].isna()]
    if not bad.empty:
        print("Unparseable timestamps:")
        print(bad[[time_col, value_col]])
        raise ValueError("Fix timestamp parsing before building the site.")

    df = df.sort_values("datetime").reset_index(drop=True)

    df["value"] = df[value_col].astype(int)
    df["date"] = df["datetime"].dt.floor("D")
    df["hour"] = df["datetime"].dt.hour
    df["hour_decimal"] = df["hour"] + df["datetime"].dt.minute / 60.0
    df["weekday"] = df["datetime"].dt.day_name()
    df["month"] = df["datetime"].dt.to_period("M")
    df["is_good"] = df["value"].eq(1)
    df["is_poor"] = df["value"].eq(0)
    return df


def daily_counts(df: pd.DataFrame, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    return (
        df.groupby("date").size()
        .reindex(date_range, fill_value=0)
        .rename("count").reset_index().rename(columns={"index": "date"})
    )


# ----------------------------------------------------------------------
# Analytics  (pure functions: no plotting, easy to test)
# ----------------------------------------------------------------------

def quality_split(df: pd.DataFrame, key: str, index) -> pd.DataFrame:
    """Counts of good/poor for each level of `key`, aligned to `index`.

    Returns a frame with columns ['good', 'poor']. Replaces the repeated
    unstack / reindex / reorder block that appeared in every stacked chart.
    """
    grid = df.groupby([key, "value"]).size().unstack(fill_value=0)
    for v in (0, 1):
        if v not in grid.columns:
            grid[v] = 0
    grid = grid.reindex(index, fill_value=0)
    return pd.DataFrame({"good": grid[1], "poor": grid[0]}, index=grid.index)


def circular_stats(hours_decimal: np.ndarray) -> dict:
    """Treat clock times as angles and measure how clustered they are.

    R (mean resultant length) is 0 when times are spread evenly around the
    clock and 1 when every poop happens at the exact same time of day, so it
    doubles as a 0-100% "regularity" score. The Rayleigh test asks whether an
    R that large could plausibly come from uniform timing (Zar 1999 approx).
    """
    theta = 2 * np.pi * (np.asarray(hours_decimal) / 24.0)
    n = len(theta)
    if n == 0:
        return {"R": 0.0, "mean_hour": 0.0, "rayleigh_p": 1.0, "n": 0}

    c, s = np.cos(theta).mean(), np.sin(theta).mean()
    R = float(np.hypot(c, s))
    mean_hour = float((np.arctan2(s, c) % (2 * np.pi)) / (2 * np.pi) * 24.0)

    Z = n * R * R
    p = np.exp(-Z) * (
        1
        + (2 * Z - Z ** 2) / (4 * n)
        - (24 * Z - 132 * Z ** 2 + 76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2)
    )
    return {"R": R, "mean_hour": mean_hour,
            "rayleigh_p": float(min(max(p, 0.0), 1.0)), "n": n}


def weekday_frequency_test(df: pd.DataFrame, date_range: pd.DatetimeIndex) -> dict:
    """Chi-square: are poops evenly spread across weekdays?

    Expected counts are weighted by how many Mondays, Tuesdays, ... the log
    actually spans -- not a flat total/7, which would be wrong for a run that
    doesn't cover whole weeks.
    """
    observed = (df["weekday"].value_counts()
                .reindex(WEEKDAY_ORDER, fill_value=0).to_numpy(float))
    weekday_days = (pd.Series(date_range.day_name())
                    .value_counts().reindex(WEEKDAY_ORDER, fill_value=0).to_numpy(float))
    expected = weekday_days / weekday_days.sum() * observed.sum()
    chi2, p = stats.chisquare(observed, f_exp=expected)
    busiest = WEEKDAY_ORDER[int(np.argmax(observed / weekday_days))]
    return {"chi2": float(chi2), "p": float(p), "busiest_weekday": busiest}


def weekday_quality_test(df: pd.DataFrame) -> dict:
    """Contingency test: does poop quality depend on the weekday?"""
    table = (pd.crosstab(df["weekday"], df["value"])
             .reindex(WEEKDAY_ORDER, fill_value=0))
    for v in (0, 1):
        if v not in table.columns:
            table[v] = 0
    chi2, p, dof, _ = stats.chi2_contingency(table[[1, 0]].to_numpy())
    small = int(table.min().min()) < 5   # flag sparse cells
    return {"chi2": float(chi2), "p": float(p), "sparse": small}


def interval_stats(df: pd.DataFrame) -> dict:
    gaps = df["datetime"].diff().dt.total_seconds().div(3600).dropna()
    return {"median_gap": float(gaps.median()),
            "mean_gap": float(gaps.mean()),
            "gaps": gaps}


def find_streaks(daily: pd.DataFrame) -> pd.DataFrame:
    """Runs of consecutive days with at least one poop (vectorised)."""
    has = daily["count"].gt(0)
    run_id = (has != has.shift()).cumsum()
    runs = (daily.assign(has=has, run=run_id)[has]
            .groupby("run")
            .agg(start=("date", "first"), end=("date", "last"), length=("date", "size"))
            .reset_index(drop=True))
    return runs.sort_values("length", ascending=False).reset_index(drop=True)


def build_summary(df, daily, circ, wf_test, wq_test, gaps, lam) -> dict:
    days = len(daily)
    zero_days = int((daily["count"] == 0).sum())
    return {
        "observations": int(len(df)),
        "days": days,
        "days_with_poop": int((daily["count"] > 0).sum()),
        "zero_poop_days": zero_days,
        "average_per_day": round(len(df) / days, 2),
        "median_per_day": float(daily["count"].median()),
        "max_per_day": int(daily["count"].max()),
        "good": int(df["is_good"].sum()),
        "poor": int(df["is_poor"].sum()),
        "good_rate": round(df["is_good"].mean() * 100, 1),
        "poor_rate": round(df["is_poor"].mean() * 100, 1),
        "most_common_hour": int(df["hour"].mode().iloc[0]),
        # New readings
        "regularity_score": round(circ["R"] * 100, 1),
        "mean_clock_time": _fmt_clock(circ["mean_hour"]),
        "rayleigh_p": circ["rayleigh_p"],
        "busiest_weekday": wf_test["busiest_weekday"],
        "weekday_freq_p": wf_test["p"],
        "weekday_quality_p": wq_test["p"],
        "weekday_quality_sparse": wq_test["sparse"],
        "typical_gap_hours": round(float(gaps.median()), 1),
        "expected_zero_day_rate": round(float(stats.poisson.pmf(0, lam)) * 100, 1),
        "observed_zero_day_rate": round(zero_days / days * 100, 1),
    }


def _fmt_clock(hour_float: float) -> str:
    h = int(hour_float) % 24
    m = int(round((hour_float - int(hour_float)) * 60)) % 60
    return f"{h:02d}:{m:02d}"


def _fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ----------------------------------------------------------------------
# Plots  (each draws, saves a PNG, and returns its filename)
# ----------------------------------------------------------------------

def _finish(fig, name: str) -> str:
    fig.tight_layout()
    fig.savefig(FIGURES / name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return name


def _stacked(ax, x, good, poor, width=None):
    kw = {} if width is None else {"width": width}
    ax.bar(x, good, color=GOOD_COLOR, label="Good", **kw)
    ax.bar(x, poor, bottom=good, color=POOR_COLOR, label="Poor", **kw)
    ax.legend(frameon=False)


def fig_dial(df) -> str:
    """Signature reading: poops arranged around a 24-hour clock face."""
    split = quality_split(df, "hour", range(24))
    theta = np.array([2 * np.pi * h / 24 for h in range(24)])
    width = 2 * np.pi / 24
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)          # clockwise, like a real clock
    ax.bar(theta, split["good"], width=width, align="edge",
           color=GOOD_COLOR, label="Good")
    ax.bar(theta, split["poor"], width=width, align="edge",
           bottom=split["good"], color=POOR_COLOR, label="Poor")
    ax.set_xticks([2 * np.pi * h / 24 for h in range(0, 24, 2)])
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticklabels([])
    ax.grid(color=GRID)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False)
    return _finish(fig, "dial.png")


def fig_cumulative(daily) -> str:
    d = daily.copy()
    d["cum"] = d["count"].cumsum()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(d["date"], d["cum"], color=GOOD_COLOR, alpha=0.15)
    ax.plot(d["date"], d["cum"], color=GOOD_COLOR, linewidth=2)
    ax.set_ylabel("Cumulative poops")
    ax.margins(x=0)
    return _finish(fig, "cumulative.png")


def fig_calendar(daily) -> str:
    d = daily.copy()
    d["weekday_num"] = d["date"].dt.weekday
    iso = d["date"].dt.isocalendar()
    d["year_week"] = (iso.year.astype(str) + "-W"
                      + iso.week.astype(int).astype(str).str.zfill(2))
    weeks = d["year_week"].drop_duplicates().tolist()
    week_idx = {w: i for i, w in enumerate(weeks)}

    z = np.full((7, len(weeks)), np.nan)
    for _, r in d.iterrows():
        z[int(r["weekday_num"])][week_idx[r["year_week"]]] = r["count"]

    cmap = LinearSegmentedColormap.from_list("poop", ["#e9f1f6", GOOD_COLOR])
    cmap.set_bad("#f4f7f9")

    fig, ax = plt.subplots(figsize=(13, 4.4))
    im = ax.imshow(np.ma.masked_invalid(z), aspect="auto",
                   interpolation="nearest", cmap=cmap, vmin=0)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ticks = [i for i, _ in enumerate(weeks) if i % 4 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([weeks[i] for i in ticks], rotation=45, ha="right")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Poops per day")
    return _finish(fig, "calendar.png")


def fig_streaks(streaks) -> str:
    top = streaks.head(3).iloc[::-1]
    labels = [f"{r.start:%Y-%m-%d} → {r.end:%Y-%m-%d}" for r in top.itertuples()]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(labels, top["length"], color=GOOD_COLOR)
    ax.set_xlabel("Consecutive days with a poop")
    ax.grid(axis="y", visible=False)
    return _finish(fig, "streaks.png")


def fig_rolling(daily) -> str:
    d = daily.copy()
    d["r7"] = d["count"].rolling(7, min_periods=1).mean()
    d["r30"] = d["count"].rolling(30, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(d["date"], d["r7"], color=GOOD_COLOR, linewidth=2, label="7-day")
    ax.plot(d["date"], d["r30"], color=POOR_COLOR, linewidth=2,
            linestyle=":", label="30-day")
    ax.set_ylabel("Poops per day")
    ax.legend(frameon=False)
    ax.margins(x=0)
    return _finish(fig, "rolling.png")


def fig_weekday(df) -> str:
    split = quality_split(df, "weekday", WEEKDAY_ORDER)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _stacked(ax, split.index, split["good"], split["poor"])
    ax.set_ylabel("Number of poops")
    ax.set_xticks(range(len(split.index)))
    ax.set_xticklabels(split.index, rotation=30, ha="right")
    return _finish(fig, "weekday.png")


def fig_weekday_quality(df) -> str:
    split = quality_split(df, "weekday", WEEKDAY_ORDER)
    total = (split["good"] + split["poor"]).replace(0, np.nan)
    rate = (split["good"] / total * 100)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(split.index, rate, color=GOOD_COLOR)
    ax.set_ylabel("% good")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(split.index)))
    ax.set_xticklabels(split.index, rotation=30, ha="right")
    return _finish(fig, "weekday_quality.png")


def fig_monthly(df, month_range) -> str:
    labels = [str(m) for m in month_range]
    split = quality_split(df, "month", month_range)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x, split["good"].to_numpy(), color=GOOD_COLOR, label="Good")
    ax.bar(x, split["poor"].to_numpy(), bottom=split["good"].to_numpy(),
           color=POOR_COLOR, label="Poor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Number of poops")
    ax.legend(frameon=False)
    return _finish(fig, "monthly.png")


def fig_daily_distribution(daily, lam) -> str:
    counts = daily["count"].value_counts().sort_index()
    ks = np.arange(0, int(daily["count"].max()) + 1)
    expected = stats.poisson.pmf(ks, lam) * len(daily)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(counts.index, counts.values, color=GOOD_COLOR, label="Observed")
    ax.plot(ks, expected, color=POOR_COLOR, marker="o", linewidth=2,
            label=f"Poisson (λ={lam:.2f})")
    ax.set_xlabel("Poops per day")
    ax.set_ylabel("Number of days")
    ax.set_xticks(ks)
    ax.legend(frameon=False)
    return _finish(fig, "daily_distribution.png")


def fig_time_between(gaps) -> str:
    gaps = gaps[gaps < np.percentile(gaps, 99)]   # trim multi-day tail
    mean_gap = gaps.mean()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts, edges, _ = ax.hist(gaps, bins=24, color=GOOD_COLOR, label="Observed")
    centres = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    expected = (np.exp(-centres / mean_gap) / mean_gap) * len(gaps) * width
    ax.plot(centres, expected, color=POOR_COLOR, linewidth=2,
            label=f"Exponential (mean {mean_gap:.1f} h)")
    ax.set_xlabel("Hours since previous poop")
    ax.set_ylabel("Number of intervals")
    ax.legend(frameon=False)
    return _finish(fig, "time_between.png")


def fig_quality_donut(summary) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([summary["good"], summary["poor"]],
           labels=["Good", "Poor"], colors=[GOOD_COLOR, POOR_COLOR],
           autopct="%1.1f%%", startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.set(aspect="equal")
    return _finish(fig, "quality_donut.png")


# ----------------------------------------------------------------------
# HTML assembly
# ----------------------------------------------------------------------

CSS = """
:root{--paper:#eef1f4;--panel:#fff;--ink:#16233b;--muted:#5b6b82;--line:#d9dfe6;}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{max-width:1250px;margin:auto;padding:24px;background:var(--paper);
 color:var(--ink);font-family:Inter,Arial,sans-serif;line-height:1.5;}
.masthead{border-bottom:2px solid var(--ink);padding-bottom:16px;margin-bottom:8px;}
.masthead h1{font-family:'Space Grotesk',sans-serif;font-weight:700;
 letter-spacing:-0.5px;margin:0;font-size:2.4rem;}
.eyebrow{font-family:'IBM Plex Mono',monospace;text-transform:uppercase;
 letter-spacing:0.18em;font-size:0.72rem;color:var(--muted);margin:0 0 6px;}
.subtitle{color:var(--muted);margin:6px 0 0;}
.page-layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:28px;
 align-items:start;margin-top:24px;}
.sidebar{position:sticky;top:20px;background:var(--panel);border:1px solid var(--line);
 border-radius:12px;padding:16px;}
.sidebar h2{font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
 text-transform:uppercase;letter-spacing:0.12em;color:var(--muted);margin:0 0 12px;}
.nav-section{margin-bottom:16px;}
.nav-section h3{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
 text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);margin:12px 0 4px;}
.nav-section:first-of-type h3{margin-top:0;}
.sidebar a{display:block;color:var(--ink);text-decoration:none;padding:6px 9px;
 border-radius:7px;font-size:0.9rem;}
.sidebar a:hover{background:var(--paper);}
.content{min-width:0;}
.readout{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
 gap:12px;margin-bottom:28px;}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:14px 16px;}
.stat h3{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
 text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);
 margin:0 0 6px;font-weight:500;}
.stat p{font-family:'IBM Plex Mono',monospace;font-size:1.7rem;font-weight:600;
 margin:0;line-height:1.1;}
.stat span{font-size:0.78rem;color:var(--muted);font-family:Inter,sans-serif;
 font-weight:400;display:block;margin-top:2px;}
.plot-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:16px 18px;margin-bottom:26px;scroll-margin-top:20px;}
.plot-card h2{font-family:'Space Grotesk',sans-serif;font-size:1.15rem;
 margin:0 0 12px;text-align:left;}
.plot-card img{width:100%;display:block;}
.note{font-size:0.82rem;color:var(--muted);margin:10px 2px 0;}
.methods{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:18px 22px;margin-top:8px;}
.methods h2{font-family:'Space Grotesk',sans-serif;font-size:1.1rem;margin:0 0 8px;}
.methods p{font-size:0.86rem;color:var(--muted);margin:6px 0;}
footer{text-align:center;color:var(--muted);margin-top:36px;font-size:0.85rem;
 font-family:'IBM Plex Mono',monospace;}
@media(max-width:850px){.page-layout{grid-template-columns:1fr;}
 .sidebar{position:static;}}
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Space+Grotesk:wght@500;600;700&'
    'family=IBM+Plex+Mono:wght@400;500;600&'
    'family=Inter:wght@400;500;600&display=swap" rel="stylesheet">'
)


def anchor(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def stat_card(label, value, sub=""):
    sub_html = f"<span>{sub}</span>" if sub else ""
    return f'<div class="stat"><h3>{label}</h3><p>{value}</p>{sub_html}</div>'


def build_html(summary, plots, methods_html):
    sections = []
    for p in plots:
        if p["section"] not in sections:
            sections.append(p["section"])

    nav = []
    for sec in sections:
        links = "".join(
            f'<a href="#{anchor(p["title"])}">{p["title"]}</a>'
            for p in plots if p["section"] == sec
        )
        nav.append(f'<div class="nav-section"><h3>{sec}</h3>{links}</div>')
    nav_html = "".join(nav)

    cards = "".join(
        f'<div class="plot-card" id="{anchor(p["title"])}">'
        f'<h2>{p["title"]}</h2>'
        f'<img src="figures/{p["file"]}" alt="{p["title"]}">'
        + (f'<p class="note">{p["note"]}</p>' if p.get("note") else "")
        + "</div>"
        for p in plots
    )

    readout = "".join([
        stat_card("Observations", summary["observations"]),
        stat_card("Days covered", summary["days"]),
        stat_card("Good rate", f'{summary["good_rate"]}%'),
        stat_card("Regularity", f'{summary["regularity_score"]}%'),
        stat_card("Mean clock time", summary["mean_clock_time"]),
        stat_card("Typical gap", f'{summary["typical_gap_hours"]} h'),
        stat_card("Busiest day", summary["busiest_weekday"]),
        stat_card("Zero-poop days", summary["zero_poop_days"],
                  f'obs {summary["observed_zero_day_rate"]}% / exp {summary["expected_zero_day_rate"]}%'),
    ])

    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>The Poop Observatory</title>"
        + FONTS
        + "<style>" + CSS + "</style></head><body>"
        + "<div class='masthead'>"
          "<p class='eyebrow'>An instrument reading of the daily rhythm</p>"
          "<h1>The Poop Observatory</h1>"
          "<p class='subtitle'>Automatically generated from an anonymised raw log. "
          "<a href='observatory.html' style='color:#2f6f8f;font-weight:600;text-decoration:none'>Observation report &rarr;</a></p></div>"
        + "<div class='page-layout'>"
        + f"<nav class='sidebar'><h2>Readings</h2>{nav_html}</nav>"
        + "<main class='content'>"
        + f"<div class='readout'>{readout}</div>"
        + f"<div class='plots'>{cards}</div>"
        + methods_html
        + "</main></div>"
        + "<footer>Raw timestamps are not displayed on this site.</footer>"
        + "</body></html>"
    )


def build_methods(summary):
    reg = summary["regularity_score"]
    return (
        "<div class='methods'><h2>Methods &amp; readings</h2>"
        f"<p><strong>Regularity ({reg}%).</strong> Clock times are treated as "
        "angles on a 24-hour circle; the score is the mean resultant length, "
        "which is 0% if timing is spread evenly around the clock and 100% if "
        "every poop lands at the same time of day. A Rayleigh test checks that "
        f"this clustering is real rather than chance ({_fmt_p(summary['rayleigh_p'])}).</p>"
        "<p><strong>Weekday frequency.</strong> A chi-square test compares poops "
        "per weekday against how many of each weekday the log actually spans "
        f"({_fmt_p(summary['weekday_freq_p'])}).</p>"
        "<p><strong>Weekday quality.</strong> A contingency test asks whether the "
        f"good/poor split depends on the day ({_fmt_p(summary['weekday_quality_p'])})"
        + (" — treat with care, some day/quality cells are sparse"
           if summary["weekday_quality_sparse"] else "") + ".</p>"
        "<p><strong>Models.</strong> Daily counts are compared with a Poisson "
        "model and gaps between poops with an exponential model; both are overlaid "
        "on the observed data above.</p></div>"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    FIGURES.mkdir(exist_ok=True)
    df = load_data(DATA)

    date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    month_range = pd.period_range(df["month"].min(), df["month"].max(), freq="M")
    daily = daily_counts(df, date_range)

    circ = circular_stats(df["hour_decimal"].to_numpy())
    wf_test = weekday_frequency_test(df, date_range)
    wq_test = weekday_quality_test(df)
    intervals = interval_stats(df)
    lam = daily["count"].mean()
    streaks = find_streaks(daily)

    summary = build_summary(df, daily, circ, wf_test, wq_test, intervals["gaps"], lam)
    SUMMARY.write_text(json.dumps(summary, indent=2))

    print(f"Range {df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d} "
          f"| {summary['observations']} obs over {summary['days']} days")
    print(f"Regularity {summary['regularity_score']}% at ~{summary['mean_clock_time']} "
          f"({_fmt_p(summary['rayleigh_p'])}) | busiest {summary['busiest_weekday']}")

    # Registry: (section, title, figure-file, optional note). Ordered as a
    # narrative that zooms out: day -> week -> whole log -> consistency ->
    # distributions. Add or reorder plots here and the page follows.
    registry = [
        ("Rhythm of the day", "The dial", fig_dial(df),
         "Angle = time of day (00:00 at the top, clockwise); bar length = count. "
         f"Regularity {summary['regularity_score']}% ({_fmt_p(summary['rayleigh_p'])})."),

        ("Rhythm of the week", "Poops by weekday", fig_weekday(df),
         f"Frequency across weekdays: {_fmt_p(summary['weekday_freq_p'])}."),
        ("Rhythm of the week", "Good rate by weekday", fig_weekday_quality(df),
         f"Quality vs weekday: {_fmt_p(summary['weekday_quality_p'])}."),

        ("Over time", "Cumulative poops over time", fig_cumulative(daily), ""),
        ("Over time", "Rolling average", fig_rolling(daily),
         "7-day (solid) and 30-day (dotted) mean poops per day."),
        ("Over time", "Poops by month", fig_monthly(df, month_range), ""),

        ("Consistency", "Poop calendar", fig_calendar(daily),
         "Each cell is a day; darker means more poops, blank means none logged."),
        ("Consistency", "Top 3 longest streaks", fig_streaks(streaks), ""),

        ("Distributions", "Daily count distribution", fig_daily_distribution(daily, lam),
         "Bars = observed; line = Poisson expectation for the same number of days."),
        ("Distributions", "Time between poops", fig_time_between(intervals["gaps"]),
         "Bars = observed gaps; line = exponential model. Long multi-day gaps trimmed."),
        ("Distributions", "Poop quality", fig_quality_donut(summary), ""),
    ]

    plots = [{"section": s, "title": t, "file": f, "note": n}
             for s, t, f, n in registry]

    OUTPUT.write_text(build_html(summary, plots, build_methods(summary)), encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(plots)} charts into {FIGURES}/.")


if __name__ == "__main__":
    main()
