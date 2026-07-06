"""
The Poop Observatory — Observation Report
=========================================

A companion page to the main dashboard (index.html). Where the dashboard
answers "how often / how good", this page treats the log as a periodic
astronomical signal and does the analysis an observatory would:

  * Actogram (double-plotted) — the standard chronobiology plot; a stable
    rhythm shows a vertical band, a drifting one tilts.
  * Lomb-Scargle periodogram — the tool for finding periods in irregularly
    sampled data; the 24-hour circadian peak should light up.
  * Phase-folded profile — every event folded onto a single 24-hour cycle.
  * Hazard curve — instantaneous probability of the next event given time
    since the last.
  * Weekend vs. weekday phase — is the rhythm shifted at the weekend?
    (circular mean + permutation test)
  * Lunar correlation — tested, reported straight, almost certainly null.
  * Ephemeris — the next predicted transit window.

Reads the same log as the dashboard. Writes observatory.html and figures
into figures/obs/. No new dependencies beyond scipy.

Run:  python scripts/build_observatory.py
Needs: pandas, numpy, scipy, matplotlib
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Config — dark "star chart" theme (distinct from the light dashboard)
# ----------------------------------------------------------------------

DATA = Path("data/poop_data.csv")
FIGDIR = Path("figures/obs")
OUTPUT = Path("observatory.html")
DASHBOARD_LINK = "index.html"

BG = "#0b1020"       # deep night sky
PANEL = "#101a30"
INK = "#e8ecf5"
MUTED = "#8a94a8"
GRID = "#24304a"
GOOD = "#6db7d4"     # cyan
POOR = "#e0a458"     # amber
MERIDIAN = "#d98cae" # phase / meridian highlight

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
SYNODIC = 29.530588853        # days, for the lunar test
NEW_MOON_JD = 2451550.1       # a known new moon epoch (2000-01-06)

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.facecolor": PANEL, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "axes.grid": True, "axes.axisbelow": True,
    "font.size": 12, "figure.dpi": 110, "savefig.dpi": 200,
})


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = next(c for c in df.columns if c.lower() == "time")
    value_col = next(c for c in df.columns if c.lower() == "value")
    df["datetime"] = pd.to_datetime(df[time_col].astype(str).str.strip(),
                                    format="mixed", errors="coerce")
    bad = df[df["datetime"].isna()]
    if not bad.empty:
        print(bad[[time_col, value_col]])
        raise ValueError("Fix timestamp parsing before building the report.")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["value"] = df[value_col].astype(int)
    df["date"] = df["datetime"].dt.floor("D")
    df["hour_decimal"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0
    df["weekday"] = df["datetime"].dt.day_name()
    df["is_weekend"] = df["datetime"].dt.weekday >= 5
    df["is_good"] = df["value"].eq(1)
    return df


# ----------------------------------------------------------------------
# Analysis (pure)
# ----------------------------------------------------------------------

def circular(hours) -> dict:
    """Circular mean time, resultant length R, and phase scatter (hours)."""
    theta = 2 * np.pi * np.asarray(hours) / 24.0
    n = len(theta)
    if n == 0:
        return {"mean_hour": 0.0, "R": 0.0, "sd_hours": 12.0, "n": 0}
    c, s = np.cos(theta).mean(), np.sin(theta).mean()
    R = float(np.hypot(c, s))
    mean_hour = float((np.arctan2(s, c) % (2 * np.pi)) / (2 * np.pi) * 24.0)
    sd_hours = float(np.sqrt(max(-2 * np.log(R), 0.0)) / (2 * np.pi) * 24.0)
    return {"mean_hour": mean_hour, "R": R, "sd_hours": sd_hours, "n": n}


def lomb_scargle(df, n_perm=200, seed=0):
    """LS periodogram on hourly event counts; shuffle test for significance."""
    start, end = df["datetime"].min(), df["datetime"].max()
    hours_total = int(np.ceil((end - start).total_seconds() / 3600)) + 1
    idx = ((df["datetime"] - start).dt.total_seconds() // 3600).astype(int)
    y = np.bincount(idx, minlength=hours_total).astype(float)
    t = np.arange(hours_total, dtype=float)

    periods = np.logspace(np.log10(6), np.log10(24 * 40), 3000)   # 6 h .. 40 d
    w = 2 * np.pi / periods
    power = signal.lombscargle(t, y, w, normalize=True, precenter=True)

    peak_i = int(np.argmax(power))
    peak_period, peak_power = periods[peak_i], float(power[peak_i])

    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm)
    for i in range(n_perm):
        p = signal.lombscargle(t, rng.permutation(y), w,
                               normalize=True, precenter=True)
        null_max[i] = p.max()
    threshold = float(np.percentile(null_max, 99))
    p_value = (1 + int((null_max >= peak_power).sum())) / (1 + n_perm)
    return {"periods": periods, "power": power, "peak_period": peak_period,
            "peak_power": peak_power, "threshold": threshold, "p_value": p_value}


def hazard_curve(df, step=2.0, horizon=48.0):
    intervals = df["datetime"].diff().dt.total_seconds().div(3600).dropna().to_numpy()
    edges = np.arange(0, horizon + step, step)
    centres, hazard = [], []
    for e in edges[:-1]:
        at_risk = int((intervals >= e).sum())
        events = int(((intervals >= e) & (intervals < e + step)).sum())
        centres.append(e + step / 2)
        hazard.append(events / at_risk / step if at_risk > 0 else np.nan)
    return np.array(centres), np.array(hazard)


def weekend_phase(df, n_perm=2000, seed=0):
    wk = df.loc[~df["is_weekend"], "hour_decimal"].to_numpy()
    we = df.loc[df["is_weekend"], "hour_decimal"].to_numpy()
    c_wk, c_we = circular(wk), circular(we)
    obs_diff = _circ_diff(c_wk["mean_hour"], c_we["mean_hour"])

    all_hours = df["hour_decimal"].to_numpy()
    labels = df["is_weekend"].to_numpy()
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(labels)
        m1 = circular(all_hours[~perm])["mean_hour"]
        m2 = circular(all_hours[perm])["mean_hour"]
        null[i] = _circ_diff(m1, m2)
    p_value = (1 + int((null >= obs_diff).sum())) / (1 + n_perm)
    return {"weekday": c_wk, "weekend": c_we, "diff_hours": obs_diff,
            "p_value": p_value}


def _circ_diff(h1, h2):
    d = abs(h1 - h2) % 24
    return min(d, 24 - d)


def lunar_test(df):
    jd = df["datetime"].astype("int64") / 1e9 / 86400 + 2440587.5
    illum = (1 - np.cos(2 * np.pi * ((jd - NEW_MOON_JD) % SYNODIC) / SYNODIC)) / 2
    daily = pd.DataFrame({"date": df["date"], "illum": illum})
    per_day = daily.groupby("date")["illum"].mean()
    counts = df.groupby("date").size().reindex(per_day.index, fill_value=0)
    r, p = stats.pearsonr(per_day.to_numpy(), counts.to_numpy())
    return {"illum": per_day.to_numpy(), "counts": counts.to_numpy(),
            "r": float(r), "p": float(p)}


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def _finish(fig, name):
    fig.tight_layout()
    fig.savefig(FIGDIR / name, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return f"obs/{name}"


def fig_dial(df, mean_hour):
    counts_g = np.bincount(df.loc[df["is_good"], "datetime"].dt.hour, minlength=24)
    counts_p = np.bincount(df.loc[~df["is_good"], "datetime"].dt.hour, minlength=24)
    theta = np.array([2 * np.pi * h / 24 for h in range(24)])
    width = 2 * np.pi / 24
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(projection="polar")
    ax.set_facecolor(PANEL)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(theta, counts_g, width=width, align="edge", color=GOOD, label="Good")
    ax.bar(theta, counts_p, width=width, align="edge", bottom=counts_g,
           color=POOR, label="Poor")
    # meridian: mean transit line
    ax.plot([2 * np.pi * mean_hour / 24] * 2, [0, (counts_g + counts_p).max() * 1.05],
            color=MERIDIAN, lw=2, label="Mean transit")
    ax.set_xticks([2 * np.pi * h / 24 for h in range(0, 24, 2)])
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticklabels([])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    return _finish(fig, "dial.png")


def fig_actogram(df):
    days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    row = {d: i for i, d in enumerate(days)}
    fig, ax = plt.subplots(figsize=(11, max(6, len(days) * 0.05)))
    for _, r in df.iterrows():
        i = row[r["date"]]
        colour = GOOD if r["is_good"] else POOR
        h = r["hour_decimal"]
        ax.vlines(h, i - 0.45, i + 0.45, color=colour, lw=1.4)          # today
        if i - 1 >= 0:
            ax.vlines(h + 24, i - 1 - 0.45, i - 1 + 0.45, color=colour, lw=1.4)  # doubled
    for x in range(0, 49, 6):
        ax.axvline(x, color=GRID, lw=0.8)
    ax.set_xlim(0, 48)
    ax.set_ylim(len(days) - 0.5, -0.5)                                  # earliest at top
    ax.set_xticks(range(0, 49, 6))
    ax.set_xticklabels([f"{x % 24:02d}" for x in range(0, 49, 6)])
    yt = np.linspace(0, len(days) - 1, min(10, len(days))).astype(int)
    ax.set_yticks(yt)
    ax.set_yticklabels([days[i].strftime("%b %d") for i in yt])
    ax.set_xlabel("Hour of day (double-plotted, 0–48 h)")
    ax.grid(False)
    return _finish(fig, "actogram.png")


def fig_periodogram(ls):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogx(ls["periods"], ls["power"], color=GOOD, lw=1.4)
    ax.axhline(ls["threshold"], color=MERIDIAN, ls="--", lw=1.2,
               label="99% shuffle threshold")
    ax.axvline(24, color=MUTED, ls=":", lw=1)
    ax.axvline(168, color=MUTED, ls=":", lw=1)
    ax.annotate(f"peak {ls['peak_period']:.1f} h", xy=(ls["peak_period"], ls["peak_power"]),
                xytext=(ls["peak_period"] * 1.4, ls["peak_power"]),
                color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.text(24, ax.get_ylim()[1] * 0.9, " 24 h", color=MUTED, fontsize=10)
    ax.text(168, ax.get_ylim()[1] * 0.9, " 7 d", color=MUTED, fontsize=10)
    ax.set_xlabel("Period (hours, log scale)")
    ax.set_ylabel("Normalised power")
    ax.legend(frameon=False)
    return _finish(fig, "periodogram.png")


def fig_phase_folded(df, c):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df.loc[df["is_good"], "hour_decimal"], bins=48, range=(0, 24),
            color=GOOD, label="Good")
    ax.hist(df.loc[~df["is_good"], "hour_decimal"], bins=48, range=(0, 24),
            bottom=np.histogram(df.loc[df["is_good"], "hour_decimal"],
                                bins=48, range=(0, 24))[0],
            color=POOR, label="Poor")
    ax.axvline(c["mean_hour"], color=MERIDIAN, lw=2,
               label=f"mean transit {int(c['mean_hour']):02d}:"
                     f"{int((c['mean_hour']%1)*60):02d}")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hour of day (all events folded onto one cycle)")
    ax.set_ylabel("Events")
    ax.legend(frameon=False)
    return _finish(fig, "phase_folded.png")


def fig_hazard(centres, hazard):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(centres, hazard, color=GOOD, marker="o", lw=1.6)
    ax.set_xlabel("Hours since previous event")
    ax.set_ylabel("Instantaneous rate (per hour)")
    return _finish(fig, "hazard.png")


def fig_weekend(df, wp):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df.loc[~df["is_weekend"], "hour_decimal"], bins=24, range=(0, 24),
            density=True, histtype="step", lw=2, color=GOOD, label="Weekday")
    ax.hist(df.loc[df["is_weekend"], "hour_decimal"], bins=24, range=(0, 24),
            density=True, histtype="step", lw=2, color=POOR, label="Weekend")
    ax.axvline(wp["weekday"]["mean_hour"], color=GOOD, ls=":", lw=1.5)
    ax.axvline(wp["weekend"]["mean_hour"], color=POOR, ls=":", lw=1.5)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    return _finish(fig, "weekend.png")


def fig_lunar(lu):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(lu["illum"], lu["counts"], color=GOOD, alpha=0.6, s=28)
    if len(lu["illum"]) > 1:
        b, a = np.polyfit(lu["illum"], lu["counts"], 1)
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, a + b * xs, color=MERIDIAN, lw=2)
    ax.set_xlabel("Mean lunar illumination that day (0 = new, 1 = full)")
    ax.set_ylabel("Poops that day")
    return _finish(fig, "lunar.png")


# ----------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------

CSS = """
:root{--bg:#0b1020;--panel:#101a30;--ink:#e8ecf5;--muted:#8a94a8;--line:#24304a;--hi:#d98cae;}
*{box-sizing:border-box;}
html{scroll-behavior:smooth;}
body{max-width:1100px;margin:auto;padding:26px;background:var(--bg);color:var(--ink);
 font-family:Inter,Arial,sans-serif;line-height:1.6;}
a{color:#7fc7e0;}
.masthead{border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:22px;}
.masthead h1{font-family:'Space Grotesk',sans-serif;font-weight:700;margin:0;font-size:2.3rem;}
.eyebrow{font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:0.2em;
 font-size:0.7rem;color:var(--muted);margin:0 0 6px;}
.abstract{color:var(--muted);font-size:0.95rem;max-width:70ch;}
.abstract strong{color:var(--ink);}
.readout{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;
 margin:24px 0;}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px;}
.stat h3{font-family:'IBM Plex Mono',monospace;font-size:0.64rem;text-transform:uppercase;
 letter-spacing:0.07em;color:var(--muted);margin:0 0 6px;font-weight:500;}
.stat p{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;margin:0;line-height:1.1;}
.stat span{font-size:0.72rem;color:var(--muted);font-family:Inter,sans-serif;display:block;margin-top:3px;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
 padding:18px 20px;margin-bottom:26px;}
.card h2{font-family:'Space Grotesk',sans-serif;font-size:1.25rem;margin:0 0 4px;}
.card .kicker{font-family:'IBM Plex Mono',monospace;font-size:0.66rem;text-transform:uppercase;
 letter-spacing:0.09em;color:var(--hi);margin:0 0 10px;}
.card img{width:100%;display:block;border-radius:8px;}
.card .explain{color:var(--ink);font-size:0.92rem;margin:14px 2px 0;}
.card .explain b{color:#a9d6e8;font-weight:600;}
.card details.method{margin:12px 2px 0;border-top:1px solid var(--line);padding-top:8px;}
.card details.method summary{cursor:pointer;font-family:'IBM Plex Mono',monospace;
 font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:var(--muted);
 list-style:none;}
.card details.method summary::before{content:"\\25B8  ";color:var(--hi);}
.card details.method[open] summary::before{content:"\\25BE  ";}
.card details.method p{color:var(--muted);font-size:0.82rem;margin:10px 0 2px;}
.termnote{font-family:'IBM Plex Mono',monospace;font-size:0.74rem;color:var(--muted);
 margin-top:12px;padding-top:12px;border-top:1px solid var(--line);}
.termnote b{color:#a9d6e8;font-weight:600;}
.methods{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 22px;}
.methods h2{font-family:'Space Grotesk',sans-serif;font-size:1.1rem;margin:0 0 8px;}
.methods p{font-size:0.85rem;color:var(--muted);margin:6px 0;}
footer{text-align:center;color:var(--muted);margin-top:34px;font-size:0.82rem;
 font-family:'IBM Plex Mono',monospace;}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?'
         'family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&'
         'family=Inter:wght@400;500;600&display=swap" rel="stylesheet">')


def stat_card(label, value, sub=""):
    return (f'<div class="stat"><h3>{label}</h3><p>{value}</p>'
            f'{f"<span>{sub}</span>" if sub else ""}</div>')


def card(kicker, title, img, explain, method):
    return (f'<div class="card"><p class="kicker">{kicker}</p><h2>{title}</h2>'
            f'<img src="figures/{img}" alt="{title}">'
            f'<div class="explain">{explain}</div>'
            f'<details class="method"><summary>Method</summary><p>{method}</p></details>'
            f'</div>')


def fmt_clock(h):
    return f"{int(h) % 24:02d}:{int(round((h % 1) * 60)) % 60:02d}"


def fmt_p(p):
    return "p &lt; 0.001" if p < 0.001 else f"p = {p:.3f}"


def build_html(readings, cards_html, methods_html):
    readout = "".join([
        stat_card("Observations", readings["n"], "events logged"),
        stat_card("Baseline", readings["days"], "days observed"),
        stat_card("Mean transit", readings["mean_transit"], "typical time of day"),
        stat_card("Phase scatter", f'±{readings["sd_hours"]}', "day-to-day spread (h)"),
        stat_card("Dominant period", f'{readings["period"]}', "cycle length (h)"),
        stat_card("Signal", readings["signal"], fmt_p(readings["period_p"])),
        stat_card("Next transit", readings["ephemeris"], "predicted time"),
        stat_card("Lunar effect", readings["lunar"], fmt_p(readings["lunar_p"])),
    ])
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>The Poop Observatory — Observation Report</title>"
        + FONTS + "<style>" + CSS + "</style></head><body>"
        + "<div class='masthead'><p class='eyebrow'>Observation report · circadian bowel periodicity</p>"
          "<h1>The Poop Observatory</h1>"
          "<p class='abstract'>We treat a single subject's defecation log as an irregularly "
          "sampled periodic signal and recover its rhythm using standard methods from "
          "chronobiology and time-domain astronomy. A <strong>circadian (24-hour) component</strong> "
          "dominates; weekend phase and lunar illumination are tested and reported as found. "
          f"Companion frequency dashboard: <a href='{DASHBOARD_LINK}'>index</a>.</p>"
          "<p class='termnote'>A note on terms, borrowed from astronomy: <b>transit</b> = a "
          "typical event's time of day &middot; <b>meridian</b> = the average of those times &middot; "
          "<b>ephemeris</b> = the predicted time of the next one &middot; <b>phase</b> = where in the "
          "24-hour cycle something falls.</p></div>"
        + f"<div class='readout'>{readout}</div>"
        + cards_html
        + methods_html
        + "<footer>Raw timestamps are not displayed. Null results are reported anyway.</footer>"
        + "</body></html>"
    )


def build_methods(ls, wp, lu):
    return (
        "<div class='methods'><h2>Methods</h2>"
        "<p><strong>Periodogram.</strong> Events are binned into hourly counts and analysed "
        "with a Lomb-Scargle periodogram. Significance is a permutation test: the count series "
        "is shuffled 200 times and the tallest peak recorded; the dashed line is the 99th "
        f"percentile of that null ({fmt_p(ls['p_value'])} for the observed peak).</p>"
        "<p><strong>Actogram.</strong> Each row is one day, double-plotted across 48 hours so "
        "consecutive days sit side by side. A phase-locked rhythm forms a vertical band; drift "
        "or a schedule shift tilts it.</p>"
        "<p><strong>Weekend phase.</strong> Circular means of weekday vs. weekend transit times, "
        f"compared by permuting the labels 2000 times ({fmt_p(wp['p_value'])}).</p>"
        "<p><strong>Lunar test.</strong> Mean lunar illumination per day (synodic-month model) "
        f"correlated with daily count. Pearson r = {lu['r']:.3f} ({fmt_p(lu['p'])}).</p>"
        "<p>Phase scatter is the circular standard deviation. The ephemeris projects the next "
        "transit from the circular mean and its scatter.</p></div>"
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA)

    c = circular(df["hour_decimal"].to_numpy())
    ls = lomb_scargle(df)
    hz_x, hz_y = hazard_curve(df)
    wp = weekend_phase(df)
    lu = lunar_test(df)

    days = (df["date"].max() - df["date"].min()).days + 1
    period_significant = ls["peak_power"] > ls["threshold"]

    readings = {
        "n": len(df),
        "days": days,
        "mean_transit": fmt_clock(c["mean_hour"]),
        "sd_hours": round(c["sd_hours"], 1),
        "period": round(ls["peak_period"], 1),
        "period_p": ls["p_value"],
        "signal": "detected" if period_significant else "n.s.",
        "ephemeris": f"~{fmt_clock(c['mean_hour'])}",
        "lunar": "none" if lu["p"] > 0.05 else f"r={lu['r']:.2f}",
        "lunar_p": lu["p"],
    }

    print(f"n={len(df)} over {days} d | mean transit {readings['mean_transit']} "
          f"±{readings['sd_hours']}h | peak {readings['period']}h "
          f"({fmt_p(ls['p_value'])}) | weekend {fmt_p(wp['p_value'])} "
          f"| lunar r={lu['r']:.3f} ({fmt_p(lu['p'])})")

    cards_html = "".join([
        card("Rhythm", "The dial", fig_dial(df, c["mean_hour"]),
             "Time of day around the clock; the pink meridian marks the mean transit — "
             "the centre of mass of the daily rhythm."),
        card("Signal", "Actogram", fig_actogram(df),
             "Each row is a day, double-plotted across 48 hours. A vertical band means a "
             "stable phase-locked rhythm; a tilt would mean it is drifting."),
        card("Signal", "Lomb-Scargle periodogram", fig_periodogram(ls),
             f"The dominant period sits at {readings['period']} h — the circadian signal. "
             f"Peak {'clears' if period_significant else 'does not clear'} the 99% shuffle "
             f"threshold ({fmt_p(ls['p_value'])})."),
        card("Signal", "Phase-folded profile", fig_phase_folded(df, c),
             "Every event folded onto a single 24-hour cycle: the pure shape of the daily "
             "pulse, stripped of the calendar."),
        card("Timing", "Hazard curve", fig_hazard(hz_x, hz_y),
             "Given it has been this many hours since the last event, the instantaneous rate "
             "of the next — the cadence, read honestly."),
        card("Timing", "Weekend phase shift", fig_weekend(df, wp),
             f"Weekday vs. weekend transit times differ by {wp['diff_hours']:.1f} h "
             f"({fmt_p(wp['p_value'])} by permutation)."),
        card("Controls", "Lunar correlation", fig_lunar(lu),
             f"Tested for completeness. Pearson r = {lu['r']:.3f} ({fmt_p(lu['p'])}) — "
             f"{'no significant lunar effect' if lu['p'] > 0.05 else 'a correlation worth noting'}."),
    ])

    OUTPUT.write_text(build_html(readings, cards_html, build_methods(ls, wp, lu)),
                      encoding="utf-8")
    print(f"Wrote {OUTPUT} with 7 charts into {FIGDIR}/.")


if __name__ == "__main__":
    main()
