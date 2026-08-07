"""
The Poop Observatory — Chronobiology Report
============================================

A companion to the main dashboard.

The dashboard answers:
    "How much? How often? How good?"

The observatory answers:
    "What rhythm does the system follow?"

Analyses:
    - Actogram: daily rhythm stability over time
    - Circular phase analysis: average event time + concentration
    - Phase density: shape of the 24-hour rhythm
    - Interval distribution: cadence between events
    - Hazard curve: probability of next event after elapsed time
    - Weekend shift: social schedule effects
    - Phase drift: whether timing changes over months
    - Anomaly detection: unusually early/late events

Input:
    data/poop_data.csv

Output:
    observatory.html
    figures/obs/*.png

Dependencies:
    pandas
    numpy
    scipy
    matplotlib

Run:
    python scripts/build_observatory.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats
from scipy.ndimage import gaussian_filter1d

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# Configuration
# ======================================================================

DATA = Path("data/poop_data.csv")
FIGDIR = Path("figures/obs")
OUTPUT = Path("observatory.html")

DASHBOARD_LINK = "index.html"


# Dark observatory theme

BG = "#0b1020"
PANEL = "#101a30"
INK = "#e8ecf5"
MUTED = "#8a94a8"
GRID = "#24304a"

GOOD = "#6db7d4"
POOR = "#e0a458"
ACCENT = "#d98cae"


plt.rcParams.update({
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "axes.grid": True,
    "font.size": 12,
    "figure.dpi": 110,
    "savefig.dpi": 200,
})


# ======================================================================
# Loading
# ======================================================================


def load_data(path):

    df = pd.read_csv(path)

    time_col = next(
        c for c in df.columns
        if c.lower() == "time"
    )

    value_col = next(
        c for c in df.columns
        if c.lower() == "value"
    )

    df["datetime"] = pd.to_datetime(
        df[time_col].astype(str).str.strip(),
        format="mixed",
        errors="coerce"
    )

    if df["datetime"].isna().any():
        raise ValueError(
            "Some timestamps could not be parsed."
        )

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    df["value"] = df[value_col].astype(int)

    df["date"] = (
        df["datetime"]
        .dt.floor("D")
    )

    df["hour"] = (
        df["datetime"].dt.hour
        +
        df["datetime"].dt.minute / 60
    )

    df["weekday"] = (
        df["datetime"]
        .dt.day_name()
    )

    df["is_weekend"] = (
        df["datetime"]
        .dt.weekday >= 5
    )

    df["is_good"] = (
        df["value"] == 1
    )

    return df



# ======================================================================
# Circular statistics
# ======================================================================


def circular_summary(hours):

    hours = np.asarray(hours)

    if len(hours) == 0:
        return {
            "mean": np.nan,
            "R": 0,
            "sd": np.nan
        }

    theta = (
        hours / 24
        *
        2*np.pi
    )

    c = np.mean(np.cos(theta))
    s = np.mean(np.sin(theta))

    R = np.sqrt(c*c + s*s)

    mean = (
        np.arctan2(s,c)
        %
        (2*np.pi)
    )

    mean_hour = (
        mean /
        (2*np.pi)
        *
        24
    )

    sd = (
        np.sqrt(
            max(
                -2*np.log(R),
                0
            )
        )
        /
        (2*np.pi)
        *
        24
    )

    return {
        "mean": float(mean_hour),
        "R": float(R),
        "sd": float(sd)
    }



# ======================================================================
# Phase density
# ======================================================================


def phase_density(hours):

    """
    Estimate the probability density
    across a 24-hour cycle.

    Circular smoothing avoids the
    midnight discontinuity.
    """

    hours = np.asarray(hours)

    bins = np.linspace(
        0,
        24,
        241
    )

    hist, edges = np.histogram(
        hours,
        bins=bins,
        density=True
    )

    # wrap around
    extended = np.concatenate(
        [
            hist,
            hist,
            hist
        ]
    )

    smooth = gaussian_filter1d(
        extended,
        sigma=5
    )

    centre = len(hist)

    smooth = smooth[
        centre:centre*2
    ]

    x = (
        edges[:-1]
        +
        edges[1:]
    ) / 2

    return x, smooth



# ======================================================================
# Interval analysis
# ======================================================================


def interval_distribution(df):

    intervals = (
        df["datetime"]
        .diff()
        .dt.total_seconds()
        /
        3600
    )

    return (
        intervals
        .dropna()
        .to_numpy()
    )



def hazard_curve(
        df,
        step=2,
        horizon=72
):

    intervals = interval_distribution(df)

    edges = np.arange(
        0,
        horizon+step,
        step
    )

    centres = []
    hazard = []

    for start in edges[:-1]:

        at_risk = (
            intervals >= start
        ).sum()

        events = (
            (intervals >= start)
            &
            (intervals < start+step)
        ).sum()

        centres.append(
            start + step/2
        )

        if at_risk:
            hazard.append(
                events /
                at_risk /
                step
            )
        else:
            hazard.append(
                np.nan
            )

    return (
        np.array(centres),
        np.array(hazard)
    )



# ======================================================================
# Weekend effect
# ======================================================================


def circular_difference(a,b):

    d = abs(a-b)%24

    return min(
        d,
        24-d
    )



def weekend_test(
        df,
        n_perm=2000,
        seed=0
):

    weekday = circular_summary(
        df.loc[
            ~df.is_weekend,
            "hour"
        ]
    )

    weekend = circular_summary(
        df.loc[
            df.is_weekend,
            "hour"
        ]
    )


    observed = circular_difference(
        weekday["mean"],
        weekend["mean"]
    )


    rng = np.random.default_rng(seed)

    labels = (
        df.is_weekend
        .to_numpy()
    )

    hours = (
        df.hour
        .to_numpy()
    )


    null = []

    for _ in range(n_perm):

        shuffled = (
            rng.permutation(labels)
        )

        a = circular_summary(
            hours[~shuffled]
        )["mean"]

        b = circular_summary(
            hours[shuffled]
        )["mean"]


        null.append(
            circular_difference(a,b)
        )


    p = (
        1 +
        np.sum(
            np.array(null)
            >= observed
        )
    ) / (
        1+n_perm
    )


    return {
        "weekday": weekday,
        "weekend": weekend,
        "difference": observed,
        "p": p
    }



# ======================================================================
# Phase drift
# ======================================================================


def monthly_phase(df):

    tmp = []

    for month, group in df.groupby(
        df.datetime.dt.to_period("M")
    ):

        c = circular_summary(
            group.hour
        )

        tmp.append(
            {
                "month": str(month),
                "mean": c["mean"],
                "n": len(group)
            }
        )

    return pd.DataFrame(tmp)



# ======================================================================
# Anomaly detection
# ======================================================================


def detect_anomalies(df):

    intervals = interval_distribution(df)

    z = stats.zscore(
        intervals
    )


    anomalies = np.where(
        abs(z) > 2.5
    )[0]


    return {
        "count": len(anomalies),
        "intervals": intervals[anomalies]
    }
 # ======================================================================
# Figures
# ======================================================================


def savefig(fig, name):

    FIGDIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.tight_layout()

    fig.savefig(
        FIGDIR / name,
        bbox_inches="tight",
        facecolor=BG
    )

    plt.close(fig)

    return f"obs/{name}"



# ----------------------------------------------------------------------
# Actogram
# ----------------------------------------------------------------------


def plot_actogram(df):

    days = pd.date_range(
        df.date.min(),
        df.date.max(),
        freq="D"
    )

    lookup = {
        d:i
        for i,d in enumerate(days)
    }


    fig, ax = plt.subplots(
        figsize=(11, max(6,len(days)*0.05))
    )


    for _, row in df.iterrows():

        y = lookup[row.date]

        colour = (
            GOOD
            if row.is_good
            else POOR
        )

        ax.vlines(
            row.hour,
            y-0.4,
            y+0.4,
            color=colour,
            lw=1.5
        )


    ax.set_xlim(
        0,
        24
    )

    ax.set_ylim(
        len(days)-0.5,
        -0.5
    )

    ax.set_xlabel(
        "Hour of day"
    )

    ax.set_ylabel(
        "Date"
    )


    ticks = np.linspace(
        0,
        len(days)-1,
        min(10,len(days))
    ).astype(int)


    ax.set_yticks(ticks)

    ax.set_yticklabels(
        [
            days[i].strftime("%b %d")
            for i in ticks
        ]
    )


    return savefig(
        fig,
        "actogram.png"
    )



# ----------------------------------------------------------------------
# Phase density
# ----------------------------------------------------------------------


def plot_phase_density(
        df,
        density
):

    x,y = density


    fig, ax = plt.subplots(
        figsize=(11,5)
    )


    ax.fill_between(
        x,
        y,
        alpha=0.35
    )


    ax.plot(
        x,
        y,
        lw=2
    )


    mean = circular_summary(
        df.hour
    )["mean"]


    ax.axvline(
        mean,
        color=ACCENT,
        lw=2,
        label=f"Mean {mean:.1f} h"
    )


    ax.set_xlim(
        0,
        24
    )

    ax.set_xticks(
        range(0,25,3)
    )

    ax.set_xlabel(
        "Time of day"
    )

    ax.set_ylabel(
        "Relative probability"
    )

    ax.legend(
        frameon=False
    )


    return savefig(
        fig,
        "phase_density.png"
    )



# ----------------------------------------------------------------------
# Interval distribution
# ----------------------------------------------------------------------


def plot_intervals(intervals):

    fig, ax = plt.subplots(
        figsize=(11,5)
    )


    ax.hist(
        intervals,
        bins=40
    )


    median = np.median(intervals)


    ax.axvline(
        median,
        color=ACCENT,
        lw=2,
        label=f"Median {median:.1f} h"
    )


    ax.set_xlim(
        0,
        np.percentile(
            intervals,
            95
        )
    )


    ax.set_xlabel(
        "Hours since previous event"
    )

    ax.set_ylabel(
        "Events"
    )

    ax.legend(
        frameon=False
    )


    return savefig(
        fig,
        "intervals.png"
    )



# ----------------------------------------------------------------------
# Hazard
# ----------------------------------------------------------------------


def plot_hazard(
        x,
        y
):

    fig, ax = plt.subplots(
        figsize=(11,5)
    )


    ax.plot(
        x,
        y,
        marker="o",
        lw=2
    )


    ax.set_xlabel(
        "Hours since last event"
    )

    ax.set_ylabel(
        "Probability per hour"
    )


    return savefig(
        fig,
        "hazard.png"
    )



# ----------------------------------------------------------------------
# Weekend comparison
# ----------------------------------------------------------------------


def plot_weekend(
        df,
        result
):

    fig, ax = plt.subplots(
        figsize=(11,5)
    )


    bins=np.linspace(
        0,
        24,
        25
    )


    ax.hist(
        df.loc[
            ~df.is_weekend,
            "hour"
        ],
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        label="Weekday"
    )


    ax.hist(
        df.loc[
            df.is_weekend,
            "hour"
        ],
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        label="Weekend"
    )


    ax.axvline(
        result["weekday"]["mean"],
        ls=":"
    )


    ax.axvline(
        result["weekend"]["mean"],
        ls=":"
    )


    ax.set_xlim(
        0,
        24
    )


    ax.set_xlabel(
        "Hour of day"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.legend(
        frameon=False
    )


    return savefig(
        fig,
        "weekend.png"
    )



# ----------------------------------------------------------------------
# Phase drift
# ----------------------------------------------------------------------


def plot_phase_drift(
        monthly
):

    fig, ax = plt.subplots(
        figsize=(11,5)
    )


    ax.plot(
        monthly.month,
        monthly.mean,
        marker="o"
    )


    ax.set_ylim(
        0,
        24
    )


    ax.set_ylabel(
        "Average event time"
    )


    ax.set_xlabel(
        "Month"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    return savefig(
        fig,
        "phase_drift.png"
    )



# ======================================================================
# HTML
# ======================================================================


CSS = """

body {
    max-width:1100px;
    margin:auto;
    padding:30px;
    background:#0b1020;
    color:#e8ecf5;
    font-family:Inter,Arial,sans-serif;
    line-height:1.6;
}


h1,h2 {
    font-family:"Space Grotesk",Arial;
}


.card {
    background:#101a30;
    border:1px solid #24304a;
    border-radius:12px;
    padding:20px;
    margin-bottom:25px;
}


img {
    width:100%;
    border-radius:8px;
}


.statgrid {
    display:grid;
    grid-template-columns:
    repeat(auto-fit,minmax(160px,1fr));
    gap:12px;
}


.stat {
    background:#101a30;
    border:1px solid #24304a;
    padding:15px;
    border-radius:10px;
}


.label {
    color:#8a94a8;
    font-size:.8rem;
}


.value {
    font-size:1.5rem;
}


footer {
    color:#8a94a8;
    margin-top:40px;
}

"""


def stat(
        label,
        value
):

    return f"""
    <div class="stat">
    <div class="label">{label}</div>
    <div class="value">{value}</div>
    </div>
    """



def card(
        title,
        image,
        text
):

    return f"""
    <div class="card">
    <h2>{title}</h2>
    <img src="figures/{image}">
    <p>{text}</p>
    </div>
    """



def build_html(
        stats_html,
        cards
):

    return f"""

<!DOCTYPE html>
<html>

<head>

<title>
The Poop Observatory
</title>

<style>
{CSS}
</style>

</head>


<body>


<h1>
The Poop Observatory
</h1>


<p>
A chronobiological analysis of a personal event stream.
The dashboard measures activity; this report examines rhythm.
</p>


<div class="statgrid">

{stats_html}

</div>


{cards}


<footer>
No timestamps displayed. The rhythm is the observation.
</footer>


</body>

</html>

"""
# ======================================================================
# Main
# ======================================================================


def fmt_clock(hour):

    h = int(hour) % 24
    m = int(
        (hour % 1) * 60
    )

    return f"{h:02d}:{m:02d}"



def main():

    FIGDIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    df = load_data(DATA)


    # --------------------------------------------------------------
    # Analyses
    # --------------------------------------------------------------

    phase = circular_summary(
        df.hour
    )

    density = phase_density(
        df.hour
    )


    intervals = interval_distribution(
        df
    )


    hazard_x, hazard_y = hazard_curve(
        df
    )


    weekend = weekend_test(
        df
    )


    monthly = monthly_phase(
        df
    )


    anomalies = detect_anomalies(
        df
    )


    days = (
        df.date.max()
        -
        df.date.min()
    ).days + 1



    # --------------------------------------------------------------
    # Figures
    # --------------------------------------------------------------

    actogram = plot_actogram(
        df
    )


    phase_plot = plot_phase_density(
        df,
        density
    )


    interval_plot = plot_intervals(
        intervals
    )


    hazard_plot = plot_hazard(
        hazard_x,
        hazard_y
    )


    weekend_plot = plot_weekend(
        df,
        weekend
    )


    drift_plot = plot_phase_drift(
        monthly
    )



    # --------------------------------------------------------------
    # Summary interpretation
    # --------------------------------------------------------------

    if phase["R"] > 0.7:
        rhythm = "strong"
    elif phase["R"] > 0.4:
        rhythm = "moderate"
    else:
        rhythm = "weak"



    weekend_statement = (
        "No meaningful weekend shift detected."
        if weekend["p"] > 0.05
        else
        "Weekend timing appears shifted."
    )



    anomaly_statement = (
        "No unusual timing excursions detected."
        if anomalies["count"] == 0
        else
        f"{anomalies['count']} unusually long or short intervals detected."
    )



    # --------------------------------------------------------------
    # Readout cards
    # --------------------------------------------------------------

    stats_html = "".join([

        stat(
            "Observations",
            len(df)
        ),

        stat(
            "Days observed",
            days
        ),

        stat(
            "Typical time",
            fmt_clock(
                phase["mean"]
            )
        ),

        stat(
            "Rhythm strength",
            f"{rhythm}"
        ),

        stat(
            "Phase scatter",
            f"±{phase['sd']:.1f} h"
        ),

        stat(
            "Median interval",
            f"{np.median(intervals):.1f} h"
        ),

        stat(
            "Weekend shift",
            f"{weekend['difference']:.1f} h"
        ),

        stat(
            "Anomalies",
            anomalies["count"]
        )

    ])



    # --------------------------------------------------------------
    # Report sections
    # --------------------------------------------------------------

    cards = "".join([


        card(
            "The orbit",
            actogram,
            """
            The actogram shows every event placed on its natural
            24-hour cycle. Vertical alignment indicates a stable
            daily schedule; drift indicates changes in timing.
            """
        ),



        card(
            "The daily waveform",
            phase_plot,
            f"""
            Folding all events into one artificial day reveals the
            underlying rhythm. The average event occurs around
            <b>{fmt_clock(phase['mean'])}</b> with a circular
            concentration of <b>{phase['R']:.2f}</b>.
            """
        ),



        card(
            "The cadence",
            interval_plot,
            """
            The interval distribution shows the natural spacing
            between events. A narrow peak suggests a regular
            rhythm; a broad distribution indicates variability.
            """
        ),



        card(
            "The biological clock",
            hazard_plot,
            """
            The hazard curve estimates when another event becomes
            increasingly likely after the previous event.
            """
        ),



        card(
            "Social perturbation",
            weekend_plot,
            f"""
            Weekday and weekend schedules were compared using
            circular permutation testing.
            {weekend_statement}
            """
        ),



        card(
            "Long-term stability",
            drift_plot,
            """
            Monthly average timing reveals whether the rhythm has
            shifted across the observation period.
            """
        ),


        card(
            "Observatory notes",
            "",
            f"""
            Overall rhythm: <b>{rhythm}</b>.<br><br>

            {weekend_statement}<br>

            {anomaly_statement}<br><br>

            This report treats the log as a behavioural time series,
            not as a medical measurement.
            """
        )

    ])



    # --------------------------------------------------------------
    # Write
    # --------------------------------------------------------------

    html = build_html(
        stats_html,
        cards
    )


    OUTPUT.write_text(
        html,
        encoding="utf-8"
    )


    print(
        "\n"
        "====================================\n"
        "Poop Observatory complete\n"
        "====================================\n"
        f"Events: {len(df)}\n"
        f"Days: {days}\n"
        f"Mean time: {fmt_clock(phase['mean'])}\n"
        f"Rhythm strength: {phase['R']:.2f}\n"
        f"Median interval: {np.median(intervals):.1f} h\n"
        f"Weekend shift: {weekend['difference']:.1f} h "
        f"(p={weekend['p']:.3f})\n"
        f"Anomalies: {anomalies['count']}\n"
        f"\nWritten: {OUTPUT}\n"
    )



if __name__ == "__main__":
    main()
