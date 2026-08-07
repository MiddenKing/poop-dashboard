"""
The Poop Observatory
====================

Builds a self-contained static site (index.html + figures/*.png) for GitHub
Pages from a raw poop log.

Input CSV:
    data/poop_data.csv

Required columns:
    Time, Value

Value:
    1 = good
    0 = poor

Dependencies:
    pandas
    numpy
    scipy
    matplotlib

Design philosophy
-----------------
The dashboard is organised as a scientific narrative:

    1. At a glance
    2. The daily rhythm
    3. Timing and intervals
    4. Stability over time
    5. Quality
    6. Null-model diagnostics

Analytics and plotting are kept separate where practical so that the
statistical calculations can be tested independently of the figures.

The 24-hour dial remains the signature visualisation.

Important statistical interpretation
------------------------------------
The "timing concentration" value is the circular mean resultant length R:

    R = 0  -> timing is spread around the whole clock
    R = 1  -> observations are concentrated at one time

It is displayed as a concentration score, not as a literal percentage of
"regularity".

The Rayleigh test tests the null hypothesis that clock times are uniformly
distributed around the 24-hour cycle.

The Poisson and exponential overlays are presented as simple null-model
diagnostics, not as claims that bowel behaviour is intrinsically Poisson or
memoryless.

Run:
    python generate_dashboard.py
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ==========================================================================
# Config
# ==========================================================================

DATA = Path("data/poop_data.csv")
FIGURES = Path("figures")
OUTPUT = Path("index.html")
SUMMARY = Path("summary.json")

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

GOOD_COLOR = "#2f6f8f"
POOR_COLOR = "#d98a3d"
INK = "#16233b"
MUTED = "#5b6b82"
GRID = "#e4e9ef"
PAPER = "#eef1f4"
PANEL = "#ffffff"
LIGHT_BLUE = "#e9f1f6"

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

HOUR_ORDER = list(range(24))

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


# ==========================================================================
# Load + validate
# ==========================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the raw poop log."""

    df = pd.read_csv(path)

    time_col = next(
        (c for c in df.columns if c.lower() == "time"),
        None,
    )
    value_col = next(
        (c for c in df.columns if c.lower() == "value"),
        None,
    )

    if time_col is None:
        raise ValueError("CSV must contain a 'Time' column.")

    if value_col is None:
        raise ValueError("CSV must contain a 'Value' column.")

    # Raw CSV may contain timestamps such as:
    # 2026-6-16 10:23:3
    # so use mixed parsing rather than assuming one layout.
    df["datetime"] = pd.to_datetime(
        df[time_col].astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )

    bad_time = df[df["datetime"].isna()]

    if not bad_time.empty:
        print("Unparseable timestamps:")
        print(bad_time[[time_col, value_col]])
        raise ValueError(
            "Fix timestamp parsing before building the site."
        )

    # Convert value safely.
    df["value"] = pd.to_numeric(
        df[value_col],
        errors="coerce",
    )

    if df["value"].isna().any():
        raise ValueError("Value column contains non-numeric entries.")

    invalid_values = ~df["value"].isin([0, 1])

    if invalid_values.any():
        print("Invalid Value entries:")
        print(df.loc[invalid_values, [time_col, value_col]])
        raise ValueError("Value must contain only 0 = poor and 1 = good.")

    df["value"] = df["value"].astype(int)

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    df["date"] = df["datetime"].dt.floor("D")
    df["hour"] = df["datetime"].dt.hour

    df["hour_decimal"] = (
        df["datetime"].dt.hour
        + df["datetime"].dt.minute / 60.0
        + df["datetime"].dt.second / 3600.0
    )

    df["weekday"] = df["datetime"].dt.day_name()
    df["month"] = df["datetime"].dt.to_period("M")

    df["is_good"] = df["value"].eq(1)
    df["is_poor"] = df["value"].eq(0)

    return df


def daily_counts(
    df: pd.DataFrame,
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return one row per calendar day, including zero-observation days."""

    return (
        df.groupby("date")
        .size()
        .reindex(date_range, fill_value=0)
        .rename("count")
        .reset_index()
        .rename(columns={"index": "date"})
    )


# ==========================================================================
# General helpers
# ==========================================================================

def _fmt_clock(hour_float: float) -> str:
    """Convert decimal hours to HH:MM."""

    total_minutes = int(round(hour_float * 60)) % (24 * 60)

    h = total_minutes // 60
    m = total_minutes % 60

    return f"{h:02d}:{m:02d}"


def _fmt_p(p: float) -> str:
    """Compact p-value formatter."""

    if not np.isfinite(p):
        return "p = NA"

    if p < 0.001:
        return "p < 0.001"

    return f"p = {p:.3f}"


def _fmt_p_short(p: float) -> str:
    """Compact p-value formatter for HTML cards."""

    if not np.isfinite(p):
        return "NA"

    if p < 0.001:
        return "<0.001"

    return f"{p:.3f}"


def _safe_rate(numerator, denominator):
    """Safe percentage calculation."""

    if denominator == 0:
        return np.nan

    return numerator / denominator * 100.0


# ==========================================================================
# Analytics
# ==========================================================================

def quality_split(
    df: pd.DataFrame,
    key: str,
    index,
) -> pd.DataFrame:
    """
    Count good and poor observations for each level of `key`.

    Returns:
        good
        poor

    The index is explicitly aligned to `index`.
    """

    grid = (
        df.groupby([key, "value"])
        .size()
        .unstack(fill_value=0)
    )

    for value in (0, 1):
        if value not in grid.columns:
            grid[value] = 0

    grid = grid.reindex(index, fill_value=0)

    return pd.DataFrame(
        {
            "good": grid[1],
            "poor": grid[0],
        },
        index=grid.index,
    )


def circular_stats(hours_decimal: np.ndarray) -> dict:
    """
    Circular statistics for time of day.

    R is the mean resultant length.

    R = 0:
        observations are evenly distributed around the clock.

    R = 1:
        all observations occur at the same time.

    Rayleigh's test evaluates whether the observed concentration differs
    from a uniform circular distribution.
    """

    hours_decimal = np.asarray(hours_decimal, dtype=float)

    hours_decimal = hours_decimal[np.isfinite(hours_decimal)]

    n = len(hours_decimal)

    if n == 0:
        return {
            "R": 0.0,
            "mean_hour": 0.0,
            "rayleigh_p": 1.0,
            "n": 0,
        }

    theta = (
        2.0
        * np.pi
        * hours_decimal
        / 24.0
    )

    c = np.cos(theta).mean()
    s = np.sin(theta).mean()

    R = float(np.hypot(c, s))

    mean_angle = np.arctan2(s, c) % (2 * np.pi)

    mean_hour = float(
        mean_angle / (2 * np.pi) * 24.0
    )

    # Zar-style approximation to Rayleigh p-value.
    Z = n * R * R

    if n <= 1:
        p = 1.0
    else:
        p = np.exp(-Z) * (
            1
            + (2 * Z - Z**2) / (4 * n)
            - (
                24 * Z
                - 132 * Z**2
                + 76 * Z**3
                - 9 * Z**4
            ) / (288 * n**2)
        )

    p = float(np.clip(p, 0.0, 1.0))

    return {
        "R": R,
        "mean_hour": mean_hour,
        "rayleigh_p": p,
        "n": n,
    }


def hour_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Observed counts by hour."""

    return (
        df["hour"]
        .value_counts()
        .reindex(HOUR_ORDER, fill_value=0)
        .rename("count")
        .to_frame()
    )


def weekday_hour_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Weekday × hour observation matrix."""

    matrix = (
        pd.crosstab(df["weekday"], df["hour"])
        .reindex(
            index=WEEKDAY_ORDER,
            columns=HOUR_ORDER,
            fill_value=0,
        )
    )

    return matrix


def interval_stats(df: pd.DataFrame) -> dict:
    """Statistics for time between consecutive observations."""

    gaps = (
        df["datetime"]
        .diff()
        .dt.total_seconds()
        .div(3600)
        .dropna()
    )

    if gaps.empty:
        return {
            "median_gap": np.nan,
            "mean_gap": np.nan,
            "q25_gap": np.nan,
            "q75_gap": np.nan,
            "gaps": gaps,
        }

    return {
        "median_gap": float(gaps.median()),
        "mean_gap": float(gaps.mean()),
        "q25_gap": float(gaps.quantile(0.25)),
        "q75_gap": float(gaps.quantile(0.75)),
        "gaps": gaps,
    }


def weekday_frequency_test(
    df: pd.DataFrame,
    date_range: pd.DatetimeIndex,
) -> dict:
    """
    Test whether observations are evenly distributed over weekdays.

    Expected counts are weighted according to how many Mondays, Tuesdays,
    etc. are actually represented in the observation window.
    """

    observed = (
        df["weekday"]
        .value_counts()
        .reindex(WEEKDAY_ORDER, fill_value=0)
        .to_numpy(float)
    )

    weekday_days = (
        pd.Series(date_range.day_name())
        .value_counts()
        .reindex(WEEKDAY_ORDER, fill_value=0)
        .to_numpy(float)
    )

    total_days = weekday_days.sum()

    if total_days == 0 or observed.sum() == 0:
        return {
            "chi2": np.nan,
            "p": np.nan,
            "busiest_weekday": "NA",
        }

    expected = (
        weekday_days
        / total_days
        * observed.sum()
    )

    # Only meaningful where expected > 0.
    mask = expected > 0

    chi2, p = stats.chisquare(
        observed[mask],
        f_exp=expected[mask],
    )

    rate = np.divide(
        observed,
        weekday_days,
        out=np.zeros_like(observed),
        where=weekday_days > 0,
    )

    busiest = WEEKDAY_ORDER[int(np.argmax(rate))]

    return {
        "chi2": float(chi2),
        "p": float(p),
        "busiest_weekday": busiest,
    }


def weekday_quality_test(df: pd.DataFrame) -> dict:
    """
    Test whether good/poor quality depends on weekday.

    Uses Fisher-Freeman-Halton only if available would be preferable, but
    scipy's standard chi-square contingency test is retained here with a
    sparse-cell flag.
    """

    table = (
        pd.crosstab(df["weekday"], df["value"])
        .reindex(
            WEEKDAY_ORDER,
            fill_value=0,
        )
    )

    for value in (0, 1):
        if value not in table.columns:
            table[value] = 0

    table = table[[1, 0]]

    chi2, p, dof, expected = stats.chi2_contingency(
        table.to_numpy()
    )

    sparse = bool((expected < 5).any())

    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "sparse": sparse,
        "table": table,
        "expected": expected,
    }


def quality_timing_test(df: pd.DataFrame) -> dict:
    """
    Compare timing of good and poor observations.

    Uses the two-sample Kolmogorov-Smirnov test on clock-time values after
    representing the day linearly from 00:00 to 24:00.

    This is deliberately treated as a supplementary diagnostic because
    clock time is circular and the KS test is not itself circular.
    """

    good = df.loc[df["is_good"], "hour_decimal"].to_numpy()
    poor = df.loc[df["is_poor"], "hour_decimal"].to_numpy()

    if len(good) < 2 or len(poor) < 2:
        return {
            "statistic": np.nan,
            "p": np.nan,
            "n_good": len(good),
            "n_poor": len(poor),
        }

    statistic, p = stats.ks_2samp(
        good,
        poor,
        alternative="two-sided",
        method="auto",
    )

    return {
        "statistic": float(statistic),
        "p": float(p),
        "n_good": len(good),
        "n_poor": len(poor),
    }


def poisson_diagnostic(daily: pd.DataFrame) -> dict:
    """
    Fit a Poisson null model using the observed mean daily count.

    Returns model expectation plus a dispersion statistic.
    """

    counts = daily["count"].to_numpy(float)

    lam = float(counts.mean())

    variance = float(counts.var(ddof=1)) if len(counts) > 1 else np.nan

    dispersion = (
        variance / lam
        if lam > 0 and np.isfinite(variance)
        else np.nan
    )

    return {
        "lambda": lam,
        "variance": variance,
        "dispersion": float(dispersion),
    }


def exponential_diagnostic(gaps: pd.Series) -> dict:
    """Fit the simple exponential null model using the observed mean gap."""

    gaps = np.asarray(gaps, dtype=float)

    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]

    if len(gaps) == 0:
        return {
            "mean_gap": np.nan,
            "median_gap": np.nan,
            "n": 0,
        }

    return {
        "mean_gap": float(gaps.mean()),
        "median_gap": float(np.median(gaps)),
        "n": int(len(gaps)),
    }


def find_streaks(daily: pd.DataFrame) -> pd.DataFrame:
    """Find consecutive runs of days with at least one observation."""

    has = daily["count"].gt(0)

    run_id = (
        has != has.shift()
    ).cumsum()

    runs = (
        daily.assign(
            has=has,
            run=run_id,
        )[has]
        .groupby("run")
        .agg(
            start=("date", "first"),
            end=("date", "last"),
            length=("date", "size"),
        )
        .reset_index(drop=True)
    )

    if runs.empty:
        return runs

    return (
        runs.sort_values(
            ["length", "start"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def poor_events_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Daily poor-observation counts, including zero days."""

    return (
        df.loc[df["is_poor"]]
        .groupby("date")
        .size()
        .rename("poor")
    )


# ==========================================================================
# Summary
# ==========================================================================

def build_summary(
    df,
    daily,
    circ,
    wf_test,
    wq_test,
    timing_test,
    intervals,
    poisson,
    exponential,
) -> dict:

    days = len(daily)

    zero_days = int(
        (daily["count"] == 0).sum()
    )

    poor = int(
        df["is_poor"].sum()
    )

    good = int(
        df["is_good"].sum()
    )

    return {
        "observations": int(len(df)),
        "days": int(days),
        "date_start": df["date"].min().strftime("%Y-%m-%d"),
        "date_end": df["date"].max().strftime("%Y-%m-%d"),

        "days_with_observation": int(
            (daily["count"] > 0).sum()
        ),

        "zero_observation_days": zero_days,

        "average_per_day": round(
            len(df) / days,
            2,
        ),

        "median_per_day": float(
            daily["count"].median()
        ),

        "max_per_day": int(
            daily["count"].max()
        ),

        "good": good,
        "poor": poor,

        "good_rate": round(
            _safe_rate(good, len(df)),
            1,
        ),

        "poor_rate": round(
            _safe_rate(poor, len(df)),
            1,
        ),

        "most_common_hour": int(
            df["hour"].mode().iloc[0]
        ),

        "timing_concentration": round(
            circ["R"],
            3,
        ),

        "timing_concentration_percent": round(
            circ["R"] * 100,
            1,
        ),

        "mean_clock_time": _fmt_clock(
            circ["mean_hour"]
        ),

        "rayleigh_p": circ["rayleigh_p"],

        "busiest_weekday": wf_test["busiest_weekday"],
        "weekday_freq_p": wf_test["p"],

        "weekday_quality_p": wq_test["p"],
        "weekday_quality_sparse": wq_test["sparse"],

        "quality_timing_p": timing_test["p"],
        "quality_timing_ks": timing_test["statistic"],

        "typical_gap_hours": round(
            float(intervals["median_gap"]),
            1,
        ),

        "mean_gap_hours": round(
            float(intervals["mean_gap"]),
            1,
        ),

        "gap_q25_hours": round(
            float(intervals["q25_gap"]),
            1,
        ),

        "gap_q75_hours": round(
            float(intervals["q75_gap"]),
            1,
        ),

        "poisson_lambda": round(
            poisson["lambda"],
            3,
        ),

        "poisson_dispersion": round(
            poisson["dispersion"],
            3,
        ),

        "expected_zero_day_rate": round(
            float(
                stats.poisson.pmf(
                    0,
                    poisson["lambda"],
                )
            ) * 100,
            1,
        ),

        "observed_zero_day_rate": round(
            zero_days / days * 100,
            1,
        ),

        "exponential_mean_gap": round(
            exponential["mean_gap"],
            1,
        ),
    }


# ==========================================================================
# Plot helpers
# ==========================================================================

def _finish(
    fig,
    name: str,
) -> str:

    fig.tight_layout()

    fig.savefig(
        FIGURES / name,
        transparent=True,
        bbox_inches="tight",
    )

    plt.close(fig)

    return name


def _stacked(
    ax,
    x,
    good,
    poor,
    width=None,
):

    kwargs = {}

    if width is not None:
        kwargs["width"] = width

    ax.bar(
        x,
        good,
        color=GOOD_COLOR,
        label="Good",
        **kwargs,
    )

    ax.bar(
        x,
        poor,
        bottom=good,
        color=POOR_COLOR,
        label="Poor",
        **kwargs,
    )

    ax.legend(
        frameon=False,
    )


# ==========================================================================
# Plots: Daily rhythm
# ==========================================================================

def fig_dial(df) -> str:
    """
    Signature visualisation.

    00:00 at top, clockwise.
    Bar length = total observations.
    Blue = good.
    Amber = poor.
    """

    split = quality_split(
        df,
        "hour",
        HOUR_ORDER,
    )

    theta = np.array([
        2 * np.pi * h / 24
        for h in HOUR_ORDER
    ])

    width = (
        2 * np.pi / 24
    )

    fig = plt.figure(
        figsize=(6.6, 6.6)
    )

    ax = fig.add_subplot(
        projection="polar"
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.bar(
        theta,
        split["good"],
        width=width,
        align="edge",
        color=GOOD_COLOR,
        label="Good",
    )

    ax.bar(
        theta,
        split["poor"],
        width=width,
        align="edge",
        bottom=split["good"],
        color=POOR_COLOR,
        label="Poor",
    )

    ax.set_xticks([
        2 * np.pi * h / 24
        for h in range(0, 24, 2)
    ])

    ax.set_xticklabels([
        f"{h:02d}"
        for h in range(0, 24, 2)
    ])

    ax.set_yticklabels([])

    ax.grid(
        color=GRID
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )

    return _finish(
        fig,
        "dial.png",
    )


def fig_hour_distribution(df) -> str:
    """Linear 24-hour histogram with good/poor stacked."""

    split = quality_split(
        df,
        "hour",
        HOUR_ORDER,
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    x = np.arange(24)

    _stacked(
        ax,
        x,
        split["good"].to_numpy(),
        split["poor"].to_numpy(),
    )

    ax.set_xlabel(
        "Hour of day"
    )

    ax.set_ylabel(
        "Observations"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [f"{h:02d}" for h in HOUR_ORDER]
    )

    return _finish(
        fig,
        "hour_distribution.png",
    )


def fig_weekday_hour_heatmap(df) -> str:
    """
    Weekday × hour heatmap.

    This combines two previously separate questions:
    when during the day, and on which days of the week?
    """

    matrix = weekday_hour_matrix(df)

    cmap = LinearSegmentedColormap.from_list(
        "activity",
        [
            "#f3f7f9",
            LIGHT_BLUE,
            GOOD_COLOR,
        ],
    )

    fig, ax = plt.subplots(
        figsize=(13, 4.8)
    )

    im = ax.imshow(
        matrix.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
    )

    ax.set_yticks(
        np.arange(7)
    )

    ax.set_yticklabels(
        [
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat",
            "Sun",
        ]
    )

    ax.set_xticks(
        np.arange(24)
    )

    ax.set_xticklabels(
        [
            f"{h:02d}"
            for h in HOUR_ORDER
        ]
    )

    ax.set_xlabel(
        "Hour of day"
    )

    ax.set_ylabel(
        "Day of week"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        "Observations"
    )

    return _finish(
        fig,
        "weekday_hour_heatmap.png",
    )


# ==========================================================================
# Plots: Intervals
# ==========================================================================

def fig_time_between(gaps) -> str:
    """
    Distribution of intervals between observations.

    The 99th percentile is used for display so a few very long gaps do not
    flatten the main distribution.

    The exponential curve is a null-model comparison.
    """

    gaps = np.asarray(
        gaps,
        dtype=float,
    )

    gaps = gaps[
        np.isfinite(gaps)
        & (gaps > 0)
    ]

    if len(gaps) == 0:
        gaps = np.array([0.0])

    cutoff = np.percentile(
        gaps,
        99,
    )

    display = gaps[
        gaps <= cutoff
    ]

    mean_gap = gaps.mean()

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    counts, edges, _ = ax.hist(
        display,
        bins=24,
        color=GOOD_COLOR,
        alpha=0.9,
        label="Observed",
    )

    centres = (
        edges[:-1]
        + edges[1:]
    ) / 2

    width = (
        edges[1]
        - edges[0]
    )

    if mean_gap > 0:
        expected = (
            np.exp(
                -centres / mean_gap
            )
            / mean_gap
            * len(display)
            * width
        )

        ax.plot(
            centres,
            expected,
            color=POOR_COLOR,
            linewidth=2,
            label=(
                f"Exponential "
                f"(mean {mean_gap:.1f} h)"
            ),
        )

    ax.set_xlabel(
        "Hours since previous observation"
    )

    ax.set_ylabel(
        "Number of intervals"
    )

    ax.legend(
        frameon=False
    )

    return _finish(
        fig,
        "time_between.png",
    )


# ==========================================================================
# Plots: Stability
# ==========================================================================

def fig_daily_counts(daily) -> str:
    """
    Daily observation count with 7-day and 30-day rolling means.
    """

    d = daily.copy()

    d["r7"] = (
        d["count"]
        .rolling(
            7,
            min_periods=1,
        )
        .mean()
    )

    d["r30"] = (
        d["count"]
        .rolling(
            30,
            min_periods=1,
        )
        .mean()
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.bar(
        d["date"],
        d["count"],
        color=GOOD_COLOR,
        alpha=0.20,
        width=1.0,
        label="Daily count",
    )

    ax.plot(
        d["date"],
        d["r7"],
        color=GOOD_COLOR,
        linewidth=2,
        label="7-day mean",
    )

    ax.plot(
        d["date"],
        d["r30"],
        color=POOR_COLOR,
        linewidth=2,
        linestyle=":",
        label="30-day mean",
    )

    ax.set_ylabel(
        "Observations per day"
    )

    ax.legend(
        frameon=False
    )

    ax.margins(
        x=0
    )

    return _finish(
        fig,
        "daily_counts.png",
    )


def fig_cumulative(daily) -> str:

    d = daily.copy()

    d["cum"] = (
        d["count"]
        .cumsum()
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.fill_between(
        d["date"],
        d["cum"],
        color=GOOD_COLOR,
        alpha=0.15,
    )

    ax.plot(
        d["date"],
        d["cum"],
        color=GOOD_COLOR,
        linewidth=2,
    )

    ax.set_ylabel(
        "Cumulative observations"
    )

    ax.margins(
        x=0
    )

    return _finish(
        fig,
        "cumulative.png",
    )


def fig_calendar(daily) -> str:

    d = daily.copy()

    d["weekday_num"] = (
        d["date"].dt.weekday
    )

    iso = d["date"].dt.isocalendar()

    d["year_week"] = (
        iso.year.astype(str)
        + "-W"
        + iso.week.astype(int)
        .astype(str)
        .str.zfill(2)
    )

    weeks = (
        d["year_week"]
        .drop_duplicates()
        .tolist()
    )

    week_idx = {
        week: i
        for i, week in enumerate(weeks)
    }

    z = np.full(
        (7, len(weeks)),
        np.nan,
    )

    for _, row in d.iterrows():

        z[
            int(row["weekday_num"])
        ][
            week_idx[row["year_week"]]
        ] = row["count"]

    cmap = LinearSegmentedColormap.from_list(
        "calendar",
        [
            "#f3f7f9",
            LIGHT_BLUE,
            GOOD_COLOR,
        ],
    )

    cmap.set_bad(
        "#f4f7f9"
    )

    fig, ax = plt.subplots(
        figsize=(13, 4.5)
    )

    im = ax.imshow(
        np.ma.masked_invalid(z),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
    )

    ax.set_yticks(
        range(7)
    )

    ax.set_yticklabels(
        [
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat",
            "Sun",
        ]
    )

    ticks = [
        i
        for i, _ in enumerate(weeks)
        if i % 4 == 0
    ]

    ax.set_xticks(
        ticks
    )

    ax.set_xticklabels(
        [
            weeks[i]
            for i in ticks
        ],
        rotation=45,
        ha="right",
    )

    ax.grid(
        False
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Observations per day",
    )

    return _finish(
        fig,
        "calendar.png",
    )


def fig_streaks(streaks) -> str:

    if streaks.empty:

        fig, ax = plt.subplots(
            figsize=FIGSIZE
        )

        ax.text(
            0.5,
            0.5,
            "No observation streaks available",
            ha="center",
            va="center",
        )

        ax.axis(
            "off"
        )

        return _finish(
            fig,
            "streaks.png",
        )

    top = (
        streaks
        .head(3)
        .iloc[::-1]
    )

    labels = [
        (
            f"{row.start:%Y-%m-%d}"
            f" → "
            f"{row.end:%Y-%m-%d}"
        )
        for row in top.itertuples()
    ]

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.barh(
        labels,
        top["length"],
        color=GOOD_COLOR,
    )

    ax.set_xlabel(
        "Consecutive days with ≥1 observation"
    )

    ax.grid(
        axis="y",
        visible=False,
    )

    return _finish(
        fig,
        "streaks.png",
    )


# ==========================================================================
# Plots: Quality
# ==========================================================================

def fig_quality_over_time(df) -> str:
    """
    Poor observations plotted as events over time.

    This is intentionally event-based rather than a misleading daily
    percentage when many days have few or no observations.
    """

    good = df.loc[
        df["is_good"]
    ]

    poor = df.loc[
        df["is_poor"]
    ]

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # All observations form a subtle baseline.
    ax.scatter(
        good["datetime"],
        np.ones(len(good)),
        s=16,
        color=GOOD_COLOR,
        alpha=0.28,
        label="Good",
    )

    ax.scatter(
        poor["datetime"],
        np.ones(len(poor)),
        s=34,
        color=POOR_COLOR,
        alpha=0.95,
        label="Poor",
        zorder=3,
    )

    ax.set_yticks(
        []
    )

    ax.set_ylabel(
        ""
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylim(
        0.7,
        1.3,
    )

    ax.legend(
        frameon=False
    )

    return _finish(
        fig,
        "quality_over_time.png",
    )


def fig_monthly_quality(df) -> str:
    """
    Monthly good/poor composition.

    Each bar shows the actual number of observations, while the labels
    communicate the good percentage.
    """

    month_range = pd.period_range(
        df["month"].min(),
        df["month"].max(),
        freq="M",
    )

    split = quality_split(
        df,
        "month",
        month_range,
    )

    total = (
        split["good"]
        + split["poor"]
    )

    good_rate = (
        split["good"]
        / total.replace(
            0,
            np.nan,
        )
        * 100
    )

    x = np.arange(
        len(month_range)
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.bar(
        x,
        split["good"].to_numpy(),
        color=GOOD_COLOR,
        label="Good",
    )

    ax.bar(
        x,
        split["poor"].to_numpy(),
        bottom=split["good"].to_numpy(),
        color=POOR_COLOR,
        label="Poor",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            str(m)
            for m in month_range
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Observations"
    )

    ax.legend(
        frameon=False
    )

    # Put good-rate labels above bars.
    ymax = max(
        total.max(),
        1,
    )

    for i, rate in enumerate(
        good_rate
    ):

        if np.isfinite(rate):

            ax.text(
                i,
                total.iloc[i]
                + ymax * 0.015,
                f"{rate:.0f}% good",
                ha="center",
                va="bottom",
                fontsize=9,
                color=MUTED,
            )

    return _finish(
        fig,
        "monthly_quality.png",
    )


def fig_poor_timing(df) -> str:
    """
    Poor-event timing compared against the total observation rhythm.

    Grey/blue bars = all observations.
    Amber = poor observations.

    The point is to reveal whether poor events occupy a different part of
    the daily rhythm.
    """

    all_counts = (
        df["hour"]
        .value_counts()
        .reindex(
            HOUR_ORDER,
            fill_value=0,
        )
    )

    poor_counts = (
        df.loc[df["is_poor"], "hour"]
        .value_counts()
        .reindex(
            HOUR_ORDER,
            fill_value=0,
        )
    )

    x = np.arange(
        24
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.bar(
        x,
        all_counts,
        color=GOOD_COLOR,
        alpha=0.18,
        label="All observations",
    )

    ax.bar(
        x,
        poor_counts,
        color=POOR_COLOR,
        alpha=0.95,
        label="Poor",
    )

    ax.set_xlabel(
        "Hour of day"
    )

    ax.set_ylabel(
        "Observations"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            f"{h:02d}"
            for h in HOUR_ORDER
        ]
    )

    ax.legend(
        frameon=False
    )

    return _finish(
        fig,
        "poor_timing.png",
    )


def fig_quality_donut(summary) -> str:
    """
    Simple overall quality composition.

    Kept as a compact visual rather than a primary analytical figure.
    """

    fig, ax = plt.subplots(
        figsize=(5.5, 5.5)
    )

    ax.pie(
        [
            summary["good"],
            summary["poor"],
        ],
        labels=[
            "Good",
            "Poor",
        ],
        colors=[
            GOOD_COLOR,
            POOR_COLOR,
        ],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        wedgeprops=dict(
            width=0.42,
            edgecolor="white",
        ),
    )

    ax.set(
        aspect="equal"
    )

    return _finish(
        fig,
        "quality_donut.png",
    )


# ==========================================================================
# Plots: Null models
# ==========================================================================

def fig_daily_distribution(
    daily,
    poisson,
) -> str:
    """
    Observed daily counts against Poisson expectation.
    """

    counts = (
        daily["count"]
        .value_counts()
        .sort_index()
    )

    max_count = int(
        daily["count"].max()
    )

    ks = np.arange(
        0,
        max_count + 1,
    )

    expected = (
        stats.poisson.pmf(
            ks,
            poisson["lambda"],
        )
        * len(daily)
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    ax.bar(
        counts.index,
        counts.values,
        color=GOOD_COLOR,
        alpha=0.85,
        label="Observed",
    )

    ax.plot(
        ks,
        expected,
        color=POOR_COLOR,
        marker="o",
        linewidth=2,
        label=(
            f"Poisson "
            f"(λ={poisson['lambda']:.2f})"
        ),
    )

    ax.set_xlabel(
        "Observations per day"
    )

    ax.set_ylabel(
        "Number of days"
    )

    ax.set_xticks(
        ks
    )

    ax.legend(
        frameon=False
    )

    return _finish(
        fig,
        "daily_distribution.png",
    )


# ==========================================================================
# HTML
# ==========================================================================

CSS = """
:root{
    --paper:#eef1f4;
    --panel:#ffffff;
    --ink:#16233b;
    --muted:#5b6b82;
    --line:#d9dfe6;
    --good:#2f6f8f;
    --poor:#d98a3d;
}

*{
    box-sizing:border-box;
}

html{
    scroll-behavior:smooth;
}

body{
    max-width:1280px;
    margin:auto;
    padding:24px;
    background:var(--paper);
    color:var(--ink);
    font-family:Inter,Arial,sans-serif;
    line-height:1.5;
}

.masthead{
    border-bottom:2px solid var(--ink);
    padding-bottom:18px;
    margin-bottom:8px;
}

.masthead h1{
    font-family:'Space Grotesk',sans-serif;
    font-weight:700;
    letter-spacing:-0.7px;
    margin:0;
    font-size:2.5rem;
}

.eyebrow{
    font-family:'IBM Plex Mono',monospace;
    text-transform:uppercase;
    letter-spacing:0.18em;
    font-size:0.72rem;
    color:var(--muted);
    margin:0 0 6px;
}

.subtitle{
    color:var(--muted);
    margin:7px 0 0;
}

.page-layout{
    display:grid;
    grid-template-columns:220px minmax(0,1fr);
    gap:28px;
    align-items:start;
    margin-top:24px;
}

.sidebar{
    position:sticky;
    top:20px;
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:12px;
    padding:16px;
}

.sidebar h2{
    font-family:'IBM Plex Mono',monospace;
    font-size:0.75rem;
    text-transform:uppercase;
    letter-spacing:0.12em;
    color:var(--muted);
    margin:0 0 12px;
}

.nav-section{
    margin-bottom:17px;
}

.nav-section h3{
    font-family:'IBM Plex Mono',monospace;
    font-size:0.67rem;
    text-transform:uppercase;
    letter-spacing:0.07em;
    color:var(--muted);
    margin:12px 0 4px;
}

.nav-section:first-of-type h3{
    margin-top:0;
}

.sidebar a{
    display:block;
    color:var(--ink);
    text-decoration:none;
    padding:6px 9px;
    border-radius:7px;
    font-size:0.88rem;
}

.sidebar a:hover{
    background:var(--paper);
}

.content{
    min-width:0;
}

.section-heading{
    margin:36px 0 14px;
    padding-top:8px;
    border-top:1px solid var(--line);
}

.section-heading h2{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.35rem;
    margin:0;
}

.section-heading p{
    color:var(--muted);
    margin:3px 0 0;
    font-size:0.88rem;
}

.readout{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(145px,1fr));
    gap:12px;
    margin-bottom:28px;
}

.stat{
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:12px;
    padding:14px 16px;
}

.stat h3{
    font-family:'IBM Plex Mono',monospace;
    font-size:0.66rem;
    text-transform:uppercase;
    letter-spacing:0.06em;
    color:var(--muted);
    margin:0 0 6px;
    font-weight:500;
}

.stat p{
    font-family:'IBM Plex Mono',monospace;
    font-size:1.65rem;
    font-weight:600;
    margin:0;
    line-height:1.1;
}

.stat span{
    font-size:0.76rem;
    color:var(--muted);
    font-family:Inter,sans-serif;
    font-weight:400;
    display:block;
    margin-top:4px;
}

.plot-card{
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:12px;
    padding:16px 18px;
    margin-bottom:22px;
    scroll-margin-top:20px;
}

.plot-card h3{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.08rem;
    margin:0 0 11px;
}

.plot-card img{
    width:100%;
    display:block;
}

.note{
    font-size:0.81rem;
    color:var(--muted);
    margin:9px 2px 0;
}

.methods{
    background:var(--panel);
    border:1px solid var(--line);
    border-radius:12px;
    padding:20px 22px;
    margin-top:30px;
}

.methods h2{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.2rem;
    margin:0 0 10px;
}

.methods h3{
    font-family:'Space Grotesk',sans-serif;
    font-size:0.98rem;
    margin:16px 0 4px;
}

.methods p{
    font-size:0.86rem;
    color:var(--muted);
    margin:6px 0;
}

footer{
    text-align:center;
    color:var(--muted);
    margin-top:36px;
    font-size:0.82rem;
    font-family:'IBM Plex Mono',monospace;
}

@media(max-width:850px){

    body{
        padding:14px;
    }

    .page-layout{
        grid-template-columns:1fr;
    }

    .sidebar{
        position:static;
    }

    .masthead h1{
        font-size:2rem;
    }
}
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Space+Grotesk:wght@500;600;700&'
    'family=IBM+Plex+Mono:wght@400;500;600&'
    'family=Inter:wght@400;500;600&display=swap" '
    'rel="stylesheet">'
)


def anchor(text: str) -> str:
    """Create a simple HTML anchor ID."""

    keep = [
        c.lower()
        if c.isalnum()
        else "-"
        for c in text
    ]

    out = "".join(
        keep
    )

    while "--" in out:
        out = out.replace(
            "--",
            "-",
        )

    return out.strip("-")


def stat_card(
    label,
    value,
    sub="",
):
    sub_html = (
        f"<span>{sub}</span>"
        if sub
        else ""
    )

    return (
        '<div class="stat">'
        f"<h3>{label}</h3>"
        f"<p>{value}</p>"
        f"{sub_html}"
        "</div>"
    )


def build_methods(
    summary,
) -> str:

    return (
        '<div class="methods">'
        "<h2>Methods &amp; interpretation</h2>"

        "<h3>Timing concentration</h3>"
        f"<p><strong>R = "
        f"{summary['timing_concentration']:.3f}.</strong> "
        "This is the mean resultant length of the clock times. "
        "R = 0 means observations are spread evenly around the 24-hour "
        "cycle; R = 1 means they are concentrated at a single time. "
        f"The circular mean time is "
        f"<strong>{summary['mean_clock_time']}</strong>. "
        f"The Rayleigh test gives "
        f"<strong>{_fmt_p(summary['rayleigh_p'])}</strong> "
        "for the null hypothesis of uniform timing."
        "</p>"

        "<h3>Weekday frequency</h3>"
        f"<p>Weekday counts are compared against the number of Mondays, "
        f"Tuesdays, etc. actually present in the observation window, "
        f"rather than assuming exactly one seventh of observations should "
        f"fall on each day. "
        f"Result: <strong>{_fmt_p(summary['weekday_freq_p'])}</strong>."
        "</p>"

        "<h3>Quality and weekday</h3>"
        f"<p>A contingency test asks whether the good/poor composition "
        f"depends on weekday. "
        f"Result: <strong>{_fmt_p(summary['weekday_quality_p'])}</strong>."
        + (
            " Some expected cells are sparse, so this result should be "
            "treated cautiously."
            if summary["weekday_quality_sparse"]
            else ""
        )
        + "</p>"

        "<h3>Quality and time of day</h3>"
        f"<p>A two-sample Kolmogorov–Smirnov test compares the clock-time "
        f"distributions of good and poor observations. "
        f"Result: <strong>{_fmt_p(summary['quality_timing_p'])}</strong>. "
        "Because clock time is circular, this is presented as a "
        "supplementary diagnostic rather than a definitive circular "
        "test.</p>"

        "<h3>Daily counts</h3>"
        f"<p>The observed daily count distribution is compared with a "
        f"Poisson model whose rate is estimated from the observed mean "
        f"(<strong>λ = {summary['poisson_lambda']:.2f}</strong>). "
        f"The variance-to-mean dispersion ratio is "
        f"<strong>{summary['poisson_dispersion']:.2f}</strong>. "
        "Values around 1 are broadly consistent with Poisson dispersion; "
        "substantially larger values indicate overdispersion."
        "</p>"

        "<h3>Intervals</h3>"
        f"<p>The interval plot compares observed gaps with an exponential "
        f"null model based on the observed mean gap "
        f"(<strong>{summary['exponential_mean_gap']:.1f} h</strong>). "
        "The model is used as a reference distribution, not as an "
        "assumption that the underlying behaviour is memoryless."
        "</p>"

        "<h3>Quality definition</h3>"
        f"<p><strong>Good</strong> = Value 1. "
        f"<strong>Poor</strong> = Value 0. "
        f"The dashboard uses the raw values directly; no quality values "
        f"are inferred or imputed.</p>"

        "</div>"
    )


def build_html(
    summary,
    plots,
) -> str:

    # Navigation grouped by section.
    sections = []

    for plot in plots:

        if plot["section"] not in sections:
            sections.append(
                plot["section"]
            )

    nav_parts = []

    for section in sections:

        links = "".join(
            (
                f'<a href="#{anchor(plot["title"])}">'
                f'{plot["title"]}</a>'
            )
            for plot in plots
            if plot["section"] == section
        )

        nav_parts.append(
            '<div class="nav-section">'
            f"<h3>{section}</h3>"
            f"{links}"
            "</div>"
        )

    nav_html = "".join(
        nav_parts
    )

    # Grouped content.
    cards = []

    previous_section = None

    for plot in plots:

        if plot["section"] != previous_section:

            cards.append(
                '<div class="section-heading">'
                f'<h2>{plot["section"]}</h2>'
                f'<p>{plot["section_description"]}</p>'
                "</div>"
            )

            previous_section = plot[
                "section"
            ]

        note = (
            f'<p class="note">{plot["note"]}</p>'
            if plot.get("note")
            else ""
        )

        cards.append(
            f'<div class="plot-card" '
            f'id="{anchor(plot["title"])}">'
            f'<h3>{plot["title"]}</h3>'
            f'<img src="figures/{plot["file"]}" '
            f'alt="{plot["title"]}">'
            f"{note}"
            "</div>"
        )

    readout = "".join([
        stat_card(
            "Observations",
            summary["observations"],
        ),

        stat_card(
            "Days covered",
            summary["days"],
            f'{summary["date_start"]} → '
            f'{summary["date_end"]}',
        ),

        stat_card(
            "Per day",
            summary["average_per_day"],
            f'median {summary["median_per_day"]:.1f}',
        ),

        stat_card(
            "Good",
            f'{summary["good_rate"]}%',
            f'{summary["good"]} observations',
        ),

        stat_card(
            "Poor",
            f'{summary["poor_rate"]}%',
            f'{summary["poor"]} observations',
        ),

        stat_card(
            "Timing concentration",
            f'{summary["timing_concentration"]:.2f}',
            f'mean time {summary["mean_clock_time"]}',
        ),

        stat_card(
            "Typical gap",
            f'{summary["typical_gap_hours"]} h',
            f'IQR {summary["gap_q25_hours"]}–'
            f'{summary["gap_q75_hours"]} h',
        ),

        stat_card(
            "Busiest weekday",
            summary["busiest_weekday"],
            f'frequency {_fmt_p_short(summary["weekday_freq_p"])}',
        ),
    ])

    methods = build_methods(
        summary
    )

    return (
        "<!DOCTYPE html>"
        "<html lang='en'>"
        "<head>"
        "<meta charset='UTF-8'>"
        "<meta name='viewport' "
        "content='width=device-width,initial-scale=1'>"

        "<title>The Poop Observatory</title>"

        + FONTS

        + "<style>"
        + CSS
        + "</style>"

        "</head>"

        "<body>"

        "<div class='masthead'>"

        "<p class='eyebrow'>"
        "An instrument reading of the daily rhythm"
        "</p>"

        "<h1>The Poop Observatory</h1>"

        "<p class='subtitle'>"
        "A statistical field notebook of frequency, timing, "
        "regularity and quality. "
        "<a href='observatory.html' "
        "style='color:#2f6f8f;font-weight:600;"
        "text-decoration:none'>"
        "Observation report →"
        "</a>"
        "</p>"

        "</div>"

        "<div class='page-layout'>"

        f"<nav class='sidebar'>"
        "<h2>Readings</h2>"
        f"{nav_html}"
        "</nav>"

        "<main class='content'>"

        f"<div class='readout'>"
        f"{readout}"
        "</div>"

        f"{''.join(cards)}"

        f"{methods}"

        "</main>"

        "</div>"

        "<footer>"
        "Raw timestamps are not displayed on this site."
        "</footer>"

        "</body>"
        "</html>"
    )


# ==========================================================================
# Main
# ==========================================================================

def main():

    FIGURES.mkdir(
        exist_ok=True
    )

    df = load_data(
        DATA
    )

    # ----------------------------------------------------------------------
    # Observation window
    # ----------------------------------------------------------------------

    date_range = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D",
    )

    daily = daily_counts(
        df,
        date_range,
    )

    # ----------------------------------------------------------------------
    # Analytics
    # ----------------------------------------------------------------------

    circ = circular_stats(
        df["hour_decimal"].to_numpy()
    )

    wf_test = weekday_frequency_test(
        df,
        date_range,
    )

    wq_test = weekday_quality_test(
        df
    )

    timing_test = quality_timing_test(
        df
    )

    intervals = interval_stats(
        df
    )

    poisson = poisson_diagnostic(
        daily
    )

    exponential = exponential_diagnostic(
        intervals["gaps"]
    )

    streaks = find_streaks(
        daily
    )

    summary = build_summary(
        df,
        daily,
        circ,
        wf_test,
        wq_test,
        timing_test,
        intervals,
        poisson,
        exponential,
    )

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ----------------------------------------------------------------------
    # Console output
    # ----------------------------------------------------------------------

    print(
        f"Range "
        f"{df['date'].min():%Y-%m-%d} "
        f"→ "
        f"{df['date'].max():%Y-%m-%d}"
        f" | "
        f"{summary['observations']} observations "
        f"over "
        f"{summary['days']} days"
    )

    print(
        f"Quality: "
        f"{summary['good']} good "
        f"({summary['good_rate']}%) | "
        f"{summary['poor']} poor "
        f"({summary['poor_rate']}%)"
    )

    print(
        f"Timing concentration: "
        f"R={summary['timing_concentration']:.3f} "
        f"at mean time "
        f"{summary['mean_clock_time']} "
        f"({_fmt_p(summary['rayleigh_p'])})"
    )

    print(
        f"Typical gap: "
        f"{summary['typical_gap_hours']} h "
        f"(IQR "
        f"{summary['gap_q25_hours']}"
        f"–"
        f"{summary['gap_q75_hours']} h)"
    )

    print(
        f"Poisson λ="
        f"{summary['poisson_lambda']:.2f}; "
        f"dispersion="
        f"{summary['poisson_dispersion']:.2f}"
    )

    # ----------------------------------------------------------------------
    # Plot registry
    #
    # The order is deliberately narrative:
    #
    #   daily rhythm
    #   timing
    #   stability
    #   quality
    #   null models
    #
    # Adding/reordering entries here changes the page automatically.
    # ----------------------------------------------------------------------

    registry = [

        # ------------------------------------------------------------------
        # Daily rhythm
        # ------------------------------------------------------------------

        (
            "The daily rhythm",
            "Signature 24-hour dial",
            fig_dial(df),
            (
                "00:00 is at the top and time runs clockwise. "
                "Bar length represents the number of observations; "
                "amber marks poor observations."
            ),
            "The daily rhythm",
            "The signature reading: when during the 24-hour cycle do observations occur?",
        ),

        (
            "The daily rhythm",
            "Observations by hour",
            fig_hour_distribution(df),
            (
                "The linear view makes peaks and troughs easier to compare "
                "than the polar dial."
            ),
            "The daily rhythm",
            "A linear view of the same daily timing pattern.",
        ),

        (
            "The daily rhythm",
            "Weekday × hour activity",
            fig_weekday_hour_heatmap(df),
            (
                "Each cell is the number of observations for that weekday "
                "and hour."
            ),
            "The daily rhythm",
            "Does the daily rhythm change depending on the day of the week?",
        ),

        # ------------------------------------------------------------------
        # Timing and intervals
        # ------------------------------------------------------------------

        (
            "Timing & intervals",
            "Time between observations",
            fig_time_between(intervals["gaps"]),
            (
                "Long gaps above the 99th percentile are omitted from the "
                "display so the main distribution remains visible. "
                "The exponential curve is a null-model reference."
            ),
            "Timing & intervals",
            "How closely spaced are consecutive observations?",
        ),

        # ------------------------------------------------------------------
        # Stability
        # ------------------------------------------------------------------

        (
            "Stability over time",
            "Daily observations and rolling means",
            fig_daily_counts(daily),
            (
                "Faint bars show individual days; solid = 7-day mean; "
                "dotted = 30-day mean."
            ),
            "Stability over time",
            "Does the overall frequency remain stable, or does it drift?",
        ),

        (
            "Stability over time",
            "Observation calendar",
            fig_calendar(daily),
            (
                "Each cell is one calendar day. Darker cells contain more "
                "observations; pale cells contain none."
            ),
            "Stability over time",
            "Where are the dense and quiet periods in the log?",
        ),

        (
            "Stability over time",
            "Top 3 longest observation streaks",
            fig_streaks(streaks),
            (
                "A streak is a consecutive run of calendar days containing "
                "at least one observation."
            ),
            "Stability over time",
            "The longest uninterrupted runs of observed days.",
        ),

        (
            "Stability over time",
            "Cumulative observations",
            fig_cumulative(daily),
            (
                "The cumulative curve provides a simple long-term view of "
                "how observations accumulate."
            ),
            "Stability over time",
            "The long-term accumulation of observations.",
        ),

        # ------------------------------------------------------------------
        # Quality
        # ------------------------------------------------------------------

        (
            "Quality",
            "Good and poor observations over time",
            fig_quality_over_time(df),
            (
                "Good observations are shown faintly; poor observations "
                "are highlighted. This shows when poor events actually "
                "occurred without creating unstable daily percentages."
            ),
            "Quality",
            "Are poor observations isolated events or do they cluster in time?",
        ),

        (
            "Quality",
            "Monthly quality",
            fig_monthly_quality(df),
            (
                "Bars show the actual number of observations. Labels give "
                "the percentage classified as good."
            ),
            "Quality",
            "Does the composition of observations change over longer periods?",
        ),

        (
            "Quality",
            "When do poor observations occur?",
            fig_poor_timing(df),
            (
                "The pale blue distribution is all observations; amber "
                "shows the poor subset."
            ),
            "Quality",
            "Do poor observations occupy a different part of the daily rhythm?",
        ),

        (
            "Quality",
            "Overall quality",
            fig_quality_donut(summary),
            (
                "Good = Value 1. Poor = Value 0. The values are taken "
                "directly from the raw log."
            ),
            "Quality",
            "The overall composition of the dataset.",
        ),

        # ------------------------------------------------------------------
        # Null models
        # ------------------------------------------------------------------

        (
            "Null-model diagnostics",
            "Daily counts vs Poisson",
            fig_daily_distribution(
                daily,
                poisson,
            ),
            (
                "Bars are observed days; the line is the expected "
                "distribution from a Poisson model with the observed mean."
            ),
            "Null-model diagnostics",
            "Would a simple Poisson process reproduce the observed daily counts?",
        ),

    ]

    plots = [
        {
            "section": section,
            "title": title,
            "file": filename,
            "note": note,
            "section_description": section_description,
        }
        for (
            section,
            title,
            filename,
            note,
            _nav_section,
            section_description,
        )
        in registry
    ]

    # ----------------------------------------------------------------------
    # Build HTML
    # ----------------------------------------------------------------------

    html = build_html(
        summary,
        plots,
    )

    OUTPUT.write_text(
        html,
        encoding="utf-8",
    )

    print(
        f"Wrote {OUTPUT} "
        f"with {len(plots)} charts "
        f"into {FIGURES}/."
    )


if __name__ == "__main__":
    main()
