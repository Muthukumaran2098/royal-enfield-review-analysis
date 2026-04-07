"""
Visualization Module — Royal Enfield App Review Analysis
Generates 6 publication-ready charts for the interview presentation.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

CLASSIFIED_CSV    = os.path.join(PARENT_DIR, "data", "processed", "reviews_classified.csv")
RAW_CSV           = os.path.join(PARENT_DIR, "data", "raw", "reviews_raw.csv")
ANALYSIS_JSON     = os.path.join(PARENT_DIR, "outputs", "analysis_results.json")
CHARTS_DIR        = os.path.join(PARENT_DIR, "outputs", "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Brand Palette ──────────────────────────────────────────────────────────────
RE_RED    = "#C8102E"   # Royal Enfield signature red
RE_BLACK  = "#1A1A1A"
RE_GOLD   = "#D4A017"
RE_STEEL  = "#607D8B"
RE_LIGHT  = "#F7F7F7"

SENTIMENT_COLORS = {
    "negative": "#E53935",
    "neutral":  "#FFA726",
    "positive": "#43A047",
}

# Topic color map — consistent across all charts
TOPIC_COLORS = {
    "Performance":       "#C8102E",
    "Bugs":              "#E57373",
    "Customer_Support":  "#FF7043",
    "Security":          "#9C27B0",
    "UI/UX":             "#1976D2",
    "Billing":           "#F57F17",
    "Features":          "#43A047",
    "Other":             "#90A4AE",
}

# ── Matplotlib global style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  RE_LIGHT,
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.color":        "#CCCCCC",
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   RE_BLACK,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})


def save(fig: plt.Figure, filename: str):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=RE_LIGHT)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Rating Distribution
# ══════════════════════════════════════════════════════════════════════════════
def chart_rating_distribution(data: dict):
    dist = data["eda"]["rating_analysis"]["distribution"]

    stars   = ["5 ★", "4 ★", "3 ★", "2 ★", "1 ★"]
    counts  = [dist["5_star"]["count"], dist["4_star"]["count"],
               dist["3_star"]["count"], dist["2_star"]["count"],
               dist["1_star"]["count"]]
    pcts    = [dist["5_star"]["percent"], dist["4_star"]["percent"],
               dist["3_star"]["percent"], dist["2_star"]["percent"],
               dist["1_star"]["percent"]]

    bar_colors = ["#43A047", "#8BC34A", "#FFA726", "#FF7043", "#E53935"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(stars, counts, color=bar_colors, height=0.55, edgecolor="white", linewidth=0.5)

    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                f"{count}  ({pct}%)", va="center", fontsize=10, color=RE_BLACK)

    ax.set_xlabel("Number of Reviews", labelpad=8)
    ax.set_title("Rating Distribution\n(371 reviews — Dec 2025 to Apr 2026)", pad=12)
    ax.set_xlim(0, max(counts) * 1.28)

    # Annotation: avg rating
    ax.axvline(0, color=RE_BLACK, lw=0.8)
    avg = data["eda"]["rating_analysis"]["mean_rating"]
    fig.text(0.72, 0.17,
             f"Avg Rating\n{avg:.1f} / 5.0",
             fontsize=12, fontweight="bold", color=RE_RED,
             ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RE_RED, lw=1.5))

    fig.tight_layout()
    save(fig, "01_rating_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Monthly Rating Trend
# ══════════════════════════════════════════════════════════════════════════════
def chart_monthly_trend(df: pd.DataFrame):
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date"])
    df2["month"] = df2["date"].dt.to_period("M")

    monthly = (
        df2.groupby("month")
           .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))
           .reset_index()
    )
    monthly["month_str"] = monthly["month"].astype(str)

    if len(monthly) < 2:
        print("  Skipping monthly trend (insufficient data).")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax2.bar(monthly["month_str"], monthly["review_count"],
            color=RE_STEEL, alpha=0.25, label="Review Count", zorder=1)
    ax1.plot(monthly["month_str"], monthly["avg_rating"],
             marker="o", color=RE_RED, linewidth=2.5, markersize=8,
             label="Avg Rating", zorder=3)

    # Shaded satisfaction zone
    ax1.axhspan(4, 5, alpha=0.06, color="#43A047", label="Target zone (4+)")
    ax1.axhline(y=monthly["avg_rating"].mean(), color=RE_GOLD, linestyle="--",
                linewidth=1.5, alpha=0.8, label=f"Overall avg ({monthly['avg_rating'].mean():.2f})")

    for x, y in zip(monthly["month_str"], monthly["avg_rating"]):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color=RE_BLACK)

    ax1.set_ylim(0.5, 5.5)
    ax1.set_ylabel("Average Star Rating", color=RE_RED)
    ax2.set_ylabel("Review Volume", color=RE_STEEL)
    ax1.set_xlabel("Month")
    ax1.set_title("Monthly Rating Trend & Review Volume", pad=12)
    ax1.tick_params(axis="x", rotation=30)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    fig.tight_layout()
    save(fig, "02_monthly_trend.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — L1 Intent Distribution
# ══════════════════════════════════════════════════════════════════════════════
def chart_l1_distribution(data: dict):
    dist   = data["l1_analysis"]["topic_distribution"]
    topics = list(dist.keys())
    counts = [dist[t]["count"] for t in topics]
    pcts   = [dist[t]["percent"] for t in topics]
    colors = [TOPIC_COLORS.get(t, RE_STEEL) for t in topics]

    # Sort by count descending
    order   = sorted(range(len(counts)), key=lambda i: counts[i])
    topics  = [topics[i] for i in order]
    counts  = [counts[i] for i in order]
    pcts    = [pcts[i] for i in order]
    colors  = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(topics, counts, color=colors, height=0.6, edgecolor="white")

    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{count}  ({pct}%)", va="center", fontsize=10, color=RE_BLACK)

    ax.set_xlabel("Number of Reviews")
    ax.set_title("L1 Intent Distribution\n(What are users talking about?)", pad=12)
    ax.set_xlim(0, max(counts) * 1.30)

    fig.tight_layout()
    save(fig, "03_l1_intent_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — L2 Business Impact (Donut)
# ══════════════════════════════════════════════════════════════════════════════
def chart_l2_impact(data: dict):
    dist    = data["l2_analysis"]["impact_distribution"]
    sev     = data["l2_analysis"]["impact_severity"]
    labels  = list(dist.keys())
    counts  = [dist[l]["count"] for l in labels]
    neg_pct = [sev[l]["negative_percent"] for l in labels]

    IMPACT_COLORS = {
        "User Retention": "#C8102E",
        "Engagement":     "#1976D2",
        "Brand Trust":    "#9C27B0",
        "Revenue":        "#F57F17",
    }
    colors = [IMPACT_COLORS.get(l, RE_STEEL) for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Donut
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, autopct="%1.0f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(10)
    ax1.set_title("L2 Business Impact Distribution\n(Which business area is affected?)", pad=12)

    # Severity bar (negative %)
    sorted_idx = sorted(range(len(neg_pct)), key=lambda i: neg_pct[i], reverse=True)
    sl = [labels[i] for i in sorted_idx]
    sp = [neg_pct[i] for i in sorted_idx]
    sc = [colors[labels.index(l)] for l in sl]

    bars = ax2.barh(sl, sp, color=sc, height=0.5, edgecolor="white")
    for bar, pct in zip(bars, sp):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.0f}%", va="center", fontsize=11, fontweight="bold", color=RE_BLACK)
    ax2.set_xlabel("Negative Review % in Impact Area")
    ax2.set_title("Severity by Business Impact Area", pad=12)
    ax2.set_xlim(0, 105)

    fig.tight_layout()
    save(fig, "04_l2_business_impact.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Sentiment by Topic (Stacked Bar)
# ══════════════════════════════════════════════════════════════════════════════
def chart_sentiment_by_topic(data: dict):
    topic_sent = data["l1_analysis"]["topic_sentiment"]

    topics   = []
    neg_vals = []
    neu_vals = []
    pos_vals = []

    for topic, info in topic_sent.items():
        total = info["count"]
        sd    = info["sentiment_dist"]
        neg   = sd.get("negative", 0) / total * 100
        neu   = sd.get("neutral",  0) / total * 100
        pos   = sd.get("positive", 0) / total * 100
        topics.append(topic)
        neg_vals.append(neg)
        neu_vals.append(neu)
        pos_vals.append(pos)

    # Sort by negative % desc
    order    = sorted(range(len(neg_vals)), key=lambda i: neg_vals[i], reverse=True)
    topics   = [topics[i] for i in order]
    neg_vals = [neg_vals[i] for i in order]
    neu_vals = [neu_vals[i] for i in order]
    pos_vals = [pos_vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(topics))
    w = 0.6

    p1 = ax.bar(x, neg_vals, w, label="Negative", color=SENTIMENT_COLORS["negative"], edgecolor="white")
    p2 = ax.bar(x, neu_vals, w, bottom=neg_vals, label="Neutral",  color=SENTIMENT_COLORS["neutral"],  edgecolor="white")
    p3 = ax.bar(x, pos_vals, w,
                bottom=[n + u for n, u in zip(neg_vals, neu_vals)],
                label="Positive", color=SENTIMENT_COLORS["positive"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=20, ha="right")
    ax.set_ylabel("Percentage of Reviews")
    ax.set_title("Sentiment Breakdown by L1 Topic\n(Ordered by negative %)", pad=12)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # Avg rating labels on top
    topic_sent_vals = {t: topic_sent[t]["avg_rating"] for t in topics}
    for xi, topic in zip(x, topics):
        ax.text(xi, 103, f"★{topic_sent_vals[topic]:.1f}",
                ha="center", fontsize=9, color=RE_BLACK, fontweight="bold")

    fig.tight_layout()
    save(fig, "05_sentiment_by_topic.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Priority Matrix (Bubble)
# ══════════════════════════════════════════════════════════════════════════════
def chart_priority_matrix(data: dict):
    priorities = data["priorities"]

    topics   = [p["topic"] for p in priorities]
    freq     = [p["frequency"] for p in priorities]
    impact   = [p["impact_score"] for p in priorities]
    scores   = [p["priority_score"] for p in priorities]
    labels_p = [p["priority"] for p in priorities]
    colors   = [TOPIC_COLORS.get(t, RE_STEEL) for t in topics]

    bubble_size = [s * 30 for s in scores]

    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(freq, impact, s=bubble_size, c=colors,
                    alpha=0.75, edgecolors="white", linewidths=1.5, zorder=3)

    for i, topic in enumerate(topics):
        ax.annotate(
            f"{topic}\n(P-Score: {scores[i]:.0f})",
            (freq[i], impact[i]),
            textcoords="offset points",
            xytext=(12, 4),
            fontsize=8.5,
            color=RE_BLACK,
        )

    # Quadrant lines
    med_freq   = np.median(freq)
    med_impact = np.median(impact)
    ax.axvline(med_freq,   color="gray", linestyle="--", alpha=0.5, lw=1)
    ax.axhline(med_impact, color="gray", linestyle="--", alpha=0.5, lw=1)

    # Quadrant labels
    ax.text(ax.get_xlim()[0] + 0.3, med_impact + 1, "Low Freq / High Impact",
            fontsize=8, color="gray", style="italic")
    ax.text(med_freq + 0.3, med_impact + 1, "High Freq / High Impact",
            fontsize=8, color=RE_RED, style="italic", fontweight="bold")
    ax.text(ax.get_xlim()[0] + 0.3, ax.get_ylim()[0] + 1, "Monitor",
            fontsize=8, color="gray", style="italic")

    ax.set_xlabel("Frequency % (how often mentioned)")
    ax.set_ylabel("Impact Score (0-100, based on avg rating)")
    ax.set_title("Issue Priority Matrix\n(Bubble size = Priority Score; higher = more urgent)", pad=12)

    # Legend patches
    legend_items = [mpatches.Patch(color=TOPIC_COLORS.get(t, RE_STEEL), label=t) for t in topics]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, ncol=2)

    fig.tight_layout()
    save(fig, "06_priority_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n=== Royal Enfield — Generating Visualizations ===\n")

    # Load data
    with open(ANALYSIS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    df_classified = pd.read_csv(CLASSIFIED_CSV)
    df_raw        = pd.read_csv(RAW_CSV)

    print("Generating charts...")
    chart_rating_distribution(data)
    chart_monthly_trend(df_raw)
    chart_l1_distribution(data)
    chart_l2_impact(data)
    chart_sentiment_by_topic(data)
    chart_priority_matrix(data)

    print(f"\n=== Done. All charts saved to: {CHARTS_DIR} ===\n")


if __name__ == "__main__":
    main()
