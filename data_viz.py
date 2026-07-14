import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from mizani.palettes import gradient_n_pal
from sklearn.cluster import KMeans

from config import HIROSHIGE_COLORS
from plotnine import (
    ggplot,
    aes,
    labs,
    theme,
    theme_classic,
    element_text,
    element_blank,
    element_rect,
    element_line,
    geom_line,
    geom_point,
    geom_col,
    geom_smooth,
    coord_flip,
    geom_abline,
    geom_segment,
    geom_text,
    annotate,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_reverse,
    scale_color_manual,
    geom_hline,
    geom_histogram,
    geom_boxplot,
)


# Defining custom theme for plotnine visualizations throughout the project
def theme_nba():
    """
    Custom plotnine theme:
    - serif font
    - classic background
    - no panel borders
    - white figure background
    """
    return theme_classic() + theme(
        text=element_text(family="serif"),
        plot_title=element_text(size=18, weight="bold"),
        axis_title=element_text(size=14),
        axis_text=element_text(size=12),
        legend_title=element_text(size=10),
        legend_text=element_text(size=9),
        # No panel border
        panel_border=element_blank(),
        # White backgrounds
        panel_background=element_rect(fill="white", color=None),
        # Faint major gridlines
        panel_grid_major=element_line(color="#e0e0e0", size=0.4),
        panel_grid_minor=element_line(color="#f0f0f0", size=0.3),
        figure_size=(10, 6),
    )


def get_xgb_feature_importance(
    model,
    feature_names=None,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Extract XGBoost feature importance as a tidy DataFrame.

    Handles both:
    - boosters where features are named 'f0', 'f1', ...
    - boosters where features are named with actual column names.
    """
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type=importance_type)

    if not score_dict:
        # No importance info, return empty frame
        return pd.DataFrame(columns=["feature", "importance", "rel_importance"])

    # If feature_names not provided, try to use model.feature_names_in_
    if feature_names is None and hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    elif feature_names is None:
        feature_names = []

    rows = []

    # Check whether keys look like 'f0', 'f1', ...
    f_pattern = re.compile(r"^f\d+$")
    keys_look_like_f = all(bool(f_pattern.match(k)) for k in score_dict.keys())

    if keys_look_like_f and feature_names:
        # Map 'f0', 'f1', ... to actual feature names by index
        for k, v in score_dict.items():
            idx = int(k[1:])  # drop 'f'
            fname = feature_names[idx] if idx < len(feature_names) else k
            rows.append((fname, v))
    else:
        # Keys are already feature names (e.g. 'year', 'career_pts', ...)
        for k, v in score_dict.items():
            rows.append((k, v))

    df_imp = pd.DataFrame(rows, columns=["feature", "importance"])
    df_imp = df_imp.sort_values("importance", ascending=False)

    total = df_imp["importance"].sum()
    df_imp["rel_importance"] = df_imp["importance"] / total if total > 0 else 0.0

    return df_imp


def plot_feature_importance(
    model,
    X_train,
    top_n: int = 20,
    importance_type: str = "gain",
):
    """
    Create a plotnine feature importance plot for an XGBoost model.
    """
    feature_names = list(X_train.columns)

    df_imp = get_xgb_feature_importance(
        model,
        feature_names=feature_names,
        importance_type=importance_type,
    )

    df_top = df_imp.head(top_n).copy()

    p = (
        ggplot(df_top, aes(x="reorder(feature, rel_importance)", y="rel_importance"))
        + geom_col(fill="#6baed6", alpha=0.9, width=0.7)
        + coord_flip()
        + labs(
            title="Feature Importance (XGBoost)",
            y="Relative Importance",
        )
        + theme_nba()
        + theme(
            axis_text=element_text(color="black"),
            axis_title_y=element_blank(),
            figure_size=(10, 8),
        )
    )

    return p


def plot_actual_vs_pred(results, color_palette, top_n=15, x_offset=10, y_offset=40):
    """
    Plot Actual vs Predicted Fantasy Points with top N outliers labeled.

    Args:
        results (pd.DataFrame): DataFrame with prediction results, must include
            'prediction_diff', 'predicted_fantasy_points', 'fantasy_points_future', 'player_name_clean', 'year'.
        color_palette (list): List of color hex codes.
        top_n (int): Number of top outliers to label.
        x_offset (int): Offset for label x position.
        y_offset (int): Offset for label y position.

    Returns:
        plotnine.ggplot: The constructed plot.
    """

    top_outliers = (
        results.reindex(
            results["prediction_diff"].abs().sort_values(ascending=False).index
        )
        .head(top_n)
        .copy()
    )

    top_outliers["year"] = top_outliers["year"] + 1
    top_outliers["label"] = (
        top_outliers["player_name_clean"]
        + " ("
        + top_outliers["year"].astype(str)
        + ")"
    )
    top_outliers["label_x"] = top_outliers["predicted_fantasy_points"] + x_offset
    top_outliers["label_y"] = top_outliers["fantasy_points_future"] + y_offset

    p_actual_vs_pred = (
        ggplot(
            results,
            aes(
                x="predicted_fantasy_points",
                y="fantasy_points_future",
            ),
        )
        + geom_point(
            alpha=0.7,
            size=1.5,
            color=color_palette[0],
        )
        + geom_smooth(
            method="lm",
            se=True,
            level=0.99,
            color="#9ecae1",
            fill="#c6dbef",
            alpha=0.2,
            size=0.5,
        )
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="grey", alpha=0.4)
        + labs(
            title="Actual vs Predicted Fantasy Points",
            x="Predicted Fantasy Points",
            y="Actual Fantasy Points (Future Season)",
        )
        + geom_segment(
            top_outliers,
            aes(
                x="predicted_fantasy_points",
                y="fantasy_points_future",
                xend="label_x",
                yend="label_y",
            ),
            color="darkgrey",
            size=0.3,
            alpha=0.8,
        )
        + geom_text(
            top_outliers,
            aes(
                x="label_x",
                y="label_y",
                label="label",
            ),
            size=7,
            ha="left",
            va="bottom",
            fontstyle="italic",
        )
        + annotate(
            "text",
            x=3000,
            y=500,
            label="Underperformers",
            size=14,
            ha="center",
            va="bottom",
            color="black",
            alpha=0.7,
            fontweight="bold",
        )
        + annotate(
            "text",
            x=500,
            y=3000,
            label="Overperformers",
            size=14,
            ha="center",
            va="top",
            color="black",
            alpha=0.7,
            fontweight="bold",
        )
        + scale_x_continuous(expand=(0.15, 0))
        + scale_y_continuous(limits=(0, None))
        + theme_nba()
        + theme(figure_size=(12, 10))
    )
    return p_actual_vs_pred


def plot_resid_vs_pred(results, color_palette, top_n=25, x_offset=20, y_offset=20):
    """
    Plot residuals (prediction_diff) vs predicted fantasy points, labeling top outliers.

    Args:
        results (pd.DataFrame): DataFrame with prediction results.
        color_palette (list): List of color hex codes.
        top_n (int): Number of top outliers to label.
        x_offset (int): Offset for label x position.
        y_offset (int): Offset for label y position.

    Returns:
        plotnine.ggplot: The constructed plot.
    """

    top_outliers = (
        results.reindex(
            results["prediction_diff"].abs().sort_values(ascending=False).index
        )
        .head(top_n)
        .copy()
    )

    top_outliers["year"] = top_outliers["year"] + 1
    top_outliers["label"] = (
        top_outliers["player_name_clean"]
        + " ("
        + top_outliers["year"].astype(str)
        + ")"
    )
    top_outliers["label_x"] = top_outliers["predicted_fantasy_points"] + x_offset
    top_outliers["label_y"] = top_outliers["prediction_diff"] + (
        y_offset * np.sign(top_outliers["prediction_diff"])
    )

    p_resid_vs_pred = (
        ggplot(
            results,
            aes(
                x="predicted_fantasy_points",
                y="prediction_diff",
            ),
        )
        + labs(
            title="Residuals vs Predicted Fantasy Points",
            x="Predicted Fantasy Points",
            y="Prediction Diff (Predicted - Actual)",
        )
        + geom_point(alpha=0.8, size=1.4, color=color_palette[0])
        + geom_hline(yintercept=0, linetype="dashed", color="grey")
        + geom_segment(
            top_outliers,
            aes(
                x="predicted_fantasy_points",
                y="prediction_diff",
                xend="label_x",
                yend="label_y",
            ),
            color="darkgrey",
            size=0.3,
            alpha=0.8,
        )
        + geom_text(
            top_outliers,
            aes(
                x="label_x",
                y="label_y",
                label="label",
            ),
            size=7,
            ha="left",
            va="bottom",
            fontstyle="italic",
        )
        + annotate(
            "text",
            x=750,
            y=1000,
            label="Underperformers",
            size=14,
            ha="center",
            va="bottom",
            color="black",
            alpha=0.7,
            fontweight="bold",
        )
        + annotate(
            "text",
            x=750,
            y=-1250,
            label="Overperformers",
            size=14,
            ha="center",
            va="top",
            color="black",
            alpha=0.7,
            fontweight="bold",
        )
        + annotate(
            "rect",
            xmin=-np.inf,
            xmax=np.inf,
            ymin=-300,
            ymax=300,
            alpha=0.2,
            fill="lightgrey",
        )
        + scale_y_reverse()
        + theme_nba()
        + theme(figure_size=(12, 10))
    )
    return p_resid_vs_pred


def plot_resid_hist(
    results, color_palette, band=300, binwidth=50, x_annotate=1500, y_annotate=15
):
    """
    Plot histogram of prediction residuals with annotation for % within +/- band.

    Args:
        results (pd.DataFrame): DataFrame with 'prediction_diff' column.
        color_palette (list): List of color hex codes.
        band (int): Absolute value for error band (e.g., 300).
        binwidth (int): Bin width for histogram.
        x_annotate (int): X position for annotation text.
        y_annotate (int): Y position for annotation text.

    Returns:
        plotnine.ggplot: The constructed plot.
    """
    within_n = int((results["prediction_diff"].abs() <= band).mean() * 100)

    p_resid_hist = (
        ggplot(results, aes(x="prediction_diff"))
        + geom_histogram(
            binwidth=binwidth, fill=color_palette[1], alpha=0.8, color="white"
        )
        + labs(
            title="Distribution of Prediction Errors",
            x="Prediction Diff (Predicted - Actual)",
            y="Count",
        )
        + annotate(
            "text",
            x=x_annotate,
            y=y_annotate,
            label=f"{within_n}% of predictions within +/- {band}",
            size=14,
            ha="center",
            va="bottom",
            color="black",
            alpha=0.7,
            fontweight="bold",
        )
        + annotate(
            "rect",
            xmin=-band,
            xmax=band,
            ymin=-float("inf"),
            ymax=float("inf"),
            alpha=0.2,
            fill="lightgrey",
        )
        + theme_nba()
    )
    return p_resid_hist


def plot_recent_seasons(results, color_palette, n_seasons=10):
    """
    Plot boxplots of prediction errors by season for the most recent n_seasons.

    Args:
        results (pd.DataFrame): DataFrame with 'year' and 'prediction_diff' columns.
        color_palette (list): List of color hex codes.
        n_seasons (int): Number of most recent seasons to plot.

    Returns:
        plotnine.ggplot: The constructed plot.
    """
    recent_years = results["year"].max() - n_seasons
    results_recent = results[results["year"] > recent_years].copy()

    p_resid_by_year = (
        ggplot(results_recent, aes(x="factor(year)", y="prediction_diff"))
        + geom_boxplot(fill=color_palette[2], alpha=0.8, outlier_alpha=0.5)
        + labs(
            title="Prediction Errors by Season",
            x="Season",
            y="Prediction Diff (Predicted - Actual)",
        )
        + theme_nba()
    )
    return p_resid_by_year


def plot_decile_calib(results, color_palette):
    """
    Plot calibration curve by predicted decile.

    Args:
        results (pd.DataFrame): DataFrame with 'predicted_fantasy_points' and 'fantasy_points_future'.
        color_palette (list): List of color hex codes.

    Returns:
        plotnine.ggplot: The constructed plot.
    """

    results = results.copy()
    results["pred_decile"] = pd.qcut(
        results["predicted_fantasy_points"],
        q=10,
        labels=False,
    )

    decile_calib = results.groupby("pred_decile", as_index=False).agg(
        mean_pred=("predicted_fantasy_points", "mean"),
        mean_actual=("fantasy_points_future", "mean"),
    )

    decile_diff = decile_calib.assign(
        diff=lambda df: (
            (df["mean_pred"] - df["mean_actual"]) / df["mean_actual"] * 100
        )
    )
    decile_diff["diff"] = (-decile_diff["diff"].round(1).astype(float)).astype(
        str
    ) + "%"

    p_decile_calib = (
        ggplot(decile_calib, aes(x="mean_pred", y="mean_actual"))
        + geom_point(fill=color_palette[2], color=color_palette[2], size=2)
        + geom_line(color=color_palette[2], alpha=1, size=0.8)
        + geom_abline(
            slope=1, intercept=0, linetype="dashed", color="lightgrey", alpha=0.6
        )
        + geom_text(
            decile_diff,
            aes(x="mean_pred", y="mean_actual", label="diff"),
            va="bottom",
            ha="center",
            color="grey",
            fontweight="bold",
            size=10,
            nudge_y=20,
        )
        + labs(
            title="Calibration by Predicted Decile",
            x="Mean Predicted Fantasy Points",
            y="Mean Actual Fantasy Points",
        )
        + theme_nba()
        + theme(figure_size=(10, 5), panel_grid_minor=element_blank())
    )
    return p_decile_calib


def plot_elbow(
    data,
    color_palette,
    value_col="relative_value",
    min_k=3,
    max_k=10,
    random_state=42,
):
    """
    Elbow plot (within-cluster sum of squares vs. K) for choosing the KMeans cluster
    count used in value tiering.

    Args:
        data (pd.DataFrame): DataFrame containing value_col.
        color_palette (list): List of color hex codes.
        value_col (str): Column clustered on (default "relative_value").
        min_k (int): Smallest K to plot. Defaults to 3 because K=1-2 have inflated WSS
            that flattens the rest of the curve.
        max_k (int): Largest K to plot.
        random_state (int): KMeans seed.

    Returns:
        plotnine.ggplot: The constructed elbow plot.
    """
    X = data[[value_col]]
    ks = list(range(min_k, max_k + 1))
    wss = [
        KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X).inertia_
        for k in ks
    ]
    elbow_df = pd.DataFrame({"k": ks, "wss": wss})

    p_elbow = (
        ggplot(elbow_df, aes(x="k", y="wss"))
        + geom_line(color=color_palette[0], size=0.8)
        + geom_point(color=color_palette[0], size=2.5)
        + labs(
            title="Elbow Plot for Optimal K",
            x="Number of Clusters (K)",
            y="Within-Cluster Sum of Squares",
        )
        + theme_nba()
    )
    return p_elbow


def abbreviate_player_names(names):
    """
    Convert cleaned player names ("nikola jokic") to a compact "F. Lastname" display form
    ("N. Jokic"). When two players in the same list share both first initial and last name,
    the first-name prefix is lengthened just enough to disambiguate them (e.g. "Ja. Williams"
    vs "Ju. Williams"). Disambiguation is scoped to the names passed in.

    Args:
        names (list[str]): Cleaned, lowercase player names.

    Returns:
        list[str]: Abbreviated display names, aligned to the input order.
    """
    from collections import defaultdict

    def split_name(name):
        parts = name.split()
        if len(parts) < 2:
            return "", name  # single token -> treat whole as last name
        return parts[0], " ".join(parts[1:])

    firsts, lasts = [], []
    for name in names:
        first, last = split_name(name)
        firsts.append(first)
        lasts.append(last)

    # Group by last name so collisions are only resolved among true clashes
    groups = defaultdict(list)
    for i, last in enumerate(lasts):
        groups[last].append(i)

    labels = [None] * len(names)
    for last, idxs in groups.items():
        last_title = last.title()
        for i in idxs:
            first = firsts[i]
            if not first:
                labels[i] = last_title
                continue
            # Shortest first-name prefix unique among same-last-name players (>=1 letter)
            others = [firsts[j] for j in idxs if j != i]
            k = 1
            while k < len(first) and any(o[:k] == first[:k] for o in others):
                k += 1
            labels[i] = f"{first[:k].capitalize()}. {last_title}"
    return labels


def plot_pred_vs_proj_dumbbell(projection_df, color_palette, position_group="G", top_n=30):
    """
    Dumbbell chart comparing the model prediction against the FantasyPros projection for the
    top_n players in a position group (ranked by the blended final_projection). Each player is
    a row with two dots - model vs expert - joined by a connector, so the gap between the two
    reads at a glance. A black diamond marks the blended final_projection between them.

    Args:
        projection_df (pd.DataFrame): Blended frame (e.g. blended_df) with 'player_name_clean',
            'position_group', 'predicted_fantasy_points', 'projected_fantasy_points', and
            'final_projection'.
        color_palette (list): List of color hex codes (model uses index 0, FantasyPros index 2).
        position_group (str): Position group to plot ("G", "W", or "B").
        top_n (int): Number of players to show, taken from the top of the final_projection ranking.

    Returns:
        plotnine.ggplot: The constructed dumbbell plot.
    """
    # Rank the position group by the blended projection and keep the top N players
    ranked = (
        projection_df[
            (projection_df["position_group"] == position_group)
            & projection_df["predicted_fantasy_points"].notna()
            & projection_df["projected_fantasy_points"].notna()
        ]
        .sort_values("final_projection", ascending=False)
        .head(top_n)
        .copy()
    )

    # Assign explicit numeric y positions (highest projection on top). A numeric y axis lets the
    # dashed separators (geom_hline) coexist with the dots/connectors, which plotnine forbids on a
    # discrete axis; we relabel the axis with player names via scale_y_continuous below.
    n_players = len(ranked)
    ranked["y_pos"] = np.arange(n_players, 0, -1)  # first (best) row -> top
    y_breaks = ranked["y_pos"].tolist()
    y_labels = abbreviate_player_names(ranked["player_name_clean"].tolist())

    # Long form drives the two colored dots; the wide ranked frame anchors the connector endpoints
    points_long = ranked.melt(
        id_vars="y_pos",
        value_vars=["predicted_fantasy_points", "projected_fantasy_points"],
        var_name="source",
        value_name="points",
    )
    source_labels = {
        "predicted_fantasy_points": "Model",
        "projected_fantasy_points": "FantasyPros",
    }
    points_long["source"] = pd.Categorical(
        points_long["source"].map(source_labels),
        categories=["Model", "FantasyPros"],
        ordered=True,
    )

    # Dashed horizontal separators every 5 players, counted from the top of the ranking; drawn at
    # the .5 boundary between each block of 5.
    separators = [n_players - k + 0.5 for k in np.arange(5, n_players, 5)]

    # x-axis scaled to NBA season fantasy-point totals (~1,000-4,600), not the R 0-1000. Start at
    # the low end of the data (floored to 250) rather than 0 so the names don't sit far left of the
    # dumbbells; major gridlines stay on clean 500s that fall within range.
    both_cols = ranked[["predicted_fantasy_points", "projected_fantasy_points"]]
    x_lower = int(np.floor(both_cols.min().min() / 250.0) * 250)
    x_upper = int(np.ceil(both_cols.max().max() / 250.0) * 250)
    major_breaks = [b for b in range(0, x_upper + 1, 500) if b >= x_lower]
    minor_breaks = [b for b in range(0, x_upper + 1, 250) if b >= x_lower]

    p_dumbbell = (
        ggplot()
        + geom_hline(
            yintercept=separators,
            linetype="dashed",
            color="#00000F",
            size=0.4,
            alpha=0.4,
        )
        # Connector between the model and expert estimate for each player
        + geom_segment(
            ranked,
            aes(
                y="y_pos",
                yend="y_pos",
                x="predicted_fantasy_points",
                xend="projected_fantasy_points",
            ),
            color="#C2C2C2",
            size=1.75,
        )
        # The two estimate dots, colored by source
        + geom_point(
            points_long,
            aes(y="y_pos", x="points", color="source"),
            size=5,
        )
        # Black diamond marking the blended final_projection (plain marker, no legend entry)
        + geom_point(
            ranked,
            aes(y="y_pos", x="final_projection"),
            shape="D",
            color="#000000",
            size=1,
        )
        + scale_color_manual(
            values={"Model": color_palette[0], "FantasyPros": color_palette[2]},
            labels=["Model Prediction", "Expert Projection"],
        )
        + scale_x_continuous(
            breaks=major_breaks, minor_breaks=minor_breaks, limits=(x_lower, x_upper)
        )
        + scale_y_continuous(breaks=y_breaks, labels=y_labels)
        + labs(
            title=f"Model vs Expert — Top {top_n} {position_group}",
            x="Projected Fantasy Points",
            y=None,
        )
        + theme_nba()
        + theme(
            figure_size=(16, 10),
            dpi=250,
            plot_title=element_text(size=24, weight="bold"),
            axis_title=element_text(size=18),
            axis_text=element_text(size=16),
            legend_text=element_text(size=16),
            legend_position="top",
            legend_title=element_blank(),
            axis_title_y=element_blank(),
            panel_grid_major_y=element_blank(),
            panel_grid_minor_y=element_blank(),
        )
    )
    return p_dumbbell


# =============================================================================
# Interactive tier / rank diagnostics (Plotly)
# =============================================================================
# These two run as the final step, after players carry value tiers. They are the only Plotly
# plots in the repo (every other plot is plotnine): with ~10-14 tier colors and a label per
# point, static versions are unreadable, so hover-to-see-player-name is what makes them useful.


def tier_palette(n):
    """
    Interpolate n hex colors across the MetBrewer "Hiroshige" ramp (config.HIROSHIGE_COLORS).

    Tier 1 (highest value) maps to the first (warm) color, the last tier to the final (cool)
    color, matching the R port. Uses mizani's gradient palette (ships with plotnine).

    Args:
        n (int): Number of tiers / colors to produce.

    Returns:
        list[str]: n interpolated hex color strings.
    """
    if n < 1:
        return []
    return gradient_n_pal(HIROSHIGE_COLORS)(list(np.linspace(0, 1, n)))


# Friendly position-group names for plot titles
POSITION_GROUP_NAMES = {"G": "Guard", "W": "Wing", "B": "Big"}


def _plotly_layout(title, x_title, y_title=None):
    """Shared Plotly layout echoing theme_nba() (serif, white background, sized fonts, no legend)."""
    return dict(
        title=dict(text=title, font=dict(size=22, color="#262626")),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=16)), tickfont=dict(size=15)
        ),
        yaxis=dict(
            title=dict(text=y_title or "", font=dict(size=16)), tickfont=dict(size=15)
        ),
        showlegend=False,
        template="plotly_white",
        font=dict(family="serif", size=14, color="#262626"),
        width=1000,
        height=700,
    )


def _style_hover(fig, color_map):
    """Give each tier trace a white tooltip card with tier-colored border + text."""
    for tr in fig.data:
        hexc = color_map.get(tr.name)
        tr.update(
            hovertemplate="%{customdata[0]}<extra></extra>",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor=hexc,
                font=dict(color=hexc, family="serif", size=13),
            ),
        )


def plot_positional_tiers(player_df, position_group="G", top_n=None):
    """
    Interactive tier structure for one position group: each player is plotted at their
    positional rank (x) against blended final_projection (y), colored by position_value_tier,
    so tier breaks read as bands down the projection curve. The highest-projected player in each
    tier is labeled to anchor where the tier begins; every other name surfaces on hover.

    Args:
        player_df (pd.DataFrame): Tiered frame (e.g. value_df) with 'position_group',
            'position_rank', 'final_projection', 'position_value_tier', 'player_name_clean', 'pos'.
        position_group (str): Position group to plot ("G", "W", or "B").
        top_n (int, optional): If given, keep only the top_n by positional rank.

    Returns:
        plotly.graph_objects.Figure: The interactive tier plot.
    """
    pos_df = (
        player_df[player_df["position_group"] == position_group]
        .sort_values("position_rank")
        .copy()
    )
    if top_n is not None:
        pos_df = pos_df.head(top_n)

    # Tiers as ordered string categories, each mapped to a Hiroshige color (tier 1 = warm)
    tiers = sorted(pos_df["position_value_tier"].unique())
    color_map = {str(t): c for t, c in zip(tiers, tier_palette(len(tiers)))}
    pos_df["tier"] = pos_df["position_value_tier"].astype(str)
    pos_df["player_label"] = pos_df["player_name_clean"].str.title()
    pos_df["hover_text"] = (
        "<b>" + pos_df["player_label"] + " - " + pos_df["pos"].astype(str) + "</b><br>"
        + "Pos rank " + pos_df["position_rank"].astype(str) + " · Tier " + pos_df["tier"] + "<br>"
        + "Projection " + pos_df["final_projection"].round(0).astype(int).astype(str)
    )

    fig = px.scatter(
        pos_df,
        x="position_rank",
        y="final_projection",
        color="tier",
        category_orders={"tier": [str(t) for t in tiers]},
        color_discrete_map=color_map,
        custom_data=["hover_text"],
    )
    # Larger markers with a faint white outline so points read cleanly (static hover highlight)
    fig.update_traces(
        marker=dict(size=11, opacity=0.9, line=dict(width=0.6, color="white"))
    )
    _style_hover(fig, color_map)

    # Label the highest-projected player in each tier (tier leader), colored to its tier
    leaders = pos_df.loc[
        pos_df.groupby("position_value_tier")["final_projection"].idxmax()
    ]
    for _, row in leaders.iterrows():
        hexc = color_map[row["tier"]]
        fig.add_annotation(
            x=row["position_rank"],
            y=row["final_projection"],
            text=row["player_label"],
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.5,
            arrowcolor=hexc,
            ax=20,
            ay=-10,
            xanchor="left",
            font=dict(size=12, color=hexc),
        )

    fig.update_layout(
        **_plotly_layout(
            title=f"{POSITION_GROUP_NAMES.get(position_group, position_group)} Projected Value Tiers",
            x_title="Positional Ranking",
            y_title=None,
        )
    )
    return fig


def plot_model_vs_expert(
    player_df,
    position_group=None,
    n_label=10,
    final_pred=True,
    gate_labels_to_replacement=True,
):
    """
    Interactive scatter of each player's model positional rank (x) against their FantasyPros
    expert positional rank (y), colored by tier. Ranks (1 = best) are used instead of raw
    projections so huge point totals don't stretch the plot. Both axes are reversed so the best
    players sit top-right; a dashed line marks agreement. Points toward the top-left are ranked
    better by the experts (expert favored); points toward the bottom-right are the model's
    relative sleepers (model favored). The n_label biggest rank disagreements are named.

    Args:
        player_df (pd.DataFrame): Tiered frame (e.g. value_df) with 'position_group',
            'predicted_fantasy_points', 'projected_fantasy_points', 'final_projection',
            'player_value_tier', 'position_value_tier', 'player_name_clean', 'pos', and
            (optionally) 'replacement_value'.
        position_group (str, optional): Focus on one group (colors then use position_value_tier);
            None plots the whole board (colors use the overall player_value_tier).
        n_label (int): Number of largest model-vs-expert disagreements to label.
        final_pred (bool): Model axis ranks by blended final_projection when True, else by the
            pure predicted_fantasy_points (pre-blend).
        gate_labels_to_replacement (bool): When True and 'replacement_value' is present, restrict
            the labeled outliers to players at/above their position's replacement level.

    Returns:
        plotly.graph_objects.Figure: The interactive model-vs-expert rank plot.
    """
    model_label = "Final" if final_pred else "Model"

    df = player_df.copy()
    if position_group is not None:
        df = df[df["position_group"] == position_group]
    df = df[
        df["predicted_fantasy_points"].notna() & df["projected_fantasy_points"].notna()
    ].copy()

    # Color by positional tier when focused, otherwise the overall tier
    tier_col = "player_value_tier" if position_group is None else "position_value_tier"
    tiers = sorted(df[tier_col].unique())
    color_map = {str(t): c for t, c in zip(tiers, tier_palette(len(tiers)))}

    # Rank within position group (1 = best) by each estimate, then measure the gap between them
    model_col = "final_projection" if final_pred else "predicted_fantasy_points"
    grouped = df.groupby("position_group")
    df["model_rank"] = grouped[model_col].rank(method="dense", ascending=False).astype(int)
    df["expert_rank"] = (
        grouped["projected_fantasy_points"].rank(method="dense", ascending=False).astype(int)
    )
    df["rank_diff"] = df["model_rank"] - df["expert_rank"]  # + = expert ranks better (lower)
    df["rank_gap"] = df["rank_diff"].abs()
    df["tier"] = df[tier_col].astype(str)
    df["player_label"] = df["player_name_clean"].str.title()
    favored = np.where(df["rank_diff"] >= 0, "Expert +", "Model +")
    df["gap_note"] = favored + df["rank_gap"].astype(str) + " spots"
    df["hover_text"] = (
        "<b>" + df["player_label"] + " — " + df["pos"].astype(str) + "</b><br>"
        + f"{model_label} rank " + df["model_rank"].astype(str)
        + " · Expert rank " + df["expert_rank"].astype(str) + "<br>"
        + df["gap_note"]
    )

    fig = px.scatter(
        df,
        x="model_rank",
        y="expert_rank",
        color="tier",
        category_orders={"tier": [str(t) for t in tiers]},
        color_discrete_map=color_map,
        custom_data=["hover_text"],
    )
    # Larger markers with a faint white outline so points read cleanly (static hover highlight)
    fig.update_traces(
        marker=dict(size=10, opacity=0.9, line=dict(width=0.6, color="white"))
    )
    _style_hover(fig, color_map)

    # Agreement line: model rank == expert rank
    max_rank = int(max(df["model_rank"].max(), df["expert_rank"].max()))
    fig.add_shape(
        type="line",
        x0=1,
        y0=1,
        x1=max_rank,
        y1=max_rank,
        line=dict(color="#999999", width=1, dash="dash"),
    )

    # Label the biggest disagreements, gated to draftable players when replacement is available
    label_pool = df
    if gate_labels_to_replacement and "replacement_value" in df.columns:
        label_pool = df[df["final_projection"] >= df["replacement_value"]]
    for _, row in label_pool.nlargest(n_label, "rank_gap").iterrows():
        hexc = color_map[row["tier"]]
        fig.add_annotation(
            x=row["model_rank"],
            y=row["expert_rank"],
            text=row["player_label"],
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.5,
            arrowcolor=hexc,
            ax=12,
            ay=-12,
            font=dict(size=12, color=hexc),
        )

    title = "Model vs Expert Positional Rank" + (
        f" — {POSITION_GROUP_NAMES.get(position_group, position_group)}"
        if position_group
        else ""
    )
    fig.update_layout(
        **_plotly_layout(
            title=title,
            x_title=f"{model_label} Positional Rank",
            y_title="Expert Positional Rank",
        )
    )
    # Reverse both axes so rank 1 (best) sits top-right; major gridlines every 10 ranks
    fig.update_xaxes(autorange="reversed", dtick=10)
    fig.update_yaxes(autorange="reversed", dtick=10)
    # Corner cues (paper coords): experts better -> top-left, model better -> bottom-right
    fig.add_annotation(
        xref="paper", yref="paper", x=0.1, y=0.9, text="Expert Favored",
        showarrow=False, font=dict(size=20, color="#595959"), opacity=0.6,
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.9, y=0.1, text="Model Favored",
        showarrow=False, font=dict(size=20, color="#595959"), opacity=0.6,
    )
    return fig
