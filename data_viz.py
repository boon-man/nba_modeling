import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
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
    y_labels = ranked["player_name_clean"].tolist()

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
            alpha=0.25,
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
            size=1,
        )
        # The two estimate dots, colored by source
        + geom_point(
            points_long,
            aes(y="y_pos", x="points", color="source"),
            size=3,
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
            labels=["Model Prediction", "FantasyPros Projection"],
        )
        + scale_x_continuous(
            breaks=major_breaks, minor_breaks=minor_breaks, limits=(x_lower, x_upper)
        )
        + scale_y_continuous(breaks=y_breaks, labels=y_labels)
        + labs(
            title=f"Model vs FantasyPros — Top {top_n} {position_group}",
            x="Projected Fantasy Points",
            y=None,
        )
        + theme_nba()
        + theme(
            figure_size=(16, 10),
            dpi=200,
            legend_position="top",
            legend_title=element_blank(),
            axis_title_y=element_blank(),
            panel_grid_major_y=element_blank(),
            panel_grid_minor_y=element_blank(),
        )
    )
    return p_dumbbell
