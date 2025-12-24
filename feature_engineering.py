import numpy as np
import pandas as pd
import re

from config import ALL_NBA_POINTS, RANKED_AWARDS


def expand_awards(
    df, player_col="player_name_clean", year_col="year", awards_col="Awards"
):
    """
    Expands a compact Awards column like "DPOY-11,DEF1" into one-hot / ranked award columns.
    Each Award_* column contains the numeric rank, or 1 if no rank is specified.

    Parameters:
        df (pd.DataFrame): Input dataframe
        player_col (str): Column containing player names
        year_col (str): Column containing the season year
        awards_col (str): Column containing the raw awards string

    Returns:
        pd.DataFrame: Wide-format dataframe with award columns expanded
    """
    # Keep only rows with awards
    awards_df = df[df[awards_col].notna() & (df[awards_col] != "")].copy()

    # Split awards string on commas
    awards_df["Award_List"] = awards_df[awards_col].str.split(",")

    # Explode list into rows
    awards_df = awards_df.explode("Award_List")
    awards_df["Award_List"] = awards_df["Award_List"].str.strip()

    # Extract award type (e.g., "DPOY" or "DEF") and rank (e.g., 11)
    awards_df["Award_Type"] = awards_df["Award_List"].apply(
        lambda x: (
            re.sub(r"[\d\-]+", "", x).strip().replace(" ", "_")
            if isinstance(x, str)
            else None
        )
    )
    awards_df["Award_Rank"] = awards_df["Award_List"].apply(
        lambda x: (
            int(re.search(r"\d+$", x).group())
            if isinstance(x, str) and re.search(r"\d+$", x)
            else 1
        )
    )

    # Pivot wider
    awards_wide = (
        awards_df[[player_col, year_col, "Award_Type", "Award_Rank"]]
        .pivot_table(
            index=[player_col, year_col],
            columns="Award_Type",
            values="Award_Rank",
            fill_value=0,
            aggfunc="first",
        )
        .reset_index()
    )

    # Rename ambiguous columns in the awards_wide DataFrame
    rename_map = {
        "AS": "all_star",
        "DEF": "all_def",
        "NBA": "all_nba",
    }

    # Apply rename
    awards_wide = awards_wide.rename(columns=rename_map)

    return awards_wide


# Separating player awards & achievements into separate columns
def parse_awards_cell(cell: str) -> dict:
    """
    Parse a compact awards string for a single player-season into a dictionary of award metrics.

    The input string may contain comma-separated tokens representing awards and ranks, such as:
        "MVP-11,CPOY-1,AS,NBA3"

    Recognized patterns:
    - All-NBA teams: "NBA1", "NBA2", "NBA3" (assigns both team number and points)
    - Ranked awards: "<AWARD>-<RANK>", e.g., "MVP-11", "CPOY-1", "DPOY-3", etc.
    - Unranked awards (e.g., "AS" for All-Star) are ignored by this function.

    Parameters
    ----------
    cell : str
        Raw awards string for a player-season.

    Returns
    -------
    res : dict
        Dictionary with the following keys:
            - all_nba_team: int or np.nan (1, 2, 3 for NBA1/2/3, else np.nan)
            - all_nba_points: int (3, 2, 1 for NBA1/2/3, else 0)
            - mvp_rank: int or np.nan
            - cpoy_rank: int or np.nan
            - dpoy_rank: int or np.nan
            - smoy_rank: int or np.nan
            - mip_rank: int or np.nan
            - roy_rank: int or np.nan

    Notes
    -----
    - If the input is not a string or is empty, returns default values.
    - Only recognized awards and patterns are parsed; others are ignored.
    - Award codes and their mapping to output keys are defined in RANKED_AWARDS and ALL_NBA_POINTS.
    """
    # Default resume values for all metrics we care about
    res = {
        "all_nba_team": np.nan,  # 1/2/3 or NaN
        "all_nba_points": 0,  # 3/2/1 or 0
        "mvp_rank": np.nan,
        "cpoy_rank": np.nan,
        "dpoy_rank": np.nan,
        "smoy_rank": np.nan,
        "mip_rank": np.nan,
        "roy_rank": np.nan,
    }

    if not isinstance(cell, str) or cell.strip() == "":
        return res

    for raw_token in cell.split(","):
        token = raw_token.strip()
        if not token:
            continue

        # All-NBA teams (NBA1, NBA2, NBA3)
        if token in ALL_NBA_POINTS:
            res["all_nba_points"] = ALL_NBA_POINTS[token]
            # Last character is the team number: "NBA3" -> 3
            res["all_nba_team"] = int(token[-1])
            continue

        # Ranked awards (MVP-11 etc.)
        if "-" in token:
            award_code, rank_str = token.split("-", 1)
            award_code = award_code.strip().upper()

            try:
                rank = int(rank_str)
            except ValueError:
                # If rank isn't an int, skip
                continue

            col_name = RANKED_AWARDS.get(award_code)
            if col_name is not None:
                res[col_name] = rank

    return res


# Creating a function to assign fantasy points scored in a season
def calculate_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total fantasy points & rank players for each player-season based
    on standard Underdog Fantasy scoring.

    Parameters:
        df (pd.DataFrame): DataFrame containing player stats.

    Returns:
        pd.DataFrame: DataFrame with an additional 'fantasy_points' column.
    """
    df["fantasy_points"] = (
        df["pts"] * 1
        + df["reb"] * 1.2
        + df["ast"] * 1.5
        + df["stl"] * 3
        + df["blk"] * 3
        - df["tov"] * 1
    )

    df["fantasy_points_rank"] = df.groupby("year")["fantasy_points"].rank(
        method="min", ascending=False
    )

    return df


def rolling_trend(feature, years, window=3):
    """
    Compute a rolling career trajectory metric for win shares over a specified window of seasons.

    For each season in a player's career, fits a quadratic regression to the most recent `window`
    seasons (including the current one) and returns a weighted sum of the linear (slope, weight=1.0)
    and quadratic (acceleration, weight=0.5) coefficients.

    Returns
    -------
    np.ndarray
        Array of shape (len(feature),) with the trajectory metric for each window.
    """
    trends = np.zeros(len(feature))
    for i in range(len(feature)):
        start = max(0, i - window + 1)
        y = feature[start : i + 1]
        x = years[start : i + 1]
        if len(y) > 2:
            x_centered = x - np.mean(x)
            coefs = np.polyfit(x_centered, y, 2)  # [quad, linear, intercept]
            trends[i] = coefs[1] + 0.5 * coefs[0]
        else:
            trends[i] = 0.0
    return trends


# Creating features for tracking core statistical career totals & year-over-year deltas
# This also tracks total fantasy points scored in prior and following seasons
def create_metrics(df: pd.DataFrame, CORE_STATS: list) -> pd.DataFrame:
    """
    Add per-player career cumulative totals, 3-year statistical averages, and year-over-year deltas for specified stats.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at minimum 'player_name_clean' and 'year' columns,
        plus any stat columns listed in CORE_STATS.
    CORE_STATS : list
        List of stat column names to compute career sums and year-over-year deltas for.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the following additional columns:
        - career_<stat>: cumulative sum of <stat> for each player up to the current row.
        - <stat>_delta: year-over-year difference of <stat> for each player.

    Notes
    -----
    - Non-numeric or missing stat names from CORE_STATS are ignored.
    - Numeric stat NaNs are filled with 0 before calculations.
    - The function sorts by ['player_name_clean', 'year'] and casts 'year' to int.
    """

    # Keep only numeric stats that actually exist
    stats = [
        c
        for c in CORE_STATS
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    df[stats] = df[stats].fillna(0)
    df["year"] = df["year"].astype(int)

    # Sort & group
    df = df.sort_values(["player_name_clean", "year"])
    g = df.groupby("player_name_clean", sort=False)

    # Career cumulative totals (up to current row)
    career = g[stats].cumsum().add_prefix("career_")

    # Tracking number of unique teams played for in career so far
    # treat '3tm' as 2 teams, otherwise count as 1 (3tm indicates that they played for 2 *new* teams that season)
    # compute weight per row: 2 for '3tm', else 1
    team_weight = (
        df["team"].astype(str).str.lower().apply(lambda t: 2 if "3tm" in t else 1)
    )

    # find first occurrence of each player-team string (so multi-team tokens count once)
    occ = df.groupby(["player_name_clean", "team"], sort=False).cumcount()
    is_first_team_occurrence = (occ == 0).astype(int)

    # cumulative unique-team count weighted by multi-team tokens
    career["career_unique_teams"] = (
        (is_first_team_occurrence * team_weight)
        .groupby(df["player_name_clean"], sort=False)
        .cumsum()
    )

    # 3-year rolling averages for each core stat
    rolling_avgs = (
        g[stats]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .add_prefix("avg3yr_")
    )

    # Process to track a player's recent & long term career arcs via win share trends over the past 3 + 6 seasons
    # Using a blended linear & quadratic fit to capture upwards/downward trajectory + acceleration/deceleration in performance
    trends = g.apply(
        lambda df: pd.DataFrame(
            {
                "fantasy_points_trend_3yr": rolling_trend(
                    df["fantasy_points"].values, df["year"].values, window=3
                ),
                "fantasy_points_trend_6yr": rolling_trend(
                    df["fantasy_points"].values, df["year"].values, window=6
                ),
                "bpm_trend_3yr": rolling_trend(
                    df["bpm"].values, df["year"].values, window=3
                ),
                "bpm_trend_6yr": rolling_trend(
                    df["bpm"].values, df["year"].values, window=6
                ),
            },
            index=df.index,
        ),
        include_groups=False,
    )
    trends.index = trends.index.droplevel(0)  # Remove groupby level to match df index

    # year-over-year deltas for all stats at once
    deltas = g[stats].diff().fillna(0)
    deltas.columns = [f"{c}_delta" for c in deltas.columns]

    # Capturing each player's fantasy points in the prior season
    # Helpful for modeling improvement or decline
    deltas["fantasy_points_prior"] = g["fantasy_points"].shift(1).fillna(0)

    # Capturing each player's fantasy points in the following season
    # This will be used as the target variable for modeling
    deltas["fantasy_points_future"] = g["fantasy_points"].shift(-1).fillna(0)

    # attach new metrics onto original dataframe
    return pd.concat([df, career, rolling_avgs, trends, deltas], axis=1)
