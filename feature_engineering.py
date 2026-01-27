import numpy as np
import pandas as pd
import re

from config import ALL_NBA_POINTS, RANKED_AWARDS


# Function to bin colleges based on number of NBA players produced
def bin_college(
    df: pd.DataFrame,
    college_col: str = "college",
    player_col: str = "player_name_clean",
    pipeline_threshold: int = 10,
    notable_threshold: int = 3,
) -> pd.DataFrame:
    """
    Bin colleges based on empirical NBA player counts from the data.

    Categories:
    - 'no_college': International/no college players
    - 'nba_pipeline': Schools with > pipeline_threshold unique NBA players
    - 'notable': Schools with > notable_threshold (but <= pipeline_threshold) players
    - 'other': Schools with <= notable_threshold players

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with college and player columns.
    college_col : str
        Name of the college column.
    player_col : str
        Name of the player identifier column.
    pipeline_threshold : int, default 10
        Minimum unique players to qualify as 'nba_pipeline'.
    notable_threshold : int, default 3
        Minimum unique players to qualify as 'notable'.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'college_bin' column.
    """
    out = df.copy()

    # Count unique players per college
    college_counts = (
        out.groupby(college_col, observed=True)[player_col]
        .nunique()
        .reset_index(name="player_count")
    )

    # Create bin mapping
    def assign_bin(row):
        college = row[college_col]
        count = row["player_count"]

        # Handle no college / international
        if pd.isna(college) or str(college).strip().upper() in (
            "NO COLLEGE",
            "NONE",
            "",
        ):
            return "no_college"

        if count > pipeline_threshold:
            return "nba_pipeline"
        elif count > notable_threshold:
            return "notable"
        else:
            return "other"

    college_counts["college_bin"] = college_counts.apply(assign_bin, axis=1)

    # Merge back to original dataframe
    bin_map = college_counts.set_index(college_col)["college_bin"].to_dict()

    # Handle no college explicitly
    out["college_bin"] = out[college_col].map(bin_map)
    out.loc[
        out[college_col].isna()
        | (out[college_col].astype(str).str.upper().isin(["NO COLLEGE", "NONE", ""])),
        "college_bin",
    ] = "no_college"

    out["college_bin"] = out["college_bin"].astype("category")

    return out


def assign_college_bin(
    df: pd.DataFrame,
    college_col: str = "college",
    player_col: str = "player_name_clean",
    pipeline_threshold: int = 10,
    notable_threshold: int = 3,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Bin colleges into categories and optionally drop the original college column.

    This is a convenience wrapper around `bin_college()` that also handles
    dropping the original college column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with college and player columns.
    college_col : str, default "college"
        Name of the college column.
    player_col : str, default "player_name_clean"
        Name of the player identifier column.
    pipeline_threshold : int, default 10
        Minimum unique players to qualify as 'nba_pipeline'.
    notable_threshold : int, default 3
        Minimum unique players to qualify as 'notable'.
    drop_original : bool, default True
        Whether to drop the original college column after binning.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'college_bin' column added (and original dropped if specified).
    """
    out = bin_college(
        df,
        college_col=college_col,
        player_col=player_col,
        pipeline_threshold=pipeline_threshold,
        notable_threshold=notable_threshold,
    )

    if drop_original and college_col in out.columns:
        out = out.drop(columns=[college_col])

    return out


def process_awards(
    df: pd.DataFrame,
    awards_col: str = "awards",
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Parse the awards column and expand into individual award metric columns.

    Uses `parse_awards_cell()` to extract structured award data from compact
    award strings (e.g., "MVP-11,NBA3,AS") and adds columns for:
    - all_nba_team, all_nba_points
    - mvp_rank, cpoy_rank, dpoy_rank, smoy_rank, mip_rank, roy_rank

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with an awards column.
    awards_col : str, default "awards"
        Name of the column containing compact award strings.
    drop_original : bool, default True
        Whether to drop the original awards column after parsing.

    Returns
    -------
    pd.DataFrame
        DataFrame with award columns added (and original dropped if specified).

    Example
    -------
    >>> combined_df = process_awards(combined_df)
    """
    out = df.copy()

    # Parse each row's awards string into a dict
    awards_parsed = out[awards_col].apply(parse_awards_cell)

    # Convert list of dicts to DataFrame
    awards_df = pd.DataFrame(list(awards_parsed), index=out.index)

    # Concatenate with original dataframe
    out = pd.concat([out, awards_df], axis=1)

    # Drop original awards column if requested
    if drop_original and awards_col in out.columns:
        out = out.drop(columns=[awards_col])

    return out


# Function to add era feature based on season year
def add_era_feature(
    df: pd.DataFrame,
    *,
    year_col: str = "year",
    era_col: str = "era",
) -> pd.DataFrame:
    """
    Add an era categorical feature based on season year using explicit sequential comparisons.

    Buckets (editable):
      - year <= 2004: early_00s
      - year <= 2009: late_00s
      - year <= 2014: early_10s
      - year <= 2019: late_10s
      - year <= 2024: early_20s
      - else:         late_20s_plus
    """
    out = df.copy()
    y = out[year_col].astype(int)

    era = np.where(
        y <= 2004,
        "early_00s",
        np.where(
            y <= 2009,
            "late_00s",
            np.where(
                y <= 2014,
                "early_10s",
                np.where(
                    y <= 2019, "late_10s", np.where(y <= 2024, "early_20s", "late_20s")
                ),
            ),
        ),
    )

    out[era_col] = pd.Categorical(
        era,
        categories=[
            "early_00s",
            "late_00s",
            "early_10s",
            "late_10s",
            "early_20s",
            "late_20s_plus",
        ],
        ordered=True,
    )

    return out


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
    # Default result values for all metrics we care about
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
    # Ranking players by fantasy points within each season (percentile rank)
    df["fantasy_points_pct"] = (
        df.groupby("year")["fantasy_points"]
        .rank(pct=True, ascending=True)
        .astype(float)
    )

    # Calculating fantasy points per minute played
    df["fpts_per_min"] = df["fantasy_points"] / df["mp"].replace(0, np.nan)

    return df


# Function to track player's "productivity scores" how productive are they relative to their age?
def calculate_productivity_score(
    df: pd.DataFrame,
    fantasy_points_col: str = "fantasy_points",
    age_col: str = "Age",
    output_col: str = "productivity_score",
) -> pd.DataFrame:
    """
    Calculate productivity score features based on player age and fantasy points.

    Creates three productivity metrics:
    1. Adjusted productivity: fantasy_points / (age^2)
    2. Productivity trend: change from previous season
    3. 3-year rolling average productivity

    This helps identify players performing above/below their career arc.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing player season data.
    fantasy_points_col : str, default 'fantasy_points'
        Column name containing fantasy points.
    age_col : str, default 'Age'
        Column name containing player age.
    output_col : str, default 'productivity_score'
        Base name for output columns (suffixes added for trend and 3yr).

    Returns
    -------
    pd.DataFrame
        Input dataframe with three new columns:
        - 'productivity_score': fantasy_points / (age^2)
        - 'productivity_trend': change from previous season
        - 'productivity_3yr': 3-year rolling average

    Notes
    -----
    Productivity score higher values indicate better efficiency relative to age.
    Trend can be positive (improving) or negative (declining).
    """
    df = df.copy()

    # Sort by player and year to ensure correct grouped operations
    df = df.sort_values(by=["player_id", "year"]).reset_index(drop=True)

    # Calculate age squared
    df["age_squared"] = df[age_col] ** 2

    # Create adjusted productivity score (points / age^2)
    df[output_col] = df[fantasy_points_col] / df["age_squared"]

    # Calculate productivity trend (change from previous season)
    # Group by player ID to ensure we're comparing within-player seasons
    df["productivity_trend"] = df.groupby("player_id")[output_col].diff()

    # Calculate 3-year rolling average productivity
    df["productivity_3yr"] = df.groupby("player_id")[output_col].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # Drop the temporary age_squared column
    df = df.drop(columns=["age_squared"])

    return df


# Function to add "years since peak" feature to dataset
def calculate_years_since_peak(
    df: pd.DataFrame,
    player_col: str = "player_name_clean",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Add peak-timing features based on each player's peak fantasy points season.

    Adds:
      - years_before_peak: max(peak_year - year, 0)
      - years_after_peak:  max(year - peak_year, 0)
      - pct_of_peak_year: fantasy_points / max_fantasy_points_for_player

    Notes
    -----
    - Peak season is defined as the season with maximum `fantasy_points` for each player.
    - This avoids a single signed feature (years_since_peak) that goes negative for pre-peak players.
    """
    out = df.copy()
    out[year_col] = out[year_col].astype(int)

    if "fantasy_points" not in out.columns:
        raise ValueError(
            "calculate_years_since_peak requires a 'fantasy_points' column on df."
        )

    # Peak year per player (row index of max fantasy_points within player)
    peak_idx = out.groupby(player_col, sort=False)["fantasy_points"].idxmax()
    peak_year_by_player = out.loc[peak_idx, [player_col, year_col]].set_index(
        player_col
    )[year_col]

    peak_year = out[player_col].map(peak_year_by_player).astype("Int64")

    out["years_before_peak"] = (peak_year - out[year_col]).clip(lower=0).astype(int)
    out["years_after_peak"] = (out[year_col] - peak_year).clip(lower=0).astype(int)

    out["pct_of_peak_year"] = (
        out["fantasy_points"]
        / out.groupby(player_col)["fantasy_points"].transform("max")
    ).astype(float)

    return out


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
    feature = np.asarray(feature, dtype=float)
    years = np.asarray(years, dtype=float)

    trends = np.full(len(feature), np.nan, dtype=float)
    # Require full history for short windows; allow >=3 seasons for longer windows (e.g., 6yr)
    min_seasons = window if window <= 3 else 3

    for i in range(len(feature)):
        start = max(0, i - window + 1)
        y = feature[start : i + 1]
        x = years[start : i + 1]

        if len(y) < min_seasons:
            continue

        if np.isnan(y).any() or np.isnan(x).any():
            continue

        # Fit quadratic trend (needs >= 3 points; enforced via min_points)
        x_centered = x - np.mean(x)
        coefs = np.polyfit(x_centered, y, 2)  # [quad, linear, intercept]
        trends[i] = coefs[1] + 0.5 * coefs[0]
    return trends


# Creating features for tracking core statistical career totals & year-over-year deltas
# This also tracks total fantasy points scored in prior and following seasons
def create_agg_metrics(
    df: pd.DataFrame, CORE_STATS: list, CAREER_STATS: list
) -> pd.DataFrame:
    """
    Add per-player career cumulative totals, 3-year statistical averages, and year-over-year deltas for specified stats.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at minimum 'player_name_clean' and 'year' columns,
        plus any stat columns listed in CORE_STATS.
    CORE_STATS : list
        List of stat column names to compute career sums and year-over-year deltas for.
    CAREER_STATS : list
        List of stat column names to compute career cumulative totals for.

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
    # Repeat for career stats
    career_stats = [
        c
        for c in CAREER_STATS
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    df["year"] = df["year"].astype(int)

    # Sort & group
    df = df.sort_values(["player_name_clean", "year"])
    g = df.groupby("player_name_clean", sort=False)

    # Seasons played prior (0 for rookie season row)
    seasons_prior = g.cumcount()

    # Window coverage counts (how many seasons included in the avg3yr window)
    seasons_in_3 = (
        g["year"]
        .rolling(window=3, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
        .astype(int)
    )
    seasons_in_6 = (
        g["year"]
        .rolling(window=6, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
        .astype(int)
    )

    # Career cumulative totals (up to current row)
    career = g[career_stats].cumsum().add_prefix("career_")

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

    # Player tiering: VORP-based role bucket (3yr avg)
    def _vorp_tier(vorp: float) -> str:
        if pd.isna(vorp):
            return "unknown"
        if vorp >= 4.0:
            return "superstar"
        if vorp >= 2.5:
            return "star"
        if vorp >= 1:
            return "starter"
        if vorp >= 0:
            return "rotation"
        return "bench"

    bpm_tier = rolling_avgs["avg3yr_bpm"].apply(_vorp_tier)
    rolling_avgs["player_tier"] = bpm_tier.astype("category")

    # Calculating total minutes played and games played in the past 3 years
    sum_cols = [
        c
        for c in ["mp", "gp", "fantasy_points"]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    # flagging missing accumulation stats as 0 (missing games played corresponds to 0 games played)
    df[sum_cols] = df[sum_cols].fillna(0)
    rolling_sums_3yr = (
        g[sum_cols]
        .rolling(window=3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .add_prefix("sum3yr_")
    )
    rolling_sums_6yr = (
        g[sum_cols]
        .rolling(window=6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .add_prefix("sum6yr_")
    )

    # Derived rate features (minutes-weighted)
    rolling_sums_3yr["fpts_per_min_3yr"] = rolling_sums_3yr[
        "sum3yr_fantasy_points"
    ] / rolling_sums_3yr["sum3yr_mp"].replace(0, np.nan)
    rolling_sums_6yr["fpts_per_min_6yr"] = rolling_sums_6yr[
        "sum6yr_fantasy_points"
    ] / rolling_sums_6yr["sum6yr_mp"].replace(0, np.nan)

    rolling_sums_3yr["min_pg_equiv_3yr"] = rolling_sums_3yr[
        "sum3yr_mp"
    ] / rolling_sums_3yr["sum3yr_gp"].replace(0, np.nan)
    rolling_sums_6yr["min_pg_equiv_6yr"] = rolling_sums_6yr[
        "sum6yr_mp"
    ] / rolling_sums_6yr["sum6yr_gp"].replace(0, np.nan)

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

    # year-over-year deltas select core stats
    deltas = g[["min_pg", "bpm", "vorp", "usg_pct", "fantasy_points"]].diff()
    deltas.columns = [f"{c}_delta" for c in deltas.columns]

    # Capturing each player's fantasy points in the prior season
    # Helpful for modeling improvement or decline
    deltas["fantasy_points_prior"] = g["fantasy_points"].shift(1)

    # Capturing each player's fantasy points in the following season
    # This will be used as the target variable for modeling
    deltas["fantasy_points_future"] = g["fantasy_points"].shift(-1)

    out = pd.concat(
        [df, career, rolling_avgs, rolling_sums_3yr, rolling_sums_6yr, trends, deltas],
        axis=1,
    )

    out["seasons_prior"] = seasons_prior.values
    out["seasons_in_3yr_window"] = seasons_in_3.values
    out["seasons_in_6yr_window"] = seasons_in_6.values

    # attach new metrics onto original dataframe
    return out


# =============================================================================
# Organized Feature Engineering Functions
# =============================================================================


def add_age_features(
    df: pd.DataFrame,
    start_year: int,
) -> pd.DataFrame:
    """
    Add age-related features to the dataframe.

    Features created:
    - years_in_league: seasons since rookie year
    - age_bucket: categorical age groupings (rookie/young/prime/veteran/late)
    - age_tier: combination of age_bucket and player_tier
    - age_x_bpm: age × BPM interaction
    - age_x_min_pg: age × minutes per game interaction
    - min_pg_headroom: upside proxy (36 - min_pg)
    - incomplete_career_history: flag for players missing full career data
    - pk_adj: draft pick adjusted for career length (decays after 4 years)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'age', 'year', 'rookie_year', 'bpm', 'min_pg',
        'player_tier', 'college_bin', 'pk' columns.
    start_year : int
        First year of data collection (used for incomplete history detection).

    Returns
    -------
    pd.DataFrame
        DataFrame with age-related features added.
    """
    out = df.copy()

    # Years in league
    out["years_in_league"] = out["year"] - out["rookie_year"]

    # Age bucket (categorical)
    out["age_bucket"] = pd.cut(
        out["age"],
        bins=[18, 22, 26, 30, 34, 45],
        labels=["rookie", "young", "prime", "veteran", "late"],
    )

    # Age tier (age bucket × player tier)
    out["age_tier"] = (
        out["age_bucket"].astype("string") + "_" + out["player_tier"].astype("string")
    ).astype("category")

    # Age-curve interactions
    out["age_x_bpm"] = out["age"] * out["bpm"]
    out["age_x_min_pg"] = out["age"] * out["min_pg"]

    # Minutes headroom (upside proxy for young players)
    out["min_pg_headroom"] = 36 - out["min_pg"]

    # Incomplete career history flag
    out["incomplete_career_history"] = (
        (out["rookie_year"] < start_year)
        | (
            (out["college_bin"] == "no_college")
            & (out["age"] >= 25)
            & (out["year"] == (start_year + 1))
        )
    ).astype(int)
    out["incomplete_career_history"] = out.groupby("player_name_clean")[
        "incomplete_career_history"
    ].transform("max")

    # Draft capital decay (pick irrelevant after 4 years)
    out["pk_adj"] = np.where(
        out["years_in_league"] <= 4,
        out["pk"],
        np.nan,
    )

    return out


def add_availability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add availability and health-related features to the dataframe.

    Features created:
    - eligible_games_played: career games / (82 × years in league)
    - avg3yr_eligible_games_played: 3-year rolling average of above
    - injury_flag: indicator for seasons with <60 GP but ≥15 min_pg
    - avg3yr_injury_flag: 3-year rolling injury frequency
    - career_injury_seasons: cumulative injury seasons
    - covid_flag: indicator for 2020 COVID-shortened season
    - post_covid_flag: indicator for 2021 post-COVID season

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'career_gp', 'years_in_league', 'gp',
        'min_pg', 'year', 'player_name_clean' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with availability features added.
    """
    out = df.copy()

    # Eligible games played (career availability rate)
    out["eligible_games_played"] = out["career_gp"] / (
        82 * (out["years_in_league"] + 1)
    )

    # Drop career_gp as it's now captured in the rate
    if "career_gp" in out.columns:
        out = out.drop(columns=["career_gp"])

    # 3-year rolling availability
    out["avg3yr_eligible_games_played"] = out.groupby("player_name_clean")[
        "eligible_games_played"
    ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Injury flag (played <60 games despite ≥15 min_pg)
    out["injury_flag"] = np.where((out["gp"] < 60) & (out["min_pg"] >= 15), 1, 0)

    # 3-year rolling injury frequency
    out["avg3yr_injury_flag"] = out.groupby("player_name_clean")[
        "injury_flag"
    ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Cumulative injury seasons
    out["career_injury_seasons"] = out.groupby("player_name_clean")[
        "injury_flag"
    ].cumsum()

    # COVID impact flags
    out["covid_flag"] = np.where(out["year"] == 2020, 1, 0)
    out["post_covid_flag"] = np.where(out["year"] == 2021, 1, 0)

    return out


def add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add efficiency and per-unit production features to the dataframe.

    Features created:
    - fpts_per_gp: fantasy points per game played
    - fpts_per_min: fantasy points per minute (duplicate removal if exists)
    - fpts_per_gp_3yr: 3-year fantasy points per game
    - fpts_per_min_3yr: 3-year fantasy points per minute (from create_agg_metrics)
    - pts_per_usg: points per game / usage percentage
    - min_pg_growth_recent: current min_pg - 3yr avg
    - usg_growth_recent: current usg_pct - 3yr avg

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'fantasy_points', 'gp', 'mp', 'pts_pg',
        'usg_pct', 'min_pg', rolling average columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with efficiency features added.
    """
    out = df.copy()

    # Fantasy points per game
    out["fpts_per_gp"] = out["fantasy_points"] / out["gp"].replace(0, np.nan)

    # Fantasy points per minute (overwrite if exists from calculate_fantasy_points)
    out["fpts_per_min"] = out["fantasy_points"] / out["mp"].replace(0, np.nan)

    # 3-year efficiency rates
    out["fpts_per_gp_3yr"] = out["sum3yr_fantasy_points"] / out["sum3yr_gp"].replace(
        0, np.nan
    )

    # Points per usage (usage efficiency)
    out["pts_per_usg"] = out["pts_pg"] / out["usg_pct"].replace(0, np.nan)

    # Recent growth trends
    out["min_pg_growth_recent"] = out["min_pg"] - out["avg3yr_min_pg"]
    out["usg_growth_recent"] = out["usg_pct"] - out["avg3yr_usg_pct"]

    return out


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend, volatility, and momentum features to the dataframe.

    Features created:
    - fpts_std_3yr: 3-year rolling standard deviation of fantasy points
    - fpts_yoy_pct_change: year-over-year percentage change
    - has_playoff_exp: flag for career playoff experience

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'fantasy_points', 'career_playoff_pts',
        'player_name_clean' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with trend features added.

    Notes
    -----
    Career trajectory trends (bpm_trend_3yr, fantasy_points_trend_3yr, etc.)
    are computed in create_agg_metrics() using rolling_trend().
    """
    out = df.copy()

    # Volatility (3-year rolling std)
    out["fpts_std_3yr"] = out.groupby("player_name_clean")["fantasy_points"].transform(
        lambda x: x.rolling(3, min_periods=2).std()
    )

    # Year-over-year momentum
    out["fpts_yoy_pct_change"] = out.groupby("player_name_clean")[
        "fantasy_points"
    ].pct_change()

    # Playoff experience flag
    out["has_playoff_exp"] = (out["career_playoff_pts"] > 0).astype(int)

    return out


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add contextual and role-based features to the dataframe.

    Features created:
    - starter_proxy: flag for players averaging ≥25 min_pg

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'min_pg' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with context features added.
    """
    out = df.copy()

    # Starter proxy based on minutes
    out["starter_proxy"] = (out["min_pg"] >= 25).astype(int)

    return out


def run_feature_pipeline(
    df: pd.DataFrame,
    start_year: int,
    core_stats: list,
    career_stats: list,
) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline on the combined dataframe.

    This master orchestrator function chains all feature engineering steps
    in the correct order after `calculate_fantasy_points()` has been called.

    Pipeline order:
    1. Base features (productivity, peak timing, era)
    2. Awards parsing and college binning
    3. Aggregation metrics (career totals, rolling averages, trends, deltas)
    4. Derived features (age, availability, efficiency, trend, context)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe after `calculate_fantasy_points()` has been applied.
        Must contain fantasy_points column.
    start_year : int
        First year of data collection (used for incomplete history detection).
    core_stats : list
        List of stat column names for rolling averages and deltas.
    career_stats : list
        List of stat column names for career cumulative totals.

    Returns
    -------
    pd.DataFrame
        DataFrame with all features added, sorted by player and year.

    Example
    -------
    >>> combined_df = calculate_fantasy_points(combined_df)
    >>> combined_df = run_feature_pipeline(combined_df, START_YEAR, CORE_STATS, CAREER_STATS)
    """
    out = df.copy()

    # -----------------------------------------------------------------
    # Step 1: Base features (player-level, no historical aggregation)
    # -----------------------------------------------------------------
    # Productivity score metrics
    out = calculate_productivity_score(out, age_col="age")

    # Peak timing features
    out = calculate_years_since_peak(out)

    # Era categorical feature
    out = add_era_feature(out)

    # -----------------------------------------------------------------
    # Step 2: Awards and college binning
    # -----------------------------------------------------------------
    # Parse awards into individual columns
    out = process_awards(out)

    # Bin colleges into categories
    out = assign_college_bin(out)

    # -----------------------------------------------------------------
    # Step 3: Aggregation metrics (career totals, rolling avgs, trends)
    # -----------------------------------------------------------------
    out = create_agg_metrics(out, core_stats, career_stats)

    # -----------------------------------------------------------------
    # Step 4: Derived features (depend on aggregation metrics)
    # -----------------------------------------------------------------
    # Age-related features
    out = add_age_features(out, start_year)

    # Availability features
    out = add_availability_features(out)

    # Efficiency features
    out = add_efficiency_features(out)

    # Trend features
    out = add_trend_features(out)

    # Context features
    out = add_context_features(out)

    # -----------------------------------------------------------------
    # Final cleanup
    # -----------------------------------------------------------------
    # Dtype cleanup for categorical columns
    if "pos" in out.columns:
        out["pos"] = out["pos"].astype("category")
    if "team" in out.columns:
        out["team"] = out["team"].astype("category")

    # Sort by player and year
    out = out.sort_values(by=["player_name_clean", "year"]).reset_index(drop=True)

    return out
