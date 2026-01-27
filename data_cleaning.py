import pandas as pd
import unicodedata

from config import NAME_OVERRIDES


def merge_nba_dataframes(
    api_df: pd.DataFrame,
    bref_df: pd.DataFrame,
    *,
    join_keys: list[str] = None,
    required_col: str = "team",
    reorder_cols: list[str] = None,
) -> pd.DataFrame:
    """
    Merge NBA API data with Basketball Reference data and clean the result.

    Performs a left join on the specified keys, removes duplicate columns,
    drops rows missing required data, and reorders key columns for readability.

    Parameters
    ----------
    api_df : pd.DataFrame
        Primary dataframe from NBA API containing player statistics.
    bref_df : pd.DataFrame
        Secondary dataframe from Basketball Reference with advanced metrics.
    join_keys : list[str], optional
        Columns to join on. Defaults to ["player_name_clean", "year"].
    required_col : str, default "team"
        Column that must be non-null; rows with missing values are dropped.
    reorder_cols : list[str], optional
        Columns to move immediately after "year" for readability.
        Defaults to ["team", "ws"].

    Returns
    -------
    pd.DataFrame
        Merged and cleaned dataframe sorted by player name and year.

    Notes
    -----
    - Columns from bref_df that duplicate api_df columns (excluding join keys)
      receive a "_bref" suffix; only the first occurrence is kept.
    - Rows where `required_col` is null indicate failed joins and are dropped.
    """
    if join_keys is None:
        join_keys = ["player_name_clean", "year"]
    if reorder_cols is None:
        reorder_cols = ["team", "ws"]

    # Merge dataframes
    combined = pd.merge(
        api_df,
        bref_df,
        on=join_keys,
        how="left",
        suffixes=("", "_bref"),
    )

    # Remove duplicate columns (keep first occurrence)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Drop rows missing required column
    combined = combined[combined[required_col].notna()]

    # Sort by player and year
    combined = combined.sort_values(by=join_keys).reset_index(drop=True)

    # Reorder specified columns to appear after "year"
    for i, col in enumerate(reorder_cols):
        if col in combined.columns:
            insert_pos = combined.columns.get_loc("year") + 1 + i
            combined.insert(insert_pos, col, combined.pop(col))

    return combined


def clean_name(name: str) -> str:
    """
    Cleans and normalizes player names by removing punctuation,
    suffixes, and converting accented characters to ASCII.
    """

    # Normalize accented characters to ASCII
    name = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode("utf-8")
    )

    # Clean punctuation, suffixes, and formatting
    name = (
        name.replace(".", "")
        .replace("*", "")
        .replace("'", "")
        .replace("-", " ")
        .replace(" jr", "")
        .replace(" iii", "")
        .lower()
        .strip()
    )

    return name


def flatten_multiindex_columns(df):
    return [
        "_".join([str(x) for x in col if "Unnamed" not in str(x)]).strip("_")
        for col in df.columns
    ]


def get_player_column(df):
    for col in df.columns:
        col_str = str(col)
        if col_str.strip().lower().startswith("player"):
            return col
    raise ValueError("No player column found!")


def keep_tot_or_first(df, team_col="Tm", player_col="Player", year_col="year"):
    """
    For each player-season, return the 'TOT' row if present (representing season totals for traded players),
    otherwise return the first available row for that player and season.

    This function is designed for Basketball Reference data, where players who switch teams mid-season
    have one row per team and an additional 'TOT' row summarizing their season stats.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing player-season rows, possibly with multiple rows per player per season.
    team_col : str, default "Tm"
        Column name indicating the team (should contain 'TOT' for season totals).
    player_col : str, default "Player"
        Column name indicating the player.
    year_col : str, default "year"
        Column name indicating the season/year.

    Returns
    -------
    pd.DataFrame
        DataFrame with at most one row per player per season, preferring the 'TOT' row if available.

    Notes
    -----
    - The function sorts the DataFrame by player, year, and team before grouping.
    - If a player has a 'TOT' row for a season, that row is selected; otherwise, the first team row is used.
    - The output DataFrame preserves the original columns and dtypes.
    """
    df = df.sort_values([player_col, year_col, team_col])
    result = []
    for (player, year), group in df.groupby([player_col, year_col]):
        if "TOT" in group[team_col].values:
            result.append(group[group[team_col] == "TOT"].iloc[0])
        else:
            result.append(group.iloc[0])
    return pd.DataFrame(result)


def clean_and_refine_nba_data(
    df: pd.DataFrame, NAME_OVERRIDES=NAME_OVERRIDES
) -> pd.DataFrame:
    """
    Cleans the NBA DataFrame by normalizing column names and removing unnecessary columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Normalize column names to lower case and replace spaces and percentage signs
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("%", "pct")

    # Ensuring that statistical columns are numeric
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                continue

    # Fill missing with zero for numeric stats that can't be negative
    stat_cols = [col for col in df.columns if df[col].dtype in ["float64", "int64"]]
    df[stat_cols] = df[stat_cols].fillna(0)

    # Cleaning player names
    df["player_name_clean"] = df["player_name_clean"].replace(NAME_OVERRIDES)

    return df
