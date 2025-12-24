import os
import time
import random
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

from config import NBA_TEAMS
from data_cleaning import clean_name, flatten_multiindex_columns, keep_tot_or_first
from feature_engineering import expand_awards


# Function for either generating or loading existing NBA data
def load_or_generate_data(start_year, end_year):
    """
    Loads existing NBA player stats data from CSV files if available,
    otherwise generates the data by pulling from NBA API and Basketball Reference.

    Parameters:
        start_year (int): Starting year of the data range.
        end_year (int): Ending year of the data range.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing NBA API stats and Basketball Reference stats.
    """
    api_path = f"data/api_player_stats_{start_year}_{end_year}.csv"
    bref_path = f"data/bref_player_stats_{start_year}_{end_year}.csv"

    # Load or generate api_df
    if os.path.exists(api_path):
        api_df = pd.read_csv(api_path)
        print("Loaded existing API data from CSV file.")
    else:
        api_df = get_multi_season_base_and_advanced_stats(
            start_year=start_year, end_year=end_year
        )
        save_dataframe_to_csv(api_df, path=api_path)

    # Load or generate bref_df
    if os.path.exists(bref_path):
        bref_df = pd.read_csv(bref_path)
        print("Loaded existing BRef data from CSV file.")
    else:
        bref_df = get_combined_bref_data(start_year=start_year, end_year=end_year)
        save_dataframe_to_csv(bref_df, path=bref_path)

    return api_df, bref_df


def get_multi_season_base_and_advanced_stats(
    start_year: int, end_year: int, min_minutes: int = 300
) -> pd.DataFrame:
    """
    Retrieve, merge, and clean NBA player statistics (base and advanced) for multiple seasons using the NBA API.

    For each season in the specified range, this function:
      - Pulls both base and advanced regular season stats for all players.
      - Merges base and advanced stats, removing duplicate columns.
      - Filters out players with fewer than `min_minutes` played in the regular season.
      - Attempts to pull playoff base and advanced stats, merges them, and attaches as suffix columns (with "playoff_" prefix).
      - Filters playoff stats to players with at least `min_minutes` played in the playoffs.
      - Merges playoff stats onto regular season stats for each player-season.
      - Filters to only NBA teams (removes WNBA or other leagues if present).
      - Drops columns related to WNBA if present.
      - Adds a cleaned player name column for downstream merging.

    Parameters
    ----------
    start_year : int
        The starting season (e.g., 2010 for the 2010-11 season).
    end_year : int
        The ending season (e.g., 2024 for the 2023-24 season).
    min_minutes : int, default 300
        Minimum minutes played required for a player to be included in the output for both regular season and playoffs.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing regular season and playoff stats for all qualifying NBA players
        across the specified seasons. Each row represents a player-season, with playoff stats as suffix columns.

    Notes
    -----
    - The function sleeps between API calls to avoid rate limiting.
    - The output DataFrame includes a "player_name_clean" column for consistent merging.
    - Columns from playoff stats are prefixed with "playoff_".
    - Only NBA teams are included in the final output.
    - The function is designed for use in automated data pipelines and may print progress messages.
    """
    all_season_data = []

    for year in range(start_year, end_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        print(f"Fetching NBA API data for {season_str}...")

        # === Regular Season (Base + Advanced) ===
        time.sleep(1)
        base = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base",
        ).get_data_frames()[0]

        time.sleep(1)
        adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
        ).get_data_frames()[0]

        base["SEASON"] = season_str
        adv["SEASON"] = season_str

        base["YEAR"] = year + 1  # For 2022-23, this will be 2023

        # Drop overlapping columns in advanced before merge
        shared_cols = [
            col
            for col in adv.columns
            if col in base.columns and col not in ["PLAYER_ID", "SEASON"]
        ]
        adv_filtered = adv.drop(columns=shared_cols)

        regular_df = pd.merge(
            base, adv_filtered, on=["PLAYER_ID", "SEASON"], how="left"
        )

        # Filtering the regular season data to only include players with enough minutes
        regular_df = regular_df[regular_df["MIN"] >= min_minutes]

        # === Playoffs Data Pull ===
        try:
            time.sleep(1)
            playoff_base = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season_str,
                season_type_all_star="Playoffs",
                measure_type_detailed_defense="Base",
            ).get_data_frames()[0]

            time.sleep(1)
            playoff_adv = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season_str,
                season_type_all_star="Playoffs",
                measure_type_detailed_defense="Advanced",
            ).get_data_frames()[0]

            playoff_base["SEASON"] = season_str
            playoff_adv["SEASON"] = season_str

            # Drop overlapping columns in advanced before merge
            shared_cols = [
                col
                for col in playoff_adv.columns
                if col in playoff_base.columns and col not in ["PLAYER_ID", "SEASON"]
            ]
            playoff_adv_filtered = adv.drop(columns=shared_cols)

            playoff_df = pd.merge(
                playoff_base,
                playoff_adv_filtered,
                on=["PLAYER_ID", "SEASON"],
                how="left",
            )

            # Keep only players with at least min_minutes in playoffs
            playoff_df = playoff_df[playoff_df["MIN"] >= min_minutes]

            # Rename columns (except keys) to add "playoff_" prefix
            playoff_df = playoff_df.rename(
                columns={
                    col: f"playoff_{col}"
                    for col in playoff_df.columns
                    if col not in ["PLAYER_ID", "SEASON"]
                }
            )

            # Merge playoff stats onto regular_df
            merged = pd.merge(
                regular_df, playoff_df, on=["PLAYER_ID", "SEASON"], how="left"
            )

        except Exception as e:
            print(f"Playoff data not available for {season_str}: {e}")
            merged = regular_df  # Use regular season only if playoffs not available

        # Filter to only include NBA players
        merged = merged[merged["TEAM_ABBREVIATION"].isin(NBA_TEAMS)]

        # Drop WNBA-related columns
        merged = merged.loc[:, ~merged.columns.str.contains("WNBA")]

        # Create cleaned player name for merging
        merged["player_name_clean"] = merged["PLAYER_NAME"].apply(clean_name)

        all_season_data.append(merged)

    return pd.concat(all_season_data, ignore_index=True)


def pull_bref_table(start_year: int, end_year: int, table_type: str) -> pd.DataFrame:
    """
    Download and concatenate Basketball Reference tables for multiple NBA seasons.

    For each season in the specified range, this function:
      - Constructs the appropriate Basketball Reference URL for the given table type (e.g., "advanced", "per_poss").
      - Downloads the table using pandas' `read_html`.
      - Removes repeated header rows (where "Rk" appears as a value).
      - Adds a "year" column corresponding to the season end year.
      - Appends the cleaned DataFrame to a list.

    Parameters
    ----------
    start_year : int
        The starting season (e.g., 2010 for the 2010-11 season).
    end_year : int
        The ending season (e.g., 2024 for the 2023-24 season).
    table_type : str
        Table type to pull from Basketball Reference (e.g., "advanced", "per_poss").

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame containing the requested table for all specified seasons.

    Notes
    -----
    - Sleeps 4-6 seconds between requests to avoid overloading the server.
    - The returned DataFrame includes a "year" column for each row.
    - Prints progress messages for each season/table fetched.
    """
    bref_list = []
    for year in range(start_year, end_year + 1):
        season_end = year + 1  # BRef uses the END year in its URL
        url = f"https://www.basketball-reference.com/leagues/NBA_{season_end}_{table_type}.html"
        print(
            f"Fetching {table_type} table for {year}-{str(season_end)[-2:]} season..."
        )
        df = pd.read_html(url, header=0)[0]
        df = df[df["Rk"] != "Rk"]
        df["year"] = season_end  # keep consistency with URL naming
        bref_list.append(df)
        time.sleep(4 + random.random() * 2)
    return pd.concat(bref_list, ignore_index=True)


def pull_bref_draft_table(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Download and concatenate NBA draft tables from Basketball Reference for multiple seasons.

    For each season in the specified range, this function:
      - Constructs the Basketball Reference draft URL for the given year.
      - Downloads all tables on the page using pandas' `read_html` with multi-level headers.
      - Flattens multi-index columns for easier access.
      - Identifies and extracts the columns for draft pick ("Pk"), player name ("Player"), and college ("College").
      - Cleans and standardizes the table, renaming columns and adding a "draft_year" column (using the season end year).
      - Drops rows where draft pick is missing or repeated header rows.
      - Appends the cleaned DataFrame to a list.

    After all years are processed:
      - Concatenates all draft DataFrames.
      - Converts "Pk" to numeric for sorting.
      - For players with multiple draft entries, keeps only the best (lowest) draft pick per player.
      - Returns a DataFrame with one row per drafted player.

    Parameters
    ----------
    start_year : int
        The starting draft year (e.g., 2000).
    end_year : int
        The ending draft year (e.g., 2024).

    Returns
    -------
    pd.DataFrame
        DataFrame containing draft pick, player name, college, and draft year for all drafted players
        across the specified years. Each player appears only once, with their best draft pick.

    Notes
    -----
    - Sleeps 4-6 seconds between requests to avoid overloading the server.
    - Prints progress messages for each draft year fetched.
    - Uses the end year of the season for the "draft_year" column for consistency with other data.
    - Assumes the presence of columns ending with "Pk", "Player", and "College" in the draft tables.
    """
    all_drafts = []
    for year in range(start_year, end_year + 1):
        url = f"https://www.basketball-reference.com/draft/NBA_{year}.html"
        print(f"Fetching draft table for {year}...")
        tables = pd.read_html(url, header=[0, 1])
        for table in tables:
            table.columns = flatten_multiindex_columns(table)
            # Find the relevant columns by suffix
            col_pk = [c for c in table.columns if c.endswith("Pk")][0]
            col_player = [c for c in table.columns if c.endswith("Player")][0]
            col_college = [c for c in table.columns if c.endswith("College")][0]
            # Subset and rename
            clean = table[[col_pk, col_player, col_college]].copy()
            clean.columns = ["Pk", "Player", "College"]
            clean["draft_year"] = year + 1  # Use end year of season for consistency
            # Drop rows where Pk is "Pk" or is null
            clean = clean[(clean["Pk"] != "Pk") & (clean["Pk"].notnull())]

            all_drafts.append(clean)
        time.sleep(4 + random.random() * 2)

    draft_df = pd.concat(all_drafts, ignore_index=True)

    # Convert Pk to numeric for sorting
    draft_df["Pk"] = pd.to_numeric(draft_df["Pk"], errors="coerce")

    # Keep only the best (lowest) draft pick per player name
    # Sort by Player and Pk, then keep first occurrence
    draft_df = draft_df.sort_values(["Player", "Pk"]).drop_duplicates(
        subset=["Player"], keep="first"
    )

    return draft_df


def get_combined_bref_data(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Retrieve, merge, and enrich Basketball Reference player data (advanced, per-possession, and draft) for multiple NBA seasons.

    This function:
      - Downloads advanced and per-possession tables for all NBA players for each season in the specified range.
      - Downloads NBA draft data for a wider range (start_year - 15 to end_year) to capture all possible active players.
      - Cleans and standardizes player/team columns, ensuring one row per player-season (using 'TOT' row if available).
      - Adds a normalized player name column for consistent merging.
      - Merges advanced and per-possession stats on player and year.
      - Merges draft info onto player records (by player, not by year).
      - Fills missing college info with 'NO COLLEGE'.
      - Determines rookie year for each player using draft year if available, otherwise the earliest playing year.
      - Flags whether a player was drafted.
      - Adds per-game stat columns and other derived features.
      - Expands awards/achievements into separate columns using `expand_awards`.
      - Returns a single DataFrame with one row per player-season, including all merged and engineered features.

    Parameters
    ----------
    start_year : int
        The starting season (e.g., 2000 for the 2000-01 season).
    end_year : int
        The ending season (e.g., 2024 for the 2023-24 season).

    Returns
    -------
    pd.DataFrame
        DataFrame containing merged advanced stats, per-possession stats, draft info, and expanded awards
        for all NBA players across the specified seasons. Each row represents a player-season.

    Notes
    -----
    - Player and team columns are dynamically detected for robustness to BRef schema changes.
    - Draft info is merged by player only, not by year.
    - Awards are parsed and expanded into separate columns for downstream modeling.
    - The function is designed for use in automated data pipelines and may print progress messages.
    """
    adv_df = pull_bref_table(start_year, end_year, "advanced")
    per_poss_df = pull_bref_table(start_year, end_year, "per_poss")

    # Pull *extra* draft years (back 15 years)
    draft_df = pull_bref_draft_table((start_year - 15), end_year)

    def get_col(cols, ending):
        return [c for c in cols if ending.lower() in c.lower()][0]

    adv_player_col = get_col(adv_df.columns, "player")
    adv_team_col = get_col(adv_df.columns, "team")
    per_poss_player_col = get_col(per_poss_df.columns, "player")
    per_poss_team_col = get_col(per_poss_df.columns, "team")
    draft_player_col = "Player"

    adv_df = keep_tot_or_first(
        adv_df, team_col=adv_team_col, player_col=adv_player_col, year_col="year"
    )
    per_poss_df = keep_tot_or_first(
        per_poss_df,
        team_col=per_poss_team_col,
        player_col=per_poss_player_col,
        year_col="year",
    )

    adv_df["player_name_clean"] = adv_df[adv_player_col].apply(clean_name)
    per_poss_df["player_name_clean"] = per_poss_df[per_poss_player_col].apply(
        clean_name
    )
    draft_df["player_name_clean"] = draft_df[draft_player_col].apply(clean_name)

    # Merge advanced and per_poss (as before)
    combined = adv_df.merge(
        per_poss_df[
            ["player_name_clean", "year"]
            + [
                col
                for col in per_poss_df.columns
                if col not in [per_poss_player_col, "Rk", "year", "player_name_clean"]
            ]
        ],
        on=["player_name_clean", "year"],
        how="outer",
        suffixes=("", "_perposs"),
    )

    # Merge draft info (by player only, not by year)
    combined = combined.merge(
        draft_df[["player_name_clean", "draft_year", "Pk", "College"]],
        on="player_name_clean",
        how="left",
        suffixes=("", "_draft"),
    )

    # Fill null College values with 'NO COLLEGE'
    combined["College"] = combined["College"].fillna("NO COLLEGE")

    # Determine rookie year for each player
    # Using draft year if available, otherwise using the minimum playing year recorded
    combined["rookie_year"] = combined.apply(
        lambda row: row["draft_year"] if pd.notnull(row["draft_year"]) else row["year"],
        axis=1,
    )
    combined["rookie_year"] = combined.groupby("player_name_clean")[
        "rookie_year"
    ].transform("min")

    # Designate undrafted players
    combined["was_drafted"] = combined["Pk"].notna().astype(int)

    # Creating a min_pg column from the existing MP column
    combined["min_pg"] = combined["MP"] / combined["G"]

    # Columns to rename to per-game equivalents within the final DataFrame
    rename_map = {
        "TRB": "TRB_pg",
        "AST": "AST_pg",
        "STL": "STL_pg",
        "BLK": "BLK_pg",
        "TOV": "TOV_pg",
        "PTS": "PTS_pg",
    }

    # Apply rename
    combined = combined.rename(columns=rename_map)

    # Fixing the Awards column format so that awards voting ranks are numeric and placing into the final output Dataframe
    expanded = expand_awards(combined[["player_name_clean", "year", "Awards"]])

    final = combined.merge(expanded, on=["player_name_clean", "year"], how="left")

    return final


def save_dataframe_to_csv(df: pd.DataFrame, path: str):
    """
    Saves a DataFrame to a CSV file, creating directories as needed.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        path (str): File path to write the CSV to.
    """
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved DataFrame to: {path}")
