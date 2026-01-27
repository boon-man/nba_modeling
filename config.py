##################################################
# Configuration file for NBA player performance prediction model
##################################################

import warnings
import numpy as np
from hyperopt import hp

# =============================================================================
# Warning Suppression
# =============================================================================
# Suppress common warnings from dependencies (pandas, sklearn, xgboost, etc.)
# This keeps notebook output clean while still allowing errors to surface.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# League Settings
# =============================================================================

# Setting roster size & league size parameters
ROSTER_SIZE = 16
LEAGUE_SIZE = 12
# Defining bonus player pool size for draft board
BONUS_PLAYER_POOL_MULT = 1.15

# Defining positional roster allocations
G_SPLIT = 0.4
W_SPLIT = 0.4
B_SPLIT = 0.2


# Defining URL for pulling NBA projections
FANTASYPROS_URL = "https://www.fantasypros.com/nba/stats/overall.php"

# =============================================================================
# Model Hyperparameter Space
# =============================================================================
# Define Hyperopt search space for final model tuning
SPACE = {
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.1)),
    # leaf-based complexity control
    "max_leaves": hp.quniform("max_leaves", 8, 48, 1),
    "subsample": hp.uniform("subsample", 0.75, 0.95),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.65, 0.95),
    "min_child_weight": hp.loguniform("min_child_weight", np.log(0.1), np.log(25.0)),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(5.0)),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-3), np.log(5.0)),
    "gamma": hp.loguniform("gamma", np.log(1e-3), np.log(2.0)),
}

# =============================================================================
# Data Processing Settings
# =============================================================================

# Define set of NBA team abbreviations (current + recent expansion/contraction awareness)
NBA_TEAMS = {
    "ATL",
    "BOS",
    "BKN",
    "CHA",
    "CHI",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GSW",
    "HOU",
    "IND",
    "LAC",
    "LAL",
    "MEM",
    "MIA",
    "MIL",
    "MIN",
    "NOP",
    "NYK",
    "OKC",
    "ORL",
    "PHI",
    "PHX",
    "POR",
    "SAC",
    "SAS",
    "TOR",
    "UTA",
    "WAS",
}

# Selecting columns to keep for the model
SELECTED_COLUMNS = [
    "player_id",
    "player_name_clean",
    "season",
    "year",
    "age",
    "pts",
    "reb",
    "ast",
    "gp",
    "usg_pct",
    "w",
    "w_pct",
    "fgm",
    "fga",
    "fg_pct",
    "fg3m",
    "fg3a",
    "fg3_pct",
    "fta",
    "ft_pct",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "tov",
    "stl",
    "blk",
    "blka",
    "pfd",
    "plus_minus",
    "dd2",
    "td3",
    "min_rank",
    "fgm_rank",
    "fg_pct_rank",
    "fg3a_rank",
    "fg3_pct_rank",
    "fta_rank",
    "oreb_rank",
    "reb_rank",
    "ast_rank",
    "stl_rank",
    "blk_rank",
    "pfd_rank",
    "off_rating",
    "def_rating",
    "net_rating",
    "ast_pct",
    "ast_to",
    "ast_ratio",
    "oreb_pct",
    "reb_pct",
    "tm_tov_pct",
    "efg_pct",
    "ts_pct",
    "pace",
    "pie",
    "fgm_pg",
    "fga_pg",
    "playoff_min",
    "playoff_pts",
    "playoff_fga",
    "playoff_fta",
    "playoff_nba_fantasy_pts",
    "playoff_plus_minus",
    "playoff_efg_pct",
]

BREF_COLS = [
    "player_name_clean",
    "year",
    "team",
    "pos",
    "mp",
    "min_pg",
    "trb_pg",
    "ast_pg",
    "stl_pg",
    "blk_pg",
    "tov_pg",
    "pts_pg",
    "3par",
    "ftr",
    "orbpct",
    "astpct",
    "stlpct",
    "blkpct",
    "tovpct",
    "ws",
    "ws/48",
    "bpm",
    "vorp",
    "awards",
    "rookie_year",
    "was_drafted",
    "pk",
    "college",
    "all_star",
    "all_def",
    "dpoy",
    "mvp",
    "all_nba",
    "roy",
]

# Core stats for tracking career totals and year over year statistical differences
CORE_STATS = [
    "gp",
    "eligible_games_played",
    "mp",
    "min_pg",
    "fgm_pg",
    "ftm",
    "pts",
    "ast",
    "reb",
    "blk",
    "stl",
    "fg3m",
    "playoff_pts",
    "usg_pct",
    "ws",
    "bpm",
    "vorp",
    "all_star",
    "all_nba_points",
    "fantasy_points",
]

# Stats to track for career totals
CAREER_STATS = [
    "eligible_games_played",
    "gp",
    "mp",
    "pts",
    "ast",
    "reb",
    "fg3m",
    "playoff_pts",
    "ws",
    "bpm",
    "vorp",
    "fantasy_points",
]

# Dictionary for name overrides to ensure matching names between datasets
NAME_OVERRIDES = {
    "jimmy butler iii": "jimmy butler",
    "kevin knox ii": "kevin knox",
    "marcus morris sr": "marcus morris",
    "reggie bullock jr": "reggie bullock",
    "robert williams iii": "robert williams",
    "xavier tillman sr": "xavier tillman",
    "brandon boston jr": "brandon boston",
    "gg jackson ii": "gg jackson",
    "trey jemison iii": "trey jemison",
}

# Awards that come with a ranking (MVP-11, CPOY-1, etc.)
RANKED_AWARDS = {
    "MVP": "mvp_rank",
    "CPOY": "cpoy_rank",
    "DPOY": "dpoy_rank",
    "6MOY": "smoy_rank",  # Sixth Man
    "MIP": "mip_rank",
    "ROY": "roy_rank",
}

# Tokens that mean "this player made the All-Star team"
ALL_STAR_TOKENS = {"AS"}

# All-NBA team points
ALL_NBA_POINTS = {
    "NBA1": 3,
    "NBA2": 2,
    "NBA3": 1,
}

# Pastel-ish palette with blue as primary
COLOR_PALETTE = [
    "#6baed6",  # primary blue
    "#fdae6b",  # stronger orange
    "#c7e9b4",  # green-ish
    "#dadaeb",  # pale purple
    "#fdd0a2",  # soft orange
    "#9ecae1",  # light blue
]
