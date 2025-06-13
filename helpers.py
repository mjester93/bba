from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NUM_TEAMS = 12


def next_pick_distribution(df: pd.DataFrame, player: str) -> pd.DataFrame:
    """
    For a given player, compute the distribution of the next player taken
    by the same team in the same draft without explicit row iteration.
    Returns a DataFrame with columns: next_player, count, pct.
    """
    # occurrences of the player with their pick number
    occ = (
        df.loc[
            df["PLAYERNAME"] == player,
            ["SNAKEDRAFTNUM", "draft_slot", "OVERALLPICKNUM"],
        ]
        .reset_index(drop=False)
        .rename(columns={"index": "occ_index", "OVERALLPICKNUM": "pick_num"})
    )
    if occ.empty:
        return pd.DataFrame(columns=["next_player", "count", "pct"])

    # merge to find candidates after pick
    merged = occ.merge(
        df[["SNAKEDRAFTNUM", "draft_slot", "OVERALLPICKNUM", "PLAYERNAME"]],
        on=["SNAKEDRAFTNUM", "draft_slot"],
        how="left",
    ).rename(columns={"OVERALLPICKNUM": "candidate_pick", "PLAYERNAME": "next_player"})

    # filter only picks after current pick
    filtered = merged[merged["candidate_pick"] > merged["pick_num"]]
    if filtered.empty:
        return pd.DataFrame(columns=["next_player", "count", "pct"])

    # find next pick per occurrence
    idx = filtered.groupby("occ_index")["candidate_pick"].idxmin()
    next_players = filtered.loc[idx, "next_player"]

    # build distribution
    dist = (
        next_players.value_counts().rename_axis("next_player").reset_index(name="count")
    )
    total = dist["count"].sum()
    dist["pct"] = dist["count"] / total * 100
    return dist


def co_draft_multiple(df: pd.DataFrame, players: list[str]) -> dict[str, Any]:
    """
    For a list of player names, compute how often they are all drafted
    by the same team in the same draft *whenever* the first player is drafted.

    Returns a dict with:
      - players: The list of players you passed in
      - num_first: Count of (draft,slot) pairs that picked players[0]
      - num_all: Count of those pairs that also picked *every* other player
      - pct_together: num_all / num_first * 100 (float)
      - common_pairs: set of (SNAKEDRAFTNUM, draft_slot) pairs where all were picked
    """
    if not players:

        raise ValueError("Please supply at least one player name.")

    pair_sets: list[set[tuple[int, int]]] = []
    for player in players:
        ps = set(
            zip(
                df.loc[df["PLAYERNAME"] == player, "SNAKEDRAFTNUM"],
                df.loc[df["PLAYERNAME"] == player, "draft_slot"],
            )
        )
        pair_sets.append(ps)

    # the drafts where the first player was picked
    first_set = pair_sets[0]

    # intersection across all players
    common = set.intersection(*pair_sets)

    num_first = len(first_set)
    num_all = len(common)
    pct = (num_all / num_first * 100) if num_first else 0.0

    return {
        "players": players,
        "num_first": num_first,
        "num_all": num_all,
        "pct_together": pct,
        "common_pairs": common,
    }


def slot(row: pd.Series[int]) -> int:
    if row["round"] % 2 == 1:
        # odd round: forward order
        return row["pick_in_round"]
    else:
        # even round: reversed order
        return NUM_TEAMS - row["pick_in_round"] + 1


def get_data() -> pd.DataFrame:
    base_path = Path(__file__).parent / "data"
    drafts = pd.read_csv(base_path / "drafts.csv")

    player_info = pd.read_csv(base_path / "player_info.csv")

    merged_df = drafts.merge(
        player_info[["Name", "Team", "Position", "ID"]],
        left_on="PLAYERNAME",
        right_on="Name",
        how="left",
    )

    # Display merged DataFrame

    merged_df["round"] = np.ceil(merged_df["OVERALLPICKNUM"] / NUM_TEAMS).astype(int)
    merged_df["pick_in_round"] = (
        merged_df["OVERALLPICKNUM"] - (merged_df["round"] - 1) * NUM_TEAMS
    )
    merged_df["draft_slot"] = merged_df.apply(slot, axis=1)

    print(merged_df.head())

    return merged_df
