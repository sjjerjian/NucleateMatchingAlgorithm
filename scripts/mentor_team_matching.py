# run one-to-one matching 
import argparse
from pathlib import Path

import pandas as pd
from scipy.optimize import linear_sum_assignment

from plot_utils import (
    visualize_prefs, visualize_matches
)


# %%

def _strip_first_name(name_str):

    name_split = name_str.split()
    name_split[0] = name_split[0][0]
    return " ".join(name_split)

def team_mentor_pivots(
    df_chapter: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    create intermediate pivot tables of team and mentor rankings for single chapter

    Args:
        df_chapter (pd.DataFrame): long format df of cleaned rankings for single chapter

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: pivot tables of 
            team_to_mentor: team-mentor preferences
            mentor_to_mentor: mentor-team preferences
            pair_types: flag for directionality of preferences
    """

    # pivot tables: team vs mentor, mentor vs team, with rankings as values
    team_df = df_chapter[df_chapter["type"] == "Team"]
    mentor_df = df_chapter[df_chapter["type"] == "Mentor"]

    team_df['requestee'] = team_df['requestee'].apply(_strip_first_name)
    mentor_df['requester'] = mentor_df['requester'].apply(_strip_first_name)

    team_to_mentor = team_df.pivot(index="requester", columns="requestee", values="rank")
    mentor_to_team = mentor_df.pivot(index="requester", columns="requestee", values="rank")

    # ensure all teams and mentors present in both tables
    all_teams = pd.concat([team_df["requester"], mentor_df["requestee"]]).unique()
    all_mentors = pd.concat([team_df["requestee"], mentor_df["requester"]]).unique()

    team_to_mentor = team_to_mentor.reindex(index=all_teams, columns=all_mentors)
    mentor_to_team = mentor_to_team.reindex(index=all_mentors, columns=all_teams)

    # get directionality
    # 0 - unmatched, 1 - team pref, 2 - mentor pref, 3 - mutual
    pair_type = (
        team_to_mentor.notna().astype(int)
        + mentor_to_team.T.notna().astype(int) * 2
    )
    return team_to_mentor, mentor_to_team, pair_type


def get_matches(
    chapter_name: str, 
    team_to_mentor: pd.DataFrame,
    mentor_to_team: pd.DataFrame,
    top_rank_bonus: float = 0,
    unranked_penalty: float = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:  
    """
    Run matching algorithm using linear sum assignment
    Minimizes total sum of pairs, where pair cost is defined as ranking sum
    (e.g. mutual rank of 1 between team and mentor will yield low score of 2)
    
    Args:
        chapter_name (str): chapter name
        team_to_mentor (pd.DataFrame): pivot table of team rankings of mentors
        mentor_to_team (pd.DataFrame): pivot table of mentor rankings of teams
        top_rank_bonus: bonus to give for mutual top 1 rankings
        unranked_penalty: penalty to add to unranked edge

    Returns:
        pd.DataFrame: dataframe of match pairs with given scores and reason
        pd.DataFrame: overall cost matrix used to match pairs
    """

    t2m_na_fill = team_to_mentor.max().max() + unranked_penalty
    m2t_na_fill = mentor_to_team.max().max() + unranked_penalty
        
    # fill missing rows/columns with fill value (max ranking?)
    t2m_filled = team_to_mentor.fillna(t2m_na_fill).astype(int)
    m2t_filled = mentor_to_team.fillna(m2t_na_fill).astype(int)

    # "cost" matrix is the summed rankings, then minimize bipartite matching
    cost_raw = t2m_filled + m2t_filled.T
    cost_adj = cost_raw.values
    cost_adj[cost_adj==2] -= top_rank_bonus
    row_ind, col_ind = linear_sum_assignment(cost_adj)

    # construct output table
    matches = pd.DataFrame({
        "chapter": chapter_name,
        "team": cost_raw.index[row_ind],
        "mentor": cost_raw.columns[col_ind],
        "mutual_rank_sum": cost_raw.values[row_ind, col_ind],
        "team_rank": team_to_mentor.to_numpy()[row_ind, col_ind],
        "mentor_rank": mentor_to_team.to_numpy()[col_ind, row_ind]
    })

    # store provenance of match reasons
    matches["match_type"] = (
        matches["team_rank"].notna().astype(int)
        + matches["mentor_rank"].notna().astype(int)
    ).map({
        3: "mutual",
        2: "mentor pref",
        1: "team pref",
        0: "unranked"
    })

    return matches, cost_raw



# %%

def main():

    parser = argparse.ArgumentParser(
        description="Per-chapter team-mentor matching pipeline"
    )
    parser.add_argument(
        "--input-csv", 
        help='',
        default="mentor_team_match_requests_clean.csv"
        )
    parser.add_argument(
        "--output-dir", 
        help='',
        default='output'
        )
    parser.add_argument(
        "--top_rank_bonus",
        help="extra bonus for mutual top 1 matches",
        default=0
    )
    parser.add_argument(
        "--unranked_penalty",
        help="extra penalty for unranked edge, on top of default (max_rank+1)",
        default=1
        )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_csv = output_dir / "all_chapter_matches.csv"
    
    df = pd.read_csv(output_dir / args.input_csv)

    # run matching
    all_chapter_matches = []
    for chapter, df_chapter in df.groupby("chapter"):

        chapter_output_dir = output_dir / chapter
        Path(chapter_output_dir).mkdir(parents=True, exist_ok=True)

        print("="*60)
        print(f"Matching {chapter}")

        team_to_mentor, mentor_to_team, pair_type = team_mentor_pivots(df_chapter)

        # option to save these out here and manually edit?
        
        matches, cost_raw = get_matches(
            chapter,
            team_to_mentor,
            mentor_to_team, 
            top_rank_bonus=args.top_rank_bonus,
            unranked_penalty=args.unranked_penalty
            )
        
        visualize_prefs(
            team_to_mentor,
            mentor_to_team,
            pair_type,
            matches=matches,
            save_path=chapter_output_dir/f"{chapter}_team_mentor_preferences.png")
        
        print(matches)
        print("="*60)

        # visualize_matches(cost_raw, matches)
        all_chapter_matches.append(matches)

    all_chapter_matches = pd.concat(all_chapter_matches)
    all_chapter_matches.to_csv(output_csv, index=False)


if __name__ == '__main__':
    main()
