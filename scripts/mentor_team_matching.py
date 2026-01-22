# run one-to-one matching 

from pathlib import Path

import pandas as pd
import seaborn as sns

from scipy.optimize import linear_sum_assignment

# %%

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
    
    team_to_mentor = team_df.pivot(index="requester", columns="requestee", values="rank")
    mentor_to_team = mentor_df.pivot(index="requester", columns="requestee", values="rank")

    # ensure all teams and mentors present in both tables
    all_teams = pd.concat([team_df["requester"], mentor_df["requestee"]]).unique()
    all_mentors = pd.concat([team_df["requestee"], mentor_df["requester"]]).unique()

    team_to_mentor = team_to_mentor.reindex(index=all_teams, columns=all_mentors)
    mentor_to_team = mentor_to_team.reindex(index=all_mentors, columns=all_teams)

    sns.heatmap(team_to_mentor, annot=team_to_mentor, fmt='.0f', cmap='YlGnBu')

    # get directionality
    # 0 - unmatched, 1 - team pref, 2 - mentor pref, 3 - mutual
    pair_type = (
        team_to_mentor.notna().astype(int)
        + mentor_to_team.T.notna().astype(int) * 2
    )
    sns.heatmap(pair_type, annot=pair_type)

    return team_to_mentor, mentor_to_team, pair_type


def get_matches(
    chapter_name: str, 
    team_to_mentor: pd.DataFrame,
    mentor_to_team: pd.DataFrame,
    na_fill_value: float
    ) -> pd.DataFrame:  
    """
    Run matching algorithm using linear sum assignment
    Minimizes total sum of pairs, where pair cost is defined as ranking sum
    (e.g. mutual rank of 1 between team and mentor will yield low score of 2)
    
    Args:
        chapter_name (str): chapter name
        team_to_mentor (pd.DataFrame): pivot table of team rankings of mentors
        mentor_to_team (pd.DataFrame): pivot table of mentor rankings of teams
        na_fill_value (float): back-fill value for unranked 

    Returns:
        pd.DataFrame: dataframe of match pairs with given scores and reason
    """

    # fill missing rows/columns with fill value (max ranking?)
    t2m_filled = team_to_mentor.fillna(na_fill_value).astype(int)
    m2t_filled = mentor_to_team.fillna(na_fill_value).astype(int)

    # "cost" matrix is the summed rankings, then minimize bipartite matching
    # TODO allow input option for custom bonus/re-weighting
    cost = t2m_filled + m2t_filled.T
    row_ind, col_ind = linear_sum_assignment(cost.values)

    # construct output table
    matches = pd.DataFrame({
        "chapter": chapter_name,
        "team": cost.index[row_ind],
        "mentor": cost.columns[col_ind],
        "mutual_rank_sum": cost.values[row_ind, col_ind],
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

    return matches

# %%

def main():

    
    output_dir = "output"

    output_all = Path(output_dir) / "all_chapter_matches.csv"

    df = pd.read_csv(Path(output_dir) / "mentor_team_match_requests_clean.csv")

    max_rank = df["rank"].max()

    all_chapter_matches = []

    for chapter, df_chapter in df.groupby("chapter"):

        print(f"Matching {chapter}")

        team_to_mentor, mentor_to_team, pair_type = team_mentor_pivots(df_chapter)
        matches = get_matches(
            chapter, team_to_mentor, mentor_to_team, na_fill_value=max_rank
            )
        print(matches)

        all_chapter_matches.append(matches)

    all_chapter_matches = pd.concat(all_chapter_matches)

    all_chapter_matches.to_csv(output_all)


if __name__ == '__main__':
    main()
