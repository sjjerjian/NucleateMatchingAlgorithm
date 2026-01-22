# run one-to-one matching based on schedule

# %%
import pandas as pd
import seaborn as sns

from pathlib import Path
from scipy.optimize import linear_sum_assignment

# %%

output_dir = "output"

output_all = Path(output_dir) / "all_chapter_matches.csv"

df = pd.read_csv(Path(output_dir) / "mentor_team_match_requests_clean.csv")

# %%

max_rank = df["rank"].max()

all_chapter_matches = []

for chapter, df_chapter in df.groupby("chapter"):

    team_df = df_chapter[df_chapter["type"] == "Team"]
    mentor_df = df_chapter[df_chapter["type"] == "Mentor"]
    team_to_mentor = team_df.pivot(index="requester", columns="requestee", values="rank")
    mentor_to_team = mentor_df.pivot(index="requester", columns="requestee", values="rank")
    
    all_teams = pd.concat([team_df["requester"], mentor_df["requestee"]]).unique()
    all_mentors = pd.concat([team_df["requestee"], mentor_df["requester"]]).unique()

    team_to_mentor = team_to_mentor.reindex(index=all_teams, columns=all_mentors)
    mentor_to_team = mentor_to_team.reindex(index=all_mentors, columns=all_teams)

    sns.heatmap(team_to_mentor, annot=team_to_mentor, fmt='.0f', cmap='YlGnBu')
    # sns.heatmap(mentor_to_team)

    # get directionality
    # 0 - unmatched, 1 - team to mentor only, 2 - mentor to team only, 3 - mutual
    pair_type = (
        team_to_mentor.notna().astype(int)
        + mentor_to_team.T.notna().astype(int) * 2
    )
    sns.heatmap(pair_type, annot=pair_type)
    
    # fill missing rows/columns with max ranking
    t2m_filled = team_to_mentor.fillna(max_rank).astype(int)
    m2t_filled = mentor_to_team.fillna(max_rank).astype(int)

    cost = t2m_filled + m2t_filled.T

    row_ind, col_ind = linear_sum_assignment(cost.values)

    matches = pd.DataFrame({
        "chapter": chapter,
        "team": cost.index[row_ind],
        "mentor": cost.columns[col_ind],
        "mutual_rank_sum": cost.values[row_ind, col_ind],
    })

    matches["team_rank"] = team_to_mentor.to_numpy()[row_ind, col_ind]
    matches["mentor_rank"] = mentor_to_team.to_numpy()[col_ind, row_ind]

    matches["match_type"] = (
        matches["team_rank"].notna().astype(int)
        + matches["mentor_rank"].notna().astype(int)
    ).map({
        3: "mutual",
        2: "mentor pref",
        1: "team pref",
        0: "unranked"
    })

    all_chapter_matches.append(matches)


all_chapter_matches = pd.concat(all_chapter_matches)

all_chapter_matches.to_csv("all_chapter_matches.csv")

# %%
