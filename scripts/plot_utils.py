import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns

from pathlib import Path
from typing import Optional

# ---------------- Brand palette (discrete) ----------------
# Core
WHITE = "#FFFFFF"
LIGHT_GREY = "#F9FAFC"
GREEN = "#13D6B0"
BLUE = "#38C7E8"
PURPLE = "#918AE1"
YELLOW = "#FFC162"
RED = "#FC827F"
GREY = "#E5E7EA"
BLACK = "#000000"

# Complementary pairs (for deeper steps if needed)
DARK_GREEN = "#045456"
LIGHT_GREEN = "#A6FBE3"
DARK_BLUE  = "#043E66"
LIGHT_BLUE = "#AFF2FB"
DARK_PURPLE = "#463473"
LIGHT_PURPLE = "#E3DEFF"
LIGHT_YELLOW = "#FFEFC2"
DARK_YELLOW = "#EC8200"
LIGHT_RED = "#FFE4E2"
DARK_RED = "#B20237"

# Rank → color (discrete up to 10). Adjust to taste.
# We prioritize strong, distinct hues for the first few ranks.
RANK_COLORS = [
    None,            # index 0 unused
    DARK_GREEN,      # 1
    DARK_BLUE,           # 2
    DARK_RED,            # 3
    RED,          # 4
    LIGHT_BLUE,          # 5
    LIGHT_GREEN,             # 6
    LIGHT_RED,      # 7
    LIGHT_YELLOW,    # 8
    GREY,    # 9
    LIGHT_GREY,       # 10
]

NAN_COLOR = LIGHT_GREY  # for missing cells (no ranking recorded)



# %% PLOTTING TOOLS

def _discrete_cmap_and_norm(max_rank_seen):
    """
    Build a ListedColormap and BoundaryNorm so ranks are *discrete* (not continuous).
    NaN shown as LIGHT_GREY.
    """
    k = min(10, max_rank_seen if max_rank_seen > 0 else 10)
    color_list = [RANK_COLORS[i] if i < len(RANK_COLORS) else GREY for i in range(1, k + 1)]
    cmap = mcolors.ListedColormap(color_list, name="rank_discrete")
    cmap.set_bad(NAN_COLOR)  # color for NaN / missing
    # boundaries from 0.5, 1.5, ..., k+0.5 so that integers map to bins cleanly
    bounds = np.arange(0.5, k + 0.6, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N, clip=False)
    ticks = list(range(1, k + 1))
    return cmap, ticks, norm

def discrete_cmap_from_continuous(
    cmap_name: str,
    n_levels: int,
):
    base = plt.get_cmap(cmap_name)
    colors = base(np.linspace(0, 1, n_levels))
    cmap = mcolors.ListedColormap(colors)

    bounds = np.arange(1, n_levels + 2) - 0.5
    ticks = list(range(1, n_levels + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, ticks, norm

def visualize_prefs(
    team_to_mentor,
    mentor_to_team,
    pair_type,
    max_rank_seen,
    matches=None,
    save_path=None
):

    # cmap, cbar_ticks, norm = _discrete_cmap_and_norm(max_rank_seen)
    cmap, cbar_ticks, norm = discrete_cmap_from_continuous("RdBu", max_rank_seen)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax_flat = ax.flatten()

    plt_titles = ['Team --> Mentor Rankings (1=high)','Mentor --> Team Rankings (1=high)']

    for i_tbl, pivot_tbl in enumerate([team_to_mentor, mentor_to_team]):
        sns.heatmap(
            pivot_tbl,
            annot=pivot_tbl,
            fmt='.0f',
            cmap=cmap,
            norm=norm,
            cbar_kws={'ticks': cbar_ticks, 'shrink': 0.75, 'label': 'Ranking'},
            ax=ax_flat[i_tbl],
            xticklabels=list(pivot_tbl.columns),
            yticklabels=(pivot_tbl.index)
        )

        ax_flat[i_tbl].set_xticklabels(
            ax_flat[i_tbl].get_xticklabels(), rotation=45, ha='right'
        )
        ax_flat[i_tbl].set_title(plt_titles[i_tbl])
        ax_flat[i_tbl].set_xlabel("")
        ax_flat[i_tbl].set_ylabel("")

        # ---- overlay chosen matches (optional) ----
        if matches is not None:
            for _, row in matches.iterrows():
                team = row["team"]
                mentor = row["mentor"]

                if i_tbl == 0:
                    # team_to_mentor: rows=team, cols=mentor
                    if team in pivot_tbl.index and mentor in pivot_tbl.columns:
                        i = pivot_tbl.index.get_loc(team)
                        j = pivot_tbl.columns.get_loc(mentor)
                        ax_flat[i_tbl].scatter(
                            j + 0.8, i + 0.8,
                            color="red", s=60, marker="o",
                            edgecolors="white", linewidths=1.0
                        )
                else:
                    # mentor_to_team: rows=mentor, cols=team
                    if mentor in pivot_tbl.index and team in pivot_tbl.columns:
                        i = pivot_tbl.index.get_loc(mentor)
                        j = pivot_tbl.columns.get_loc(team)
                        ax_flat[i_tbl].scatter(
                            j + 0.8, i + 0.8,
                            color="red", s=60, marker="o",
                            edgecolors="white", linewidths=1.0
                        )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        
    return fig, ax


def visualize_matches(cost, matches, save_path=None):    

    fig, ax = plt.subplots()

    sns.heatmap(
        cost,
        annot=cost,
        fmt='.2f',
        cmap='viridis',
        ax=ax
    )
    for _, row in matches.iterrows():
        i = cost.index.get_loc(row["team"])
        j = cost.columns.get_loc(row["mentor"])
        ax.scatter(j + 0.5, i + 0.5, color="red", s=50)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    
    return fig, ax


def cost_to_strength(c, c_min, c_max):
    # normalize so best cost = 1, worst = 0
    return (c_max - c) / (c_max - c_min)

def plot_chapter_graph(
    matches: pd.DataFrame,
    cost: pd.DataFrame,
    save_path: Optional[Path | str] = None,
):

    chosen = set(zip(matches["team"], matches["mentor"]))

    teams = cost.index.tolist()
    mentors = cost.columns.tolist()

    team_pos = {t: (0, i) for i, t in enumerate(teams)}
    mentor_pos = {m: (1, i) for i, m in enumerate(mentors)}

    # finite costs only
    cost_finite = cost.replace([np.inf, -np.inf], np.nan).stack().dropna()
    c_min, c_max = cost_finite.min(), cost_finite.max()

    # ---- edges ----
    edges = []
    for team in teams:
        for mentor in mentors:
            c = cost_finite.loc[team, mentor]
            if pd.isna(c):
                continue

            s = cost_to_strength(c, c_min, c_max)
            edges.append((team, mentor, s))

    # ---- node strengths ----
    team_strength = {t: 0.0 for t in teams}
    mentor_strength = {m: 0.0 for m in mentors}

    for t, m, s in edges:
        team_strength[t] += s
        mentor_strength[m] += s

    def scale(vals, min_size=50, max_size=300):
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            return [min_size] * len(vals)
        return [
            min_size + (v - vmin) / (vmax - vmin) * (max_size - min_size)
            for v in vals
        ]

    team_sizes = scale(team_strength.values())
    mentor_sizes = scale(mentor_strength.values())

    # ---- plot ----
    fig, ax = plt.subplots(
        figsize=(8, max(len(teams), len(mentors)) * 0.4)
    )

    ax.scatter(
        [0] * len(teams),
        range(len(teams)),
        s=team_sizes,
        color="tab:blue",
        marker="o",
        label="Teams",
        zorder=3,
    )

    ax.scatter(
        [1] * len(mentors),
        range(len(mentors)),
        s=mentor_sizes,
        color="tab:orange",
        marker="s",
        label="Mentors",
        zorder=3,
    )

    # ---- edges ----
    for team, mentor, strength in edges:
        lw = 0.5 + 2.5 * strength
        color = "red" if (team, mentor) in chosen else "black"

        ax.plot(
            [0, 1],
            [team_pos[team][1], mentor_pos[mentor][1]],
            linewidth=lw,
            color=color,
            alpha=0.8,
            zorder=1,
        )

    # ---- labels ----
    for team, (_, y) in team_pos.items():
        ax.text(-0.1, y, team, ha="right", va="center")

    for mentor, (_, y) in mentor_pos.items():
        ax.text(1.1, y, mentor, ha="left", va="center")

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-1, max(len(teams), len(mentors)))
    ax.axis("off")
    ax.legend(loc="upper center", ncol=2)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()