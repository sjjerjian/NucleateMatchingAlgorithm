#!/usr/bin/env python3
"""
data_cleaner_mentor_team.py

Cleans Softr-exported mentor matching preference data for the pipeline.

Steps:
  1. Select relevant columns and rename to: chapter, requester, requestee, type, rank
  2. Flip 'type':
        if 'Inventor/Lead' → 'contributor'
        if 'Contributor'   → 'inventor'
  3. Fix ranks per (chapter, requester, type):
        - If ranks are missing, assign random consecutive ranks starting from 1.
        - If ranks skip numbers (e.g. 1,2,4,5) → make them consecutive 1..N.
  4. Output cleaned file ready for run_match_pipeline.py.
"""

import os
import argparse

import numpy as np
import pandas as pd

# ------------------ Helpers ------------------

def fix_ranks(group: pd.DataFrame) -> pd.DataFrame:
    """Fix missing and skipped ranks for a given requester group."""
    g = group.copy()

    # Convert to numeric, ignore non-numeric values
    g["rank"] = pd.to_numeric(g["rank"], errors="coerce")

    if g["rank"].isna().all():
        # Assign random ranks deterministically (fixed seed)
        g = g.sample(frac=1, random_state=42).reset_index(drop=True)
        g["rank"] = range(1, len(g) + 1)
        return g

    # Sort by rank (ignoring NaN)
    g = g.sort_values("rank", na_position="last").reset_index(drop=True)

    # Replace NaNs at end with consecutive ranks continuing from max seen
    existing = g["rank"].dropna().astype(int).tolist()
    if existing:
        filled = list(range(1, len(g) + 1))
        g["rank"] = filled
    else:
        g["rank"] = range(1, len(g) + 1)

    return g



# ------------------ Main ------------------

def main():
    parser = argparse.ArgumentParser(description="Clean-up Softr-exported data for pipeline")
    parser.add_argument("--input", required=True,
                    help="Input exported CSV")
    parser.add_argument("--output", required=True,
                    help="Cleaned unified preferences CSV (chapter,requester,requestee,type,rank)")
    args = parser.parse_args()

    print(f"📥 Reading {args.input} ...")
    raw_df = pd.read_csv(args.input)

    # corresponding columns with different names depending on mentor or team
    mentor_team_cols = {
        # 'chapter': ('team_chapter', 'Mentor Chapter'),
        'requester': ('00_Team_Name (from Team requesting)', 'Lookup_Full Name'), # who was the requester?
        'requestee': ('Lookup_Full Name', 'Team ranked')
    }

    # Construct unified columns: chapter, requester, requestee, type, rank
    df = raw_df.rename(
    columns={
        'Type': 'type',
        'Ranking': 'rank',
        'chapter_coalesced': 'chapter',
        }
    )[['type', 'rank', 'chapter']]

    for k, (team_col, mentor_col) in mentor_team_cols.items():
        df[k] = np.where(
            df['type'] == 'Mentor',
            raw_df[mentor_col],
            raw_df[team_col]
        )
            
    df = df[['chapter', 'requester', 'requestee', 'type', 'rank']]
    df = df.dropna(subset=['chapter','requester','requestee','type'])
    # Clean per requester
    cleaned_groups = []
    for keys, grp in df.groupby(["chapter", "requester", "type"], sort=False):
        cleaned_groups.append(fix_ranks(grp))
    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)

    cleaned_df = cleaned_df.sort_values(["chapter", "requester", "rank"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cleaned_df.to_csv(args.output, index=False)
    print(f"✅ Cleaned data written to {args.output}")
    print(f"🔢 {len(cleaned_df)} total rows processed.")


if __name__ == "__main__":
    main()
