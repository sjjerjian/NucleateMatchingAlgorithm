#!/usr/bin/env python3
"""
data_cleaner.py

Cleans Softr-exported matching preference data for the pipeline.

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

import argparse
import pandas as pd
import numpy as np
import os

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


def flip_type(t: str) -> str:
    """Flip Inventor/Lead ↔ Contributor → inventor/contributor"""
    if not isinstance(t, str):
        return ""
    t = t.strip().lower()
    if "inventor" in t or "lead" in t:
        return "contributor"
    elif "contributor" in t:
        return "inventor"
    else:
        return t  # leave as-is if unknown


# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser(description="Clean-up Softr-exported data for pipeline")
    ap.add_argument("--input", required=True,
                    help="Input Softr export CSV")
    ap.add_argument("--output", required=True,
                    help="Cleaned unified preferences CSV (chapter,requester,requestee,type,rank)")
    args = ap.parse_args()

    input_path = args.input
    output_path = args.output

    print(f"📥 Reading {input_path} ...")
    raw_df = pd.read_csv(input_path)

    # Map Softr columns to pipeline columns
    colmap = {
        "Rollup_Invited Semi Final Chapter (from Requested connections Activator Semi Finalist)": "chapter",
        "Lookup_ Name of requestor": "requester",
        "Member_Name_Requestee": "requestee",
        "Application Track Requestee": "type",
        "Ranking": "rank"
    }

    missing_cols = [c for c in colmap if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing expected columns in input: {missing_cols}")

    df = raw_df[list(colmap.keys())].rename(columns=colmap)

    # Flip the 'type'
    df["type"] = df["type"].apply(flip_type)

    # Clean per requester
    cleaned_groups = []
    for keys, grp in df.groupby(["chapter", "requester", "type"], sort=False):
        cleaned_groups.append(fix_ranks(grp))
    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)

    cleaned_df = cleaned_df.sort_values(["chapter", "requester", "rank"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data written to {output_path}")
    print(f"🔢 {len(cleaned_df)} total rows processed.")


if __name__ == "__main__":
    main()
