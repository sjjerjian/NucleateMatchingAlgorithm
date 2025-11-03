#!/usr/bin/env python3
"""
build_shortlist.py

Build shortlist from editable inventor/contributor matrices,
and visualize as a heatmap.

Inputs:
  --chapter CHAPTER
  --inv-matrix editable_01_inventor_prefs.csv
  --con-matrix editable_02_contrib_prefs.csv
  --out 05_shortlist_from_editable.csv

Outputs:
  <chapter>/05_shortlist_from_editable.csv
  <chapter>/06_shortlist_heatmap.png
"""

import csv, os, argparse, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

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
    DARK_RED,           # 2
    DARK_YELLOW,            # 3
    YELLOW,          # 4
    RED,          # 5
    GREEN,             # 6
    LIGHT_GREEN,      # 7
    LIGHT_RED,    # 8
    LIGHT_YELLOW,    # 9
    GREY,       # 10
]


# ---------- helpers ----------

def read_matrix(path):
    data = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return data
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        person = row[0].strip()
        ranks = {}
        for i, val in enumerate(row[1:], start=1):
            val = val.strip()
            if val:
                ranks[val] = i
        data[person] = ranks
    return data


def build_shortlist(ch, inv_rank, con_rank, inv_rank_org, con_rank_org):
    out = []
    pairs = set()
    for i, d in inv_rank.items():
        for c in d.keys():
            pairs.add((i, c))
    for c, d in con_rank.items():
        for i in d.keys():
            pairs.add((i, c))

    # Classify each pair
    for inv, con in sorted(pairs):
        r_i = inv_rank.get(inv, {}).get(con)
        r_c = con_rank.get(con, {}).get(inv)

        r_i_org = inv_rank_org.get(inv, {}).get(con)
        r_c_org = con_rank_org.get(con, {}).get(inv)

        # Determine reason
        if r_i_org is not None and r_c_org is not None:
            reason = "mutual"
        elif r_i_org is not None:
            reason = "inv_pref"
        elif r_c_org is not None:
            reason = "con_pref"
        else:
            reason = "backfilled"

        out.append({
            "chapter": ch,
            "inventor": inv,
            "contributor": con,
            "inventor_rank": r_i,
            "contributor_rank": r_c,
            "mutual_rank_sum": (r_i + r_c if (r_i is not None and r_c is not None) else ""),
            "reason": reason,
        })
            
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["chapter","inventor","contributor",
              "inventor_rank","contributor_rank",
              "mutual_rank_sum","reason"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------- visualization ----------

def _discrete_cmap_and_norm(max_rank_seen):
    colors = RANK_COLORS[1:max_rank_seen + 1]
    cmap = ListedColormap(colors)
    bounds = np.arange(0.5, max_rank_seen + 1.5)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, list(range(1, max_rank_seen + 1))


def visualize_shortlist(chapter, shortlist_csv, outdir):
    """
    Generate a heatmap of inventor x contributor using `reason` as color category.
    """
    df = pd.read_csv(shortlist_csv)
    if df.empty:
        print(f"[{chapter}] No shortlist data to visualize.")
        return

    # Only keep rows with valid reason
    df = df[df["reason"].notna() & (df["reason"].astype(str).str.strip() != "")]
    if df.empty:
        print(f"[{chapter}] No valid reasons to visualize.")
        return

    # Pivot table (inventor × contributor, value = reason)
    pivot = df.pivot_table(
        index="inventor", columns="contributor", values="reason", aggfunc="first"
    )

    if pivot.empty:
        print(f"[{chapter}] No valid pairs to visualize.")
        return

    # --- categorical color mapping ---
    categories = ["mutual", "inv_pref", "con_pref", "backfilled"]
    cat_colors = {
        "mutual": DARK_GREEN,
        "inv_pref": BLUE,
        "con_pref": LIGHT_GREEN,
        "backfilled": LIGHT_PURPLE
    }
    cat_labels = {
        "mutual": "Mutaual match",
        "inv_pref": "One-sided (Inventor)",
        "con_pref": "One_sided (Contributor)",
        "backfilled": "Backfilled"
    }
    colors = [cat_colors.get(c, "#E5E7EA") for c in categories]
    cmap = ListedColormap(colors)
    cat_to_int = {c: i for i, c in enumerate(categories)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        pivot_encoded = pivot.replace(cat_to_int).infer_objects(copy=False)
    data = pivot_encoded.to_numpy(dtype=float)

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(4, len(pivot.index)*0.2)))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=len(categories)-1)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Contributors", fontsize=11, color="#000000")
    ax.set_ylabel("Inventors", fontsize=11, color="#000000")

    ax.set_title(f"{chapter} - Shortlist Heatmap (by Reason)", fontsize=12, weight="bold")

    # Create custom legend (not numeric colorbar)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=cat_labels[c],
                   markerfacecolor=cat_colors[c], markersize=10)
        for c in categories
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", title="Reason")

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, "06_shortlist_heatmap.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[{chapter}] Saved shortlist reason heatmap → {out_png}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", required=True)
    ap.add_argument("--inv-matrix", required=True)
    ap.add_argument("--con-matrix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inv_rank = read_matrix(args.inv_matrix)
    con_rank = read_matrix(args.con_matrix)
    inv_rank_org = read_matrix(args.inv_matrix.replace("editable_", ""))
    con_rank_org = read_matrix(args.con_matrix.replace("editable_", ""))

    # inv_rank = read_matrix("output/example_realistic_ext/Boston/editable_01_inventor_prefs.csv")
    # con_rank = read_matrix("output/example_realistic_ext/Boston/editable_02_contrib_prefs.csv")
    # inv_rank_org = read_matrix("output/example_realistic_ext/Boston/inventor_prefs.csv")
    # con_rank_org = read_matrix("output/example_realistic_ext/Boston/contrib_prefs.csv")

    rows = build_shortlist(args.chapter, inv_rank, con_rank, inv_rank_org, con_rank_org)
    write_csv(args.out, rows)
    print(f"[{args.chapter}] Wrote shortlist: {args.out}")

    visualize_shortlist(args.chapter, args.out, os.path.dirname(args.out))


if __name__ == "__main__":
    main()
