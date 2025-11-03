#!/usr/bin/env python3
"""
visualize_prefs.py

Create per-chapter visualizations from the unified preferences CSV.

Input CSV (one file):
  chapter, requester, requestee, type, rank
  - type = "inventor"     -> requester is an inventor ranking a contributor
  - type = "contributor"  -> requester is a contributor ranking an inventor
  - rank: integer, 1 = best

Outputs (default: grouped by chapter):
  <outdir>/<chapter>/
    - 01_inventor_prefs.csv   (rows inventors x cols rank_1..rank_K; values = contributor names)
    - 02_contrib_prefs.csv    (rows contributors x cols rank_1..rank_K; values = inventor names)
    - 01_inventor_pref_matrix.png   (Inventor→Contributor rank heatmap; discrete colors)
    - 02_contrib_pref_matrix.png    (Contributor→Inventor rank heatmap; discrete colors)
    - 04_contributor_popularity.png (#inventors who ranked each contributor)
    - 03_inventor_popularity.png    (#contributors who ranked each inventor)

Single-chapter mode (write directly into --outdir):
  python visualize_prefs.py --input prefs.csv --outdir <chapter_viz_dir> --chapter Boston
"""

import argparse
import csv
import os
import collections

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colormaps

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

# ----------------------------------------------------------------

def safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in (name or ""))

def load_chapter_rows(path, chapter):
    """
    Read unified CSV, skip commented rows, coerce rank to int.
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        need = {"chapter", "requester", "requestee", "type", "rank"}
        if not need.issubset({(c or "").strip() for c in (r.fieldnames or [])}):
            raise SystemExit("Input CSV must include columns: chapter, requester, requestee, type, rank")
        for row in r:
            # Skip commented rows (if any cell begins with '#')
            if any((v or "").strip().startswith("#") for v in row.values()):
                continue
            row = {k: (v or "").strip() for k, v in row.items()}
            if not row["chapter"] or not row["requester"] or not row["requestee"] or not row["type"] or not row["rank"]:
                continue
            if row["chapter"] == chapter:
                try:
                    row["rank"] = int(row["rank"])
                except:
                    continue
                rows.append(row)
    return rows


def build_rank_maps(chapter_rows):
    """
    Return:
      invs, cons, inv_rank, con_rank
      inv_rank[inv][con] = rank (int)
      con_rank[con][inv] = rank (int)
    """
    invs, cons = set(), set()
    inv_rank = collections.defaultdict(dict)
    con_rank = collections.defaultdict(dict)
    for r in chapter_rows:
        typ = r["type"].lower()
        req, tgt, rk = r["requester"], r["requestee"], r["rank"]
        if typ == "inventor":
            invs.add(req); cons.add(tgt)
            inv_rank[req][tgt] = rk
        elif typ == "contributor":
            cons.add(req); invs.add(tgt)
            con_rank[req][tgt] = rk
    invs = sorted(invs)
    cons = sorted(cons)
    return invs, cons, inv_rank, con_rank

def write_rank_table_csv(path, row_labels, rank_map, who_is_rows="inventor"):
    """
    For each row subject, produce columns rank_1..rank_K (K up to max 10 or observed).
    Cell = the partner ID who is at that rank (empty if none).
    """
    # Find max rank observed (cap at 10)
    max_rank = 0
    for s in row_labels:
        if s in rank_map:
            if rank_map[s]:
                max_rank = max(max_rank, max(rank_map[s].values()))
    max_rank = min(10, max_rank if max_rank > 0 else 10)
    header = [""] + [f"{i}" for i in range(1, max_rank + 1)]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in row_labels:
            row = [s]
            inverse = {}  # rank -> partner
            for partner, rk in rank_map.get(s, {}).items():
                if 1 <= rk <= 10:
                    # assuming strict ranks (no ties)
                    inverse[rk] = partner
            for i in range(1, max_rank + 1):
                row.append(inverse.get(i, ""))
            w.writerow(row)

def _discrete_cmap_and_norm(max_rank_seen):
    """
    Build a ListedColormap and BoundaryNorm so ranks are *discrete* (not continuous).
    NaN shown as LIGHT_GREY.
    """
    k = min(10, max_rank_seen if max_rank_seen > 0 else 10)
    colors = [RANK_COLORS[i] if i < len(RANK_COLORS) else GREY for i in range(1, k + 1)]
    cmap = ListedColormap(colors, name="rank_discrete")
    cmap.set_bad(NAN_COLOR)  # color for NaN / missing
    # boundaries from 0.5, 1.5, ..., k+0.5 so that integers map to bins cleanly
    bounds = np.arange(0.5, k + 1.6, 1.0)
    # norm = BoundaryNorm(bounds, cmap.N, clip=False)
    ticks = list(range(1, k + 1))
    return cmap, ticks

def plot_rank_heatmap(path, rows_axis, cols_axis, get_rank, title, xlabel, ylabel):
    """
    rows_axis: list[str] (y-axis)
    cols_axis: list[str] (x-axis)
    get_rank(r, c) -> int or None
    """
    if not rows_axis or not cols_axis:
        return

    mat = np.full((len(rows_axis), len(cols_axis)), np.nan, dtype=float)
    max_rank_seen = 0
    for i, r in enumerate(rows_axis):
        for j, c in enumerate(cols_axis):
            rk = get_rank(r, c)
            if rk is not None:
                mat[i, j] = float(rk)
                if rk > max_rank_seen:
                    max_rank_seen = rk

    if max_rank_seen == 0:  # nothing to draw
        return

    cmap, ticks = _discrete_cmap_and_norm(max_rank_seen)

    plt.figure(figsize=(max(6, len(cols_axis) * 0.45), max(4, len(rows_axis) * 0.38)))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)
    plt.title(title, color=BLACK)
    plt.xlabel(xlabel, color=BLACK)
    plt.ylabel(ylabel, color=BLACK)
    plt.xticks(range(len(cols_axis)), cols_axis, rotation=90, color=BLACK)
    plt.yticks(range(len(rows_axis)), rows_axis, color=BLACK)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, ticks=ticks)
    cbar.ax.set_ylabel("Rank", rotation=90, va="center", color=BLACK)
    cbar.ax.tick_params(labelcolor=BLACK)

    # clean white background
    ax = plt.gca()
    ax.set_facecolor(WHITE)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def plot_bar_counts(path, labels, counts, title, xlabel, ylabel):
    if not labels:
        return
    plt.figure(figsize=(max(6, len(labels) * 0.35), 4))
    plt.bar(range(len(labels)), counts, color=DARK_GREEN, edgecolor=WHITE, linewidth=0.5)
    plt.xticks(range(len(labels)), labels, rotation=90, color=BLACK)
    plt.yticks(color=BLACK)
    plt.title(title, color=BLACK)
    plt.xlabel(xlabel, color=BLACK)
    plt.ylabel(ylabel, color=BLACK)
    plt.gca().set_facecolor(WHITE)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def render_chapter_files(outdir, chapter, ch_rows):
    invs, cons, inv_rank, con_rank = build_rank_maps(ch_rows)
    os.makedirs(outdir, exist_ok=True)

    # 1) CSV: Rank-order tables (names in cells)
    write_rank_table_csv(
        os.path.join(outdir, "01_inventor_prefs.csv"),
        invs, inv_rank, who_is_rows="inventor"
    )
    write_rank_table_csv(
        os.path.join(outdir, "02_contrib_prefs.csv"),
        cons, con_rank, who_is_rows="contributor"
    )

    # 2) PNG: Discrete rank heatmaps (not continuous)
    # Inventor → Contributor (rows=inventors, cols=contributors)
    plot_rank_heatmap(
        os.path.join(outdir, "01_inventor_pref_matrix.png"),
        invs, cons,
        lambda inv, con: inv_rank.get(inv, {}).get(con),
        title=f"{chapter} — Inventor → Contributor ranks",
        xlabel="Contributors", ylabel="Inventors"
    )
    # Contributor → Inventor (rows=contributors, cols=inventors)
    plot_rank_heatmap(
        os.path.join(outdir, "02_contrib_pref_matrix.png"),
        cons, invs,
        lambda con, inv: con_rank.get(con, {}).get(inv),
        title=f"{chapter} — Contributor → Inventor ranks",
        xlabel="Inventors", ylabel="Contributors"
    )

    # 3) Popularity bars
    # # inventors who ranked each contributor
    contrib_pop = [sum(1 for inv in invs if inv_rank.get(inv, {}).get(c) is not None) for c in cons]
    plot_bar_counts(
        os.path.join(outdir, "04_contributor_popularity.png"),
        cons, contrib_pop, f"{chapter} — # inventors ranking each contributor",
        "Contributor", "# inventors"
    )
    # # contributors who ranked each inventor
    inventor_pop = [sum(1 for con in cons if con_rank.get(con, {}).get(inv) is not None) for inv in invs]
    plot_bar_counts(
        os.path.join(outdir, "03_inventor_popularity.png"),
        invs, inventor_pop, f"{chapter} — # contributors ranking each inventor",
        "Inventor", "# contributors"
    )

def main():
    ap = argparse.ArgumentParser(description="Visualize chapter preferences from unified CSV (discrete ranks).")
    ap.add_argument("--input", required=True, help="Unified CSV: chapter, requester, requestee, type, rank")
    ap.add_argument("--outdir", required=True, help="Output root directory")
    ap.add_argument("--chapter", required=True, help="If set, visualize only this chapter and write files directly into --outdir")
    args = ap.parse_args()

    # input = "input_data/realistic_prefs.csv"
    # rows = load_chapter_rows(input)
    ch_rows = load_chapter_rows(args.input, args.chapter)
    render_chapter_files(args.outdir, args.chapter, ch_rows)
    return

if __name__ == "__main__":
    main()
