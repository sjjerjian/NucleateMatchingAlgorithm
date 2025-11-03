#!/usr/bin/env python3
"""
make_schedule.py  —  Greedy + Repair deterministic scheduler with dual labeled heatmaps

Purpose:
  Build per-chapter meeting schedules from shortlist files.

Input:
  --input   shortlist_from_editable.csv
  --chapter <chapter name>
  --out     output CSV path (07_schedule.csv)
  --rounds  (optional) default number of rounds
  --rounds-map  e.g. "Boston=6,Texas=4,Default=5"

Output files per chapter:
  - 07_schedule.csv
  - 07_schedule_wide.csv
  - 07_schedule_heatmap_inventors.png
  - 07_schedule_heatmap_contributors.png
"""

import os, argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------- Brand palette ----------------
WHITE = "#FFFFFF"
LIGHT_GREY = "#F9FAFC"
GREEN = "#13D6B0"
BLUE = "#38C7E8"
PURPLE = "#918AE1"
YELLOW = "#FFC162"
RED = "#FC827F"
GREY = "#E5E7EA"
BLACK = "#000000"
DARK_GREEN = "#045456"
LIGHT_GREEN = "#A6FBE3"
LIGHT_PURPLE = "#E3DEFF"

# ---------------- Category colors and labels ----------------
cat_colors = {
    "mutual": DARK_GREEN,
    "inv_pref": BLUE,
    "con_pref": LIGHT_GREEN,
    "backfilled": LIGHT_PURPLE,
}
cat_labels = {
    "mutual": "Mutual match",
    "inv_pref": "One-sided (Inventor)",
    "con_pref": "One-sided (Contributor)",
    "backfilled": "Backfilled",
}

# ---------------- Core scheduler ----------------
def schedule_pairs(df):
    """Greedy + repair deterministic scheduler."""
    reason_order = {"mutual": 1, "inv_pref": 2, "con_pref": 3, "backfilled": 4}
    df = df.copy()
    df["reason_order"] = df["reason"].map(reason_order).fillna(5)
    df["score"] = df.apply(
        lambda x: (x["mutual_rank_sum"] if pd.notnull(x["mutual_rank_sum"]) else 999)
                  + x["reason_order"] * 0.01,
        axis=1,
    )
    df = df.sort_values(["score", "inventor", "contributor"]).reset_index(drop=True)

    rounds = []
    # --- Greedy assignment ---
    for _, row in df.iterrows():
        inv, con = row["inventor"], row["contributor"]
        for rnd in rounds:
            if inv not in rnd["invs"] and con not in rnd["cons"]:
                rnd["pairs"].append(row)
                rnd["invs"].add(inv)
                rnd["cons"].add(con)
                break
        else:
            rounds.append({"pairs": [row], "invs": {inv}, "cons": {con}})

    # --- Repair step ---
    for i in range(1, len(rounds)):
        moved = []
        for row in list(rounds[i]["pairs"]):
            inv, con = row["inventor"], row["contributor"]
            for j in range(i):
                if inv not in rounds[j]["invs"] and con not in rounds[j]["cons"]:
                    rounds[j]["pairs"].append(row)
                    rounds[j]["invs"].add(inv)
                    rounds[j]["cons"].add(con)
                    rounds[i]["pairs"].remove(row)
                    moved.append(row)
                    break
        if moved:
            print(f"  ⮡ Moved {len(moved)} pairs from round {i+1} earlier")

    # --- Flatten results ---
    scheduled = []
    for ridx, rnd in enumerate(rounds, start=1):
        for row in rnd["pairs"]:
            rec = row.to_dict()
            rec["round"] = ridx
            scheduled.append(rec)
    return pd.DataFrame(scheduled).sort_values(["round", "inventor", "contributor"])


# ---------------- Visualization helpers ----------------
def _plot_heatmap(df, index_col, partner_col, chapter, out_path, ylabel):
    """Plot inventor/contributor heatmaps with partner name labels."""
    entities = sorted(df[index_col].unique())
    rounds = sorted(df["round"].unique())
    pivot_reason = pd.DataFrame(index=entities, columns=rounds)
    pivot_partner = pd.DataFrame(index=entities, columns=rounds)

    for _, r in df.iterrows():
        pivot_reason.loc[r[index_col], r["round"]] = r["reason"]
        pivot_partner.loc[r[index_col], r["round"]] = r[partner_col]

    pivot_reason = pivot_reason.fillna("")
    pivot_partner = pivot_partner.fillna("")

    cat_to_int = {k: i for i, k in enumerate(cat_colors.keys())}
    pivot_encoded = pivot_reason.replace(cat_to_int).infer_objects(copy=False)
    pivot_encoded = pivot_encoded.replace("", np.nan)
    data = pivot_encoded.to_numpy(dtype=float)

    cmap = ListedColormap(list(cat_colors.values()))
    cmap.set_bad(color=GREY)

    fig, ax = plt.subplots(figsize=(len(rounds)*0.8 + 4, len(entities)*0.25 + 3))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=len(cat_colors) - 1)

    # axis labels
    ax.set_xticks(np.arange(len(rounds)))
    ax.set_xticklabels(rounds, fontsize=9)
    ax.set_xlabel("Rounds", fontsize=11, color=BLACK)
    ax.set_yticks(np.arange(len(entities)))
    ax.set_yticklabels(entities, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11, color=BLACK)
    ax.set_title(f"{chapter} — Schedule Heatmap ({ylabel})", fontsize=12, weight="bold")

    # --- Add partner names as text labels ---
    for i, ent in enumerate(entities):
        for j, rnd in enumerate(rounds):
            partner = pivot_partner.iloc[i, j]
            if partner and not pd.isna(data[i, j]):
                ax.text(
                    j, i, str(partner),
                    ha="center", va="center",
                    fontsize=6, color="white" if data[i, j] in [0, 1] else "black",
                    wrap=True,
                )

    # --- Legend ---
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=cat_labels[c],
                   markerfacecolor=cat_colors[c], markersize=10)
        for c in cat_colors
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", title="Reason")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[{chapter}] Saved schedule heatmap → {out_path}")


def visualize_schedule(df, chapter, out_dir):
    """Generate inventor- and contributor-focused schedule heatmaps."""
    os.makedirs(out_dir, exist_ok=True)
    _plot_heatmap(df, "inventor", "contributor", chapter,
                  os.path.join(out_dir, "07_schedule_heatmap_inventors.png"),
                  "Inventors")
    _plot_heatmap(df, "contributor", "inventor", chapter,
                  os.path.join(out_dir, "07_schedule_heatmap_contributors.png"),
                  "Contributors")


# ---------------- Helpers ----------------
def parse_rounds_map(s, default_rounds):
    out = {}
    if not s:
        return out, default_rounds
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = int(v.strip())
        except:
            pass
    return out, default_rounds


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Deterministic Greedy+Repair scheduler")
    ap.add_argument("--input", required=True)
    ap.add_argument("--chapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--rounds-map",
                    help='Override per chapter, e.g. "Boston=6,Texas=4,Default=5"')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print(f"[{args.chapter}] No data in shortlist, skipping.")
        return

    rounds_map, default_rounds = parse_rounds_map(args.rounds_map, args.rounds)
    rounds_for_ch = rounds_map.get(args.chapter, rounds_map.get("Default", default_rounds))

    sched = schedule_pairs(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sched.to_csv(args.out, index=False)
    print(f"[{args.chapter}] Saved schedule → {args.out}")

    # --- wide output ---
    wide_blocks = []
    for rnd in sorted(sched["round"].unique()):
        blk = sched[sched["round"] == rnd][["inventor", "contributor"]].reset_index(drop=True)
        blk.columns = [f"round_{rnd}_inventor", f"round_{rnd}_contributor"]
        wide_blocks.append(blk)
    wide = pd.concat(wide_blocks, axis=1)
    wide_out = os.path.join(os.path.dirname(args.out), "07_schedule_wide.csv")
    wide.to_csv(wide_out, index=False)
    print(f"[{args.chapter}] Saved wide format → {wide_out}")

    # --- visualization ---
    visualize_schedule(sched, args.chapter, os.path.dirname(args.out))


if __name__ == "__main__":
    main()
