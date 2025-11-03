#!/usr/bin/env python3
"""
run_match_pipeline.py

Lean, per-chapter pipeline (auto-rerun aware).

For each chapter found in --prefs:
  1) Rank matrices + heatmaps (or reuse if already present)
       <project>/<Chapter>/
         01_inventor_prefs.csv
         02_contrib_prefs.csv
         inventor_prefs.png
         contrib_prefs.png
         contributor_popularity.png
         inventor_popularity.png
     If BOTH CSVs already exist, they are not rewritten but just used for the next steps.

  2) Build an EDITABLE shortlist from the matrices
       <project>/<Chapter>/05_shortlist_from_editable.csv
       columns: chapter, inventor, contributor, inventor_rank, contributor_rank,
                mutual_rank_sum, reason

  3) Create the round schedule (greedy, best-first, one meeting/person/round)
       <project>/<Chapter>/07_schedule.csv
       columns: chapter, round, inventor, contributor, mutual_rank_sum,
                inventor_rank, contributor_rank, reason

Rounds:
  --rounds sets the default #rounds (also target shortlist size per inventor).
  You can override per chapter with --rounds-map like "Boston=6,Texas=4,Default=5".

Usage:
  python scripts/run_match_pipeline.py \
    --prefs input_data/10-22-25_export_Requested Connections.csv \
    --folder output/10-22-25_export \
    --rounds 5
"""

import os
import sys
import csv
import argparse
import subprocess
import shutil
import glob
import pandas as pd


# ---------------- helpers ----------------

def must_exist(path, label):
    if not os.path.isfile(path):
        sys.exit(f"Missing {label}: {path}")

def run(cmd):
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)

def list_chapters_from_input(path):
    chapters = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            # ignore commented rows (any cell beginning with '#')
            if any((v or "").strip().startswith("#") for v in row.values()):
                continue
            ch = (row.get("chapter") or "").strip()
            if ch:
                chapters.add(ch)
    return sorted(chapters)

def safe(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in (name or ""))

def parse_rounds_map(s: str, default_rounds: int):
    """
    Parse "Boston=6,Texas=4,Default=5" -> dict; use dict.get(ch, dict.get('Default', default_rounds)).
    """
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


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Per-chapter pipeline: prefs -> matrices/heatmaps (reuse if present) -> editable shortlist -> schedule"
    )
    # Inputs / locations
    ap.add_argument("--prefs", required=True,
                    help="Unified preferences CSV (chapter,requester,requestee,type,rank)")
    ap.add_argument("--folder", required=True,
                    help="Output project folder, e.g. output/example_run")
    ap.add_argument("--scripts-dir", default="scripts",
                    help="Directory with visualize_prefs.py, build_shortlist.py, make_schedule.py")
    ap.add_argument("--python", default="python",
                    help="Python executable (default: python)")

    # Rounds / shortlist size
    ap.add_argument("--rounds", type=int, default=5,
                    help="Default rounds per chapter (also target shortlist size per inventor).")
    ap.add_argument("--rounds-map",
                    help='Override rounds per chapter, e.g. "Boston=6,Texas=4,Default=5"')

    args = ap.parse_args()

    # Inputs & required scripts

    prefs_raw = args.prefs
    prefs = prefs_raw.replace(".csv", "_clean.csv")

    data_cleaner      = os.path.join(args.scripts_dir, "data_cleaner.py")
    v_prefs         = os.path.join(args.scripts_dir, "visualize_prefs.py")
    build_shortlist = os.path.join(args.scripts_dir, "build_shortlist.py")
    make_schedule   = os.path.join(args.scripts_dir, "make_schedule.py")

    must_exist(data_cleaner, "data_cleaner.py")
    must_exist(v_prefs, "visualize_prefs.py")
    # build_shortlist and make_schedule will be added one by one
    # so we only check when they exist
    if os.path.exists(build_shortlist):
        print("✅ build_shortlist.py detected")
    else:
        print("⚠️  build_shortlist.py not yet present (will skip shortlist step)")
    if os.path.exists(make_schedule):
        print("✅ make_schedule.py detected")
    else:
        print("⚠️  make_schedule.py not yet present (will skip schedule step)")

    print("Cleaning input data")

    if not os.path.exists(prefs):
        cmd = [args.python, data_cleaner, "--input", prefs_raw, "--output", prefs]
        run(cmd)
    
    must_exist(prefs, "preferences CSV")

    # Project root
    base_output = os.path.dirname(args.folder) or "."
    project_name = os.path.basename(args.folder.rstrip("/"))
    project_root = os.path.join(base_output, project_name)
    os.makedirs(project_root, exist_ok=True)

    # Chapters present in prefs
    chapters = list_chapters_from_input(prefs)
    if not chapters:
        sys.exit("No chapters found in --prefs")

    # Rounds map
    rounds_map, default_rounds = parse_rounds_map(args.rounds_map, args.rounds)

    # Banner
    print("\n==== Nucleate Matching (lean) ====")
    print(f"Project root: {project_root}")
    print(f"Chapters: {', '.join(chapters)}")
    print(f"Default rounds: {default_rounds}")
    if rounds_map:
        print(f"Rounds map: {rounds_map}")
    print("Auto-rerun mode: matrices are reused if their CSVs already exist.")
    print("==================================\n")


    for ch in chapters:
        ch_root = os.path.join(project_root, safe(ch))
        os.makedirs(ch_root, exist_ok=True)

        inv_matrix = os.path.join(ch_root, "01_inventor_prefs.csv")
        con_matrix = os.path.join(ch_root, "02_contrib_prefs.csv")

        # 1) Matrices (+heatmaps) or reuse
        if os.path.exists(inv_matrix) and os.path.exists(con_matrix):
            print(f"[{ch}] Matrices present → reusing")
        else:
            print(f"[{ch}] Building matrices + heatmaps …")
            cmd = [args.python, v_prefs, "--input", prefs, "--outdir", ch_root, "--chapter", ch]
            run(cmd)

        # 2) Ensure EDITABLE copies (users can modify these in place)
        editable_inv = os.path.join(ch_root, "editable_01_inventor_prefs.csv")
        editable_con = os.path.join(ch_root, "editable_02_contrib_prefs.csv")

        def _copy_if_missing(src, dst):
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
                print(f"[{ch}] Created editable: {os.path.relpath(dst)}")
            else:
                print(f"[{ch}] Editable present: {os.path.relpath(dst)}")

        _copy_if_missing(inv_matrix, editable_inv)
        _copy_if_missing(con_matrix, editable_con)

        # 3) Build shortlist *from editable files* (if script present)
        rounds_for_ch = rounds_map.get(ch, rounds_map.get("Default", default_rounds))
        shortlist_csv = os.path.join(ch_root, "05_shortlist_from_editable.csv")

        if os.path.exists(build_shortlist):
            print(f"[{ch}] Building shortlist from edited matrices (target per-inventor = {rounds_for_ch}) …")
            cmd = [
                args.python, build_shortlist,
                "--chapter", ch,
                "--inv-matrix", editable_inv,
                "--con-matrix", editable_con,
                "--out", shortlist_csv
            ]
            run(cmd)
        else:
            print(f"[{ch}] Skipping shortlist step (build_shortlist.py not found)")

        # 4) Schedule (if script present)
        schedule_csv = os.path.join(ch_root, "07_schedule.csv")
        if os.path.exists(make_schedule):
            print(f"[{ch}] Scheduling {rounds_for_ch} rounds …")
            cmd = [
                args.python, make_schedule,
                "--input", shortlist_csv,
                "--chapter", ch,
                "--rounds", str(rounds_for_ch),
                "--out", schedule_csv,
            ]
            run(cmd)
        else:
            print(f"[{ch}] Skipping schedule step (make_schedule.py not found)")

    print("\n✅ All chapters processed.... Aggregating")

    output_agg = os.path.join(project_root, "merged_matches.csv")

    out_files = sorted(glob.glob(f"{project_root}/*/07_schedule.csv"))
    if not out_files:
        raise SystemExit("No schedule files found.")

    dfs = [pd.read_csv(f) for f in out_files]
    agg = pd.concat(dfs, ignore_index=True)

    # --- Rename reason values ---
    reason_map = {
        "mutual": "Mutual",
        "inv_pref": "Inventor Preference",
        "con_pref": "Contributor Preference",
        "backfilled": "Backfilled"
    }
    if "reason" in agg.columns:
        agg["reason"] = agg["reason"].replace(reason_map)


    agg.to_csv(output_agg, index=False)
    print(f"✅ Combined {len(out_files)} schedule files → {output_agg}")

if __name__ == "__main__":
    main()
