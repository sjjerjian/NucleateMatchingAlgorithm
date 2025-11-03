
# MatchingAlgorithm

A small toolkit and pipeline I developed for [Nucleate](nucleate.org) for generating meeting/interview matches, shortlists, and schedules from preference and connection data. This repository contains data-cleaning utilities (tailored to export format from our internal databases), scripts to build shortlists and schedules, connectors for Airtable, and visualization helpers used during Nucleate matching workflows.

## Key ideas

- Input: exported or collected preference/connection CSVs (see `input_data/` and `output/input_data/`).
- Pipeline: clean and transform inputs, build shortlists and matches, generate schedules and visualizations.
- Output: per-event folders under `output/` with matches, shortlists, and schedule CSVs.

## Repo layout (important files)

- `environment.yml` - conda environment to reproduce the runtime environment.
- `Airtable_token.txt` - (optional) contains Airtable API token used by `Collect_from_Airtable.py`.
- `scripts/` - main scripts used by the pipeline:
	- `Collect_from_Airtable.py` - fetch data from Airtable (requires token).
	- `data_cleaner.py` - clean and normalize raw exports.
	- `build_shortlist.py` - create initial shortlists from cleaned prefs.
	- `run_match_pipeline.py` - orchestrates end-to-end matching (clean → shortlist → match → schedule).
	- `make_schedule.py` - take matches/shortlists and convert to schedules.
	- `visualize_prefs.py`, `visualize_matches.py` - simple plotting/visualization helpers.
- `input_data/` - example or exported CSVs used as inputs (multiple snapshots included).
- `output/` - generated matches, shortlists, and schedules. Subfolders organize by event and export date.

## Quickstart — setup

We recommend using conda to create the environment described in `environment.yml`.

1. Create the environment (macOS / Linux / WSL):

```bash
conda env create -f environment.yml -n matching
conda activate matching
```

If you prefer pip/venv, create a virtual environment and install the packages listed in `environment.yml` manually.

2. (Optional) If you will fetch data from Airtable, put your API token in `Airtable_token.txt` (single line, no newline padding issues). The scripts expect that file in the repository root. {this functionality was a WIP not implemented yet}

## Usage examples

All examples assume you're at the repository root and have the environment activated.

- Run the full pipeline (clean → shortlist → match → schedule):

```bash
python scripts/run_match_pipeline.py --prefs input_data/realistic_prefs.csv --folder output/input_data/realistic_prefs
```
<!-- 
- If you only need to build shortlists from cleaned preference CSVs:

```bash
python scripts/build_shortlist.py --input input_data/realistic_prefs_clean.csv --output output/example_shortlist/
```

- Generate schedule from matches:

```bash
python scripts/make_schedule.py --matches output/10-22-25_export/merged_matches.csv --outdir output/10-22-25_export/schedules/
```

- Collect data from Airtable (requires `Airtable_token.txt`):

```bash
python scripts/Collect_from_Airtable.py --base BASE_ID --table TableName --out input_data/airtable_export.csv
```

See each script's `--help` for available arguments, e.g. `python scripts/build_shortlist.py --help`. -->

## Input & output conventions

- Inputs are CSVs with preference or connection data. Several sample exports live in `input_data/` with names including export dates.
- Outputs are written to `output/` and organized by export date and event name. Look for files like `merged_matches.csv`, `interview_shortlist.csv`, and `07_schedule.csv`.


## Troubleshooting

- Missing dependencies: use `conda env update -f environment.yml` or inspect `environment.yml` for required packages.
- Incorrect CSV columns: the cleaning scripts attempt to normalize column names, but if upstream exports change you may need to adapt `data_cleaner.py` to match the new field names.

## Tests & validation

Example data used so far is included in input_data. It demonstrates a full export layout (inventor/contrib prefs, editable files, schedules per location).

This repo currently doesn't include an automated test suite. If you add functions that should be tested, add `pytest` tests under a new `tests/` folder and include them in CI.

## Contributing

If you'd like to contribute:

1. Open an issue describing the change or bug.
2. Create a branch for your change and open a PR with tests where appropriate.

## License & contact

Add license metadata later if you planning to fully open-source the repository. For internal use only.

