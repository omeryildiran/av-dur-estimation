"""
Replaces participant initials with anonymous IDs (P01, P02, ...) across:
  - participantID columns in all CSV files that contain one
  - filenames in model_fits/, bootstrapped_params/, model_recovery_results/

Saves a PRIVATE mapping file (participant_id_mapping.txt) — do NOT commit this.
Run once before git add, then delete the script.
"""

import os
import glob
import json
import pandas as pd

# ── 1. Define mapping (initials → anonymous ID) ───────────────────────────────
# Covers all variants seen across experiments (case-insensitive key lookup).
# ln1 and ln2 are treated as two separate participants.

MAPPING = {
    "as":  "P01",
    "dt":  "P02",
    "hh":  "P03",
    "ip":  "P04",
    "lc":  "P05",
    "ln":   "P06",   # generic LN in unimodal files
    "ln1":  "P06",   # same person as LN in main exp
    "ln01": "P06",   # alternate format in some result files
    "ln2":  "P07",
    "mh":  "P08",
    "ml":  "P09",
    "mt":  "P10",
    "oy":  "P11",
    "qs":  "P12",
    "sx":  "P13",
}

# ── 2. Save private mapping ───────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

mapping_path = os.path.join(REPO_ROOT, "participant_id_mapping.txt")
with open(mapping_path, "w") as f:
    f.write("PRIVATE — do not commit\n\n")
    f.write(f"{'Initials':<12} {'Anonymous ID'}\n")
    f.write("-" * 25 + "\n")
    for initials, pid in sorted(MAPPING.items(), key=lambda x: x[1]):
        f.write(f"{initials:<12} {pid}\n")
print(f"Saved private mapping → {mapping_path}")

# ── helpers ───────────────────────────────────────────────────────────────────

def anonymize_id(value):
    """Replace a participantID value with its anonymous equivalent."""
    if pd.isna(value):
        return value
    return MAPPING.get(str(value).lower().strip(), value)  # leave unknown IDs unchanged


def anonymize_csv(path):
    df = pd.read_csv(path)
    if "participantID" not in df.columns:
        return
    before = df["participantID"].unique().tolist()
    df["participantID"] = df["participantID"].apply(anonymize_id)
    after = df["participantID"].unique().tolist()
    df.to_csv(path, index=False)
    print(f"  CSV: {os.path.relpath(path, REPO_ROOT)}")
    print(f"       {before} → {after}")


def anonymize_filename(path):
    """Rename a file if its basename starts with a known initials prefix."""
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    # Match prefix like "as_", "HH_", "ln1_", "ln2_" etc.
    for initials, pid in MAPPING.items():
        prefix = initials + "_"
        if basename.lower().startswith(prefix):
            new_basename = pid + "_" + basename[len(prefix):]
            new_path = os.path.join(dirname, new_basename)
            os.rename(path, new_path)
            print(f"  RENAME: {basename} → {new_basename}")
            return
    # No match — leave unchanged
    print(f"  (unchanged) {basename}")

# ── 3. Anonymize CSV files ────────────────────────────────────────────────────

print("\n── Anonymizing CSV columns ──────────────────────────────────────")

csv_targets = [
    "all_crossmodal.csv",
    "fitted_parameters_all_models.csv",
    "best_models_by_participant.csv",
    "model_comparison_all_results.csv",
    "pse_summary_statistics.csv",
    "parameter_summary_table.csv",
]

# Files that live inside data/
csv_targets_in_data = [
    "all_main.csv",
    "all_auditory.csv",
    "all_visual.csv",
    "all_woBiasedParticipants.csv",
    "all_wo_ln1.csv",
]

for name in csv_targets:
    path = os.path.join(REPO_ROOT, name)
    if os.path.exists(path):
        anonymize_csv(path)
    else:
        print(f"  (not found, skipping) {name}")

for name in csv_targets_in_data:
    path = os.path.join(REPO_ROOT, "data", name)
    if os.path.exists(path):
        anonymize_csv(path)
    else:
        print(f"  (not found, skipping) data/{name}")

# Also catch any CSVs inside data/ that have a participantID column
for path in glob.glob(os.path.join(REPO_ROOT, "data", "*.csv")):
    anonymize_csv(path)

# ── 4. Rename files in results directories ───────────────────────────────────

print("\n── Renaming files in model_fits/ ────────────────────────────────")
for path in sorted(glob.glob(os.path.join(REPO_ROOT, "model_fits", "*", "*.json"))):
    anonymize_filename(path)

print("\n── Renaming files in bootstrapped_params/ ───────────────────────")
for path in sorted(glob.glob(os.path.join(REPO_ROOT, "bootstrapped_params", "*", "*.json"))):
    anonymize_filename(path)
# Also rename the participant subdirectories themselves
for subdir in sorted(glob.glob(os.path.join(REPO_ROOT, "bootstrapped_params", "*"))):
    if os.path.isdir(subdir):
        dirname = os.path.dirname(subdir)
        basename = os.path.basename(subdir)
        for initials, pid in MAPPING.items():
            if basename.lower() == initials:
                new_path = os.path.join(dirname, pid)
                os.rename(subdir, new_path)
                print(f"  DIR: {basename} → {pid}")
                break

print("\n── Renaming files in model_recovery_results/ ────────────────────")
for path in sorted(glob.glob(os.path.join(REPO_ROOT, "model_recovery_results", "*.json"))):
    anonymize_filename(path)

print("\n── Renaming files in model_fits/ subdirs ────────────────────────")
for subdir in sorted(glob.glob(os.path.join(REPO_ROOT, "model_fits", "*"))):
    if os.path.isdir(subdir):
        dirname = os.path.dirname(subdir)
        basename = os.path.basename(subdir)
        for initials, pid in MAPPING.items():
            if basename.lower() == initials:
                new_path = os.path.join(dirname, pid)
                os.rename(subdir, new_path)
                print(f"  DIR: {basename} → {pid}")
                break

print("\nDone. Remember to:")
print("  1. Add participant_id_mapping.txt to .gitignore")
print("  2. Delete anonymize_participants.py before committing")
