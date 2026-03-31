import csv
import glob
import json
import os

FOLDER = "experiments/classical_ml"
OUTPUT = os.path.join(FOLDER, "results_table.csv")

CLASSIFIER_PREFIXES = [
    "classical_ml_linear_svm_",
    "classical_ml_logreg_",
]

rows = []
for path in sorted(glob.glob(os.path.join(FOLDER, "*.json"))):
    with open(path) as f:
        data = json.load(f)
    stem = os.path.splitext(os.path.basename(path))[0]
    backbone = stem
    for prefix in CLASSIFIER_PREFIXES:
        if stem.startswith(prefix):
            backbone = stem[len(prefix):]
            break
    rows.append({
        "backbone": backbone,
        "classifier": data["classifier"],
        "val_acc": data["val_acc"],
        "test_acc": data["test_acc"],
        "feature_time_sec": round(data["feature_time_sec"], 1),
        "train_time_sec": round(data["train_time_sec"], 1),
    })

rows.sort(key=lambda r: (r["backbone"], r["classifier"]))

COLUMNS = ["backbone", "classifier", "val_acc", "test_acc", "feature_time_sec", "train_time_sec"]

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

# Pretty-print the table
col_widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in COLUMNS}
header = "  ".join(c.ljust(col_widths[c]) for c in COLUMNS)
sep = "  ".join("-" * col_widths[c] for c in COLUMNS)
print(header)
print(sep)
for r in rows:
    print("  ".join(str(r[c]).ljust(col_widths[c]) for c in COLUMNS))

print(f"\nSaved to {OUTPUT}")
