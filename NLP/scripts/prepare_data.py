# scripts/00_prepare_data.py
import json
from pruning.data import load_banking77, make_splits, save_splits, SPLIT_DIR

train, test, names = load_banking77()
splits = make_splits(train, test, val_per_class=10, seed=0)
save_splits(splits)
(SPLIT_DIR / "label_names.json").write_text(json.dumps(names, indent=2))

for k, df in splits.items():
    print(f"{k}: {len(df)} exemples, {df['label'].nunique()} classes")