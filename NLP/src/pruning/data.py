# src/pruning/data.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

SPLIT_DIR = Path("artifacts/splits")
CSV_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{}.csv"


def load_banking77() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Retourne (train, test, label_names). Colonnes : text, label (int)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("PolyAI/banking77")
        names = ds["train"].features["label"].names
        train, test = ds["train"].to_pandas(), ds["test"].to_pandas()
    except Exception:
        # Secours si l'erreur "dataset scripts are no longer supported" apparaît
        train, test = pd.read_csv(CSV_URL.format("train")), pd.read_csv(CSV_URL.format("test"))
        names = sorted(train["category"].unique())
        to_id = {n: i for i, n in enumerate(names)}
        for df in (train, test):
            df["label"] = df["category"].map(to_id)
        train, test = train[["text", "label"]], test[["text", "label"]]
    return train, test, names


def make_splits(train: pd.DataFrame, test: pd.DataFrame,
                val_per_class: int = 10, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Validation stratifiée tirée du train ; pool = reste du train ; test officiel intact."""
    train = train.reset_index(drop=True).copy()
    train["id"] = [f"tr{i:05d}" for i in range(len(train))]
    test = test.reset_index(drop=True).copy()
    test["id"] = [f"te{i:05d}" for i in range(len(test))]

    rng = np.random.default_rng(seed)
    val_idx = np.concatenate([
        rng.choice(g.index.to_numpy(), size=val_per_class, replace=False)
        for _, g in train.groupby("label")
    ])
    val = train.loc[val_idx]
    pool = train.drop(index=val_idx)
    return {"pool": pool, "val": val, "test": test}


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path = SPLIT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df[["id", "text", "label"]].to_json(out_dir / f"{name}.jsonl", orient="records", lines=True)


def load_split(name: str, split_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    """À utiliser partout ailleurs dans le pipeline."""
    return pd.read_json(split_dir / f"{name}.jsonl", lines=True)