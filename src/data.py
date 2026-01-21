"""
Docstring kontrolujący, przygotowujący dataset z HF pod dane gotowe po ML;
Izoluje logikę danych poza notebookami, zapewniając powtarzalność klas dla split'ów , które są zapisywane lokalnie;
Chciałem sobie ułatwić pracę, trzymać się wytycznych i utrzymać minimalną wagę repo.

"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


# const seed zapewniający powtarzalność danych
RANDOM_SEED = 67

# klasas sprawdza, czy dataset ściągniety z HF posiada wymagane kolumny 'question', 'student_answer" oraz "score" + obsłuyga błędów
@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "nkazi/MohlerASAG"
    split_name: str = "train"
    sample_size: int = 200
    text_sep: str = " [SEP] "
    q_col: str = "question"
    a_col: str = "student_answer"
    score_col_hint: str = "score_avg"  # automatycznie wykrywanie brakujących kolumny


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)

# obsługa wypadku braku kolumny "score"
def infer_score_column(df: pd.DataFrame, hint: str) -> str:
    if hint in df.columns:
        return hint
    candidates = [c for c in df.columns if "score" in c.lower()]
    if not candidates:
        raise ValueError(f"Nie znaleziono kolumny ze score. Kolumny: {list(df.columns)}")
    # ustawienie preferencji avg/mean
    for c in candidates:
        if "avg" in c.lower() or "mean" in c.lower():
            return c
    return candidates[0]

# pobieranie surowego datasetu
def load_raw_df(cfg: DataConfig) -> Tuple[pd.DataFrame, str]:
    ds: DatasetDict = load_dataset(cfg.dataset_name)
    split = cfg.split_name if cfg.split_name in ds else list(ds.keys())[0]
    d: Dataset = ds[split]
    df = d.to_pandas()

    missing = {cfg.q_col, cfg.a_col} - set(df.columns)
    if missing:
        raise ValueError(f"Brakuje kolumn: {missing}. Kolumny: {list(df.columns)}")

    score_col = infer_score_column(df, cfg.score_col_hint)
    return df, score_col

# obróbka ciągu dla dataframe z ustalonym separatorem
def build_input_text(df: pd.DataFrame, cfg: DataConfig) -> pd.Series:
    q = df[cfg.q_col].fillna("").astype(str)
    a = df[cfg.a_col].fillna("").astype(str)
    return q + cfg.text_sep + a


""" ustawienie progów na kwantylach nie zadziałało,
    więc zmieniłem klasę na deterministyczny binning pod score"""
# def make_labels_quantile(score: pd.Series) -> Tuple[pd.Series, Dict[int, str]]:
def make_labels_fixed(score: pd.Series):
    # q1 = float(s.quantile(1 / 3))
    # q2 = float(s.quantile(2 / 3))

    s = pd.to_numeric(score, errors="coerce")

    def to_class(v: float) -> int:
    #     if v <= q1:
    #         return 0
    #     if v <= q2:
    #         return 1
    #     return 2
        if v <= 2.99:
            return 0 #low
        elif v < 4 :
            return 1 #mid
        else:
            return 2 #high 

    # y = pd.to_numeric(score, errors="coerce").apply(to_class)
    y = s.apply(to_class)
    id2label = {0: "low", 1: "mid", 2: "high"}
    return y, id2label

# split danych 80% treningowe, 10% walidacyjne, 10% testowe
def stratified_split(df: pd.DataFrame, label_col: str = "label", seed: int = RANDOM_SEED):
    set_seed(seed)
    parts = {"train": [], "val": [], "test": []}

    # cls potrzebny tutaj dla tuple przy inicjalizacji
    for cls, grp in df.groupby(label_col):
        grp = grp.sample(frac=1.0, random_state=seed)
        n = len(grp)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        parts["train"].append(grp.iloc[:n_train])
        parts["val"].append(grp.iloc[n_train:n_train + n_val])
        parts["test"].append(grp.iloc[n_train + n_val:])

    out = {}
    for k in ["train", "val", "test"]:
        out[k] = pd.concat(parts[k]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out["train"], out["val"], out["test"]

# zapisz próbkę danych do ustalonego folderu i pliku
def save_sample(df: pd.DataFrame, path: Path, n: int, seed: int = RANDOM_SEED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sample(n=min(n, len(df)), random_state=seed).to_csv(path, index=False, encoding="utf-8")

# zapisz split
def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)


# init, wystarczy zmienić nazwę datasetu i ilość danych, żeby otrzymać określone dane, default na HF z MohlerASAG, 200 próbek
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nkazi/MohlerASAG")
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()

    cfg = DataConfig(dataset_name=args.dataset, sample_size=args.sample_size)

    raw_df, score_col = load_raw_df(cfg)
    raw_df["input_text"] = build_input_text(raw_df, cfg)

    raw_df["label"], id2label = make_labels_fixed(raw_df[score_col])
    raw_df["label"] = raw_df["label"].astype(int)
    raw_df["label_name"] = raw_df["label"].map(id2label)

    train_df, val_df, test_df = stratified_split(raw_df.dropna(subset=["label"]))

    # sample data do repo (wg Pani wymagań)
    save_sample(raw_df, Path("data/sample/sample.csv"), n=cfg.sample_size)

    # przeprocesowane splity lokalnie (ignorowane w gicie)
    save_splits(train_df, val_df, test_df, Path("data/processed"))

    print("Score column:", score_col)
    print("Label distribution:")
    print(raw_df["label_name"].value_counts(dropna=False))
    print("Split sizes:", len(train_df), len(val_df), len(test_df))
    print("Saved: data/sample/sample.csv + data/processed/*.parquet")


if __name__ == "__main__":
    main()
