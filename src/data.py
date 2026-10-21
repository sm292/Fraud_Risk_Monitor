"""Load a local ULB CSV or fetch OpenML's mirror of the benchmark."""
from pathlib import Path
import argparse
import pandas as pd
import openml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "creditcard.csv"
TARGET = "Class"


def validate_frame(df: pd.DataFrame) -> None:
    missing = {"Time", "Amount", TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty or df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected non-empty data with binary Class labels 0 and 1")
    features = df.drop(columns=TARGET)
    if features.isna().any().any() or not all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes):
        raise ValueError("Expected complete numeric feature columns")


def download_dataset(force: bool = False, local_path: Path | None = None) -> Path:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if local_path is not None:
        source = Path(local_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Local dataset CSV not found: {source}")
        frame = pd.read_csv(source)
        validate_frame(frame)
        if source != RAW.resolve():
            frame.to_csv(RAW, index=False)
        return RAW
    if RAW.exists() and not force:
        validate_frame(pd.read_csv(RAW))
        return RAW
    try:
        dataset = openml.datasets.get_dataset(1597, download_data=True)
        X, y, _, _ = dataset.get_data(target=TARGET, dataset_format="dataframe")
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve OpenML dataset 1597 (often a network timeout). "
            "A missing OpenML cache file is normal on first run. Download the ULB "
            "CSV from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud, "
            f"place it at {RAW}, then rerun `python -m src.data`; or use "
            "`python -m src.data --local-path <path-to-creditcard.csv>`."
        ) from exc
    if y is None:
        raise ValueError("OpenML dataset 1597 has no Class target")
    frame = X.copy()
    frame[TARGET] = pd.Series(y, index=X.index).astype(int)
    validate_frame(frame)
    frame.to_csv(RAW, index=False)
    return RAW


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-path", type=Path, help="Use an already-downloaded source CSV")
    parser.add_argument("--force", action="store_true", help="Refetch OpenML instead of using the cached local CSV")
    args = parser.parse_args()
    path = download_dataset(force=args.force, local_path=args.local_path)
    frame = pd.read_csv(path)
    print(f"Validated {path}: {len(frame):,} rows, {int(frame[TARGET].sum()):,} frauds")
