import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_docs",
        type=int,
        default=100,
        help="number of documents (default: 100)",
    )
    return parser.parse_args()


def check_env():
    zhiyuhe = os.environ.get("ZHIYUHE")
    if zhiyuhe is None:
        raise EnvironmentError("Environment variable $ZHIYUHE is not set.")
    return Path(zhiyuhe)


def convert_dataframe(df, num_docs):
    results = []
    for _, row in df.iterrows():
        curr_q = row["extra_info"]["question"]
        curr_a = row["reward_model"]["ground_truth"]
        context = row["context"]

        results.append(
            {
                "task_type": "qa",
                "query": curr_q,
                "context": context,
                # for multiple answer
                "ground_truths": curr_a,
                "num_docs": num_docs,
            }
        )
    return results


def main():
    args = parse_args()
    num_docs = args.num_docs

    # 1. check env
    zhiyuhe = check_env()

    # 2. parquet paths
    base_dir = zhiyuhe / "memagent_hotpotqa"
    train_path = base_dir / f"hotpotqa_train_doc{num_docs}.parquet"
    dev_path = base_dir / f"hotpotqa_dev_doc{num_docs}.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"File not found: {train_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"File not found: {dev_path}")

    # 3. read parquet
    train_df = pd.read_parquet(train_path)
    dev_df = pd.read_parquet(dev_path)

    # 4. convert
    train_data = convert_dataframe(train_df, num_docs)
    test_data = convert_dataframe(dev_df, num_docs)

    # 5. save json
    out_dir = Path.home() / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = out_dir / f"train_doc{num_docs}.json"
    test_out = out_dir / f"test_doc{num_docs}.json"

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_data, f)

    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(test_data, f)

    print(f"Saved train data to {train_out}")
    print(f"Saved test data to {test_out}")


if __name__ == "__main__":
    main()
