import random
import string
import os
import json
import argparse
from typing import List, Dict

length_str_to_line_num = {
    "8K": 680,
    "16K": 1360,
}

def rand_key() -> str:
    """random 6-letter lowercase string"""
    return "".join(random.choices(string.ascii_lowercase, k=6))

def rand_6digit_int() -> int:
    """random 6-digit integer"""
    return random.randint(100000, 999999)

def generate_sample(length_str: str, min_answer_num: int, max_answer_num: int) -> Dict:
    total_lines = length_str_to_line_num[length_str]

    # 1) sample answer count x
    x = random.randint(min_answer_num, max_answer_num)

    # 2) sample ground-truth 6-digit integer
    gt = rand_6digit_int()

    # 3) build assignment chain with random variable names (each is 6 lowercase letters)
    ans_vars = [rand_key() for _ in range(x)]
    chain_lines = [f"{ans_vars[0]} = {gt}"]
    for i in range(1, x):
        chain_lines.append(f"{ans_vars[i]} = {ans_vars[i-1]}")

    # 4) build noise lines to reach fixed line count
    remain = total_lines - len(chain_lines)
    if remain < 0:
        chain_lines = chain_lines[:total_lines]
        ans_vars = ans_vars[:len(chain_lines)]
        remain = 0
        x = len(ans_vars)

    noise_lines = []
    used_lines = set(chain_lines)

    while len(noise_lines) < remain:
        if random.random() < 0.5:
            lhs = rand_key()
            rhs = rand_key()
            line = f"{lhs} = {rhs}"
        else:
            lhs = rand_key()
            num = rand_6digit_int()
            if num == gt:
                continue
            line = f"{lhs} = {num}"

        if line not in used_lines:
            used_lines.add(line)
            noise_lines.append(line)

    # 5) mix chain + noise into final context
    context_lines = chain_lines + noise_lines
    random.shuffle(context_lines)
    context = "\n".join(context_lines)

    # 6) English query
    query = (
        "The context contains assignment statements of the form `var = var` or `var = number`.\n"
        f"Find all variables that are ultimately assigned the value {gt} (by repeatedly following `var = var`).\n"
        "Output in the format: <answer>var1, var2, ..., varn</answer> (comma+space separated)."
    )

    return {
        "task_type": "vt",
        "query": query,
        "answers": ans_vars,
        "context": context,
        "length_str": length_str,
        "answer_num": x,
    }

def build_data_list(num_samples: int, length_str: str, min_answer_num: int, max_answer_num: int) -> List[Dict]:
    return [generate_sample(length_str, min_answer_num, max_answer_num) for _ in range(num_samples)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_answer_num", type=int, default=1)
    parser.add_argument("--max_answer_num", type=int, required=True)
    parser.add_argument("--train_num", type=int, default=5000)
    parser.add_argument("--test_num", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    if args.min_answer_num < 1:
        raise ValueError("--min_answer_num must be >= 1")
    if args.max_answer_num < args.min_answer_num:
        raise ValueError("--max_answer_num must be >= --min_answer_num")

    save_dir = args.save_dir or os.path.expanduser(
        f"~/vt_minans{args.min_answer_num}_maxans{args.max_answer_num}"
    )
    os.makedirs(save_dir, exist_ok=True)

    for length_str in ["8K", "16K"]:
        train_path = os.path.join(save_dir, f"train_{length_str}.json")
        test_path = os.path.join(save_dir, f"test_{length_str}.json")

        train_data = build_data_list(args.train_num, length_str, args.min_answer_num, args.max_answer_num)
        test_data = build_data_list(args.test_num, length_str, args.min_answer_num, args.max_answer_num)

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False)

        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False)

    print(f"Saved to: {save_dir}")
