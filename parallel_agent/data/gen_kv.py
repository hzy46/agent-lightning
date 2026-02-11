import random
import string
import numpy as np
import os
import json
import argparse

length_str_to_key_num = {
    "8K": 600,
    "16K": 1200,
    "32K": 2400,
}

def rand_key():
    return ''.join(random.choices(string.ascii_lowercase, k=6))

def generate_dataset(length_str, min_answer_num, max_answer_num, min_hop_num, max_hop_num):
    total_path_num = length_str_to_key_num[length_str]

    # 0. 每个 sample 随机答案条数
    answer_num = random.randint(min_answer_num, max_answer_num)

    # 1. 随机 start_key
    start_key = rand_key()

    # 2. 生成每条答案路径的 hop 数
    hop_num_list = [random.randint(min_hop_num, max_hop_num) for _ in range(answer_num)]

    context_lines = []
    answers = []
    used_edges = set()

    # 3. 构造 answer 相关路径
    for hop in hop_num_list:
        keys = [start_key]
        for _ in range(hop):
            keys.append(rand_key())

        # 生成 Path 链
        for i in range(len(keys) - 1):
            edge = (keys[i], keys[i + 1])
            if edge not in used_edges:
                context_lines.append(f"Path: {keys[i]} ---> {keys[i + 1]}")
                used_edges.add(edge)

        # 最后一个是答案
        answers.append(keys[-1])

    # 4. 生成噪声 Path
    remain_num = total_path_num - len(context_lines)
    while remain_num > 0:
        a, b = rand_key(), rand_key()
        edge = (a, b)
        if edge not in used_edges:
            context_lines.append(f"Path: {a} ---> {b}")
            used_edges.add(edge)
            remain_num -= 1

    random.shuffle(context_lines)

    context = "\n".join(context_lines)
    query = (
        f"Starting from key {start_key}, trace all possible paths and list all final values "
        f"(e.g., if A→B→C and A→D→E, then the final values of key A are C and E)."
        "Output all possible final values with the following format: "
        "<answer>final_value_1, ..., final_value_n</answer>. "
        "If there is only one final value, output <answer>final_value_1</answer>"
    )

    return {
        "task_type": "kv_retrieval",
        "query": query,
        "answers": answers,
        "context": context,
        "length_str": length_str,
        "hop_num_list": hop_num_list,
        "mean_hop_num": float(np.mean(hop_num_list)),
        "answer_num": int(answer_num),
    }

def build_data_list(num_samples, length_str, min_answer_num, max_answer_num, min_hop_num, max_hop_num):
    data_list = []
    for _ in range(num_samples):
        data_list.append(
            generate_dataset(
                length_str=length_str,
                min_answer_num=min_answer_num,
                max_answer_num=max_answer_num,
                min_hop_num=min_hop_num,
                max_hop_num=max_hop_num,
            )
        )
    return data_list


# 示例
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_hop_num", type=int, default=2)
    parser.add_argument("--min_hop_num", type=int, default=1)
    parser.add_argument("--max_answer_num", type=int, default=2)
    parser.add_argument("--min_answer_num", type=int, default=1)
    args = parser.parse_args()

    max_hop_num = args.max_hop_num
    max_answer_num = args.max_answer_num
    min_hop_num = args.min_hop_num
    min_answer_num = args.min_answer_num



    if min_answer_num != 1 and min_hop_num != 1:
        save_dir = os.path.expanduser(
            f"~/multi_hop_kv_retrieval_v2_minhop{min_hop_num}_maxhop{max_hop_num}_minans{min_answer_num}_maxans{max_answer_num}"
        )
    elif min_answer_num != 1:
        save_dir = os.path.expanduser(
            f"~/multi_hop_kv_retrieval_v2_maxhop{max_hop_num}_minans{min_answer_num}_maxans{max_answer_num}"
        )
    elif min_hop_num != 1:
        save_dir = os.path.expanduser(
            f"~/multi_hop_kv_retrieval_v2_minhop{min_hop_num}_maxhop{max_hop_num}_maxans{max_answer_num}"
        )
    else:
        save_dir = os.path.expanduser(
            f"~/multi_hop_kv_retrieval_v2_maxhop{max_hop_num}_maxans{max_answer_num}"
        )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(save_dir)

    train_num = 5000
    test_num = 300

    for length_str in ["8K", "16K"]:
        train_save_path = os.path.join(
            save_dir,
            f"train_{length_str}.json"
        )
        test_save_path = os.path.join(
            save_dir,
            f"test_{length_str}.json"
        )

        train_data_list = build_data_list(
            num_samples=train_num,
            length_str=length_str,
            min_answer_num=min_answer_num,
            max_answer_num=max_answer_num,
            min_hop_num=min_hop_num,
            max_hop_num=max_hop_num,
        )
        test_data_list = build_data_list(
            num_samples=test_num,
            length_str=length_str,
            min_answer_num=min_answer_num,
            max_answer_num=max_answer_num,
            min_hop_num=min_hop_num,
            max_hop_num=max_hop_num,
        )

        with open(train_save_path, "w") as f:
            json.dump(train_data_list, f)
        with open(test_save_path, "w") as f:
            json.dump(test_data_list, f)