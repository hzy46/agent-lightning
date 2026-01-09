import datasets
from datasets import Dataset, load_dataset, concatenate_datasets
import os
import json
from collections import defaultdict

# get hint: 64K 128K 的数据里面，没有 hint，所以这里先提前把 hint 拿到

type_to_hint_list = defaultdict(list)

full_dataset = load_dataset("InfiniAILab/gsm_infinite_hard_8K")
for row in full_dataset["ops_2"]:
    # 根据这两项决定 hint
    type_to_hint_list[(row["template"], row["mode"])].append(row["messages"][1]["content"])
type_to_hint = {}
for key, hints in type_to_hint_list.items():
    assert len(set(hints)) == 1
    type_to_hint[key] = hints[0]


# split to different parts
for is_tail in [False, True]:
    if is_tail is True:
        length_strs = ["8K", "16K", "32K"]
    else:
        length_strs = ["8K", "16K", "32K", "64K", "128K"]
    for length_str in length_strs:
        limit_n_per_op = 200 #  乘 0.05，0.4 后要是整数不然会被约去少量样本
        dataset_name = f"InfiniAILab/gsm_infinite_hard_{length_str}"
        op_list = [2, 4, 6, 8, 10]
        if is_tail:
            save_dir = os.path.expanduser("~/gsm_infinite_parsed_tail")
        else:
            save_dir = os.path.expanduser("~/gsm_infinite_parsed")
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "hard_{}.json".format(length_str))
        
        full_dataset = load_dataset(dataset_name)
        # from gsm infinite
        filter_config = [
            {"percentage":0.4,"template":"crazy_zootopia","mode":"normalforward"},
            {"percentage":0.05,"template":"movie_festival_awards","mode":"normalforward"},
            {"percentage":0.05,"template":"teachers_in_school","mode":"normalforward"},
            {"percentage":0.4,"template":"crazy_zootopia","mode":"forwardreverse"},
            {"percentage":0.05,"template":"movie_festival_awards","mode":"forwardreverse"},
            {"percentage":0.05,"template":"teachers_in_school","mode":"forwardreverse"}
        ]
        
        subsets = [f"ops_{x}" for x in op_list]
        
        filtered_datasets = []
        for split in subsets:
            dataset_split = full_dataset[split]
            total_samples = min(limit_n_per_op, len(dataset_split))
            filtered_data = []
            for config in filter_config:
                num_to_add = int(total_samples * config["percentage"])
                current_filter = {key: value for key, value in config.items() if key not in ["percentage"]}
                filtered_subset = dataset_split.filter(lambda example: all(example[key] == value for key, value in current_filter.items()))
                # naive split by head and tail
                if length_str in ["8K", "16K", "32K"]: # both in train and test
                    assert num_to_add * 2 <= len(filtered_subset), f"num_to_add: {num_to_add}  len(filtered_subset): {len(filtered_subset)}"
                if is_tail:
                    filtered_data.extend(filtered_subset.select(range(len(filtered_subset) - num_to_add, len(filtered_subset))))
                else:
                    filtered_data.extend(filtered_subset.select(range(min(num_to_add, len(filtered_subset)))))
            filtered_datasets.append(Dataset.from_list(filtered_data))
            print(f"split={split} sample_num={len(filtered_data)}")
        unprocessed_dataset = concatenate_datasets(filtered_datasets)
        
        data_list = []
        for row in unprocessed_dataset:
            context = row["problem"]
            hint = type_to_hint[(row["template"], row["mode"])]
            query = hint + "\n\nQuestion: {}".format(row["question"])
            data_list.append({
                "task_type": "gsm_infinite",
                "context": context,
                "query": query,
                "solution": row["solution"],
                "op": row["op"],
                "id": row["id"],
                "template": row["template"],
                "mode": row["mode"],
            })
        
        with open(save_path, "w") as f:
            json.dump(data_list, f)
        print("saved to {}".format(save_path))