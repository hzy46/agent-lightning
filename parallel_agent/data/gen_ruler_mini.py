import os
import json
import copy

root_dir = os.path.expanduser("~/ruler_from_memagent")
save_path = os.path.expanduser("~/ruler_mini.json")

tasks = [
    # "niah_single_1",
    # "niah_single_2",
    # "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "fwe",
    "qa_1",
]

lengths = [
    # "64K",
    "128K",
]

length_to_length_num = {
    # "64K": 65536,
    "128K": 131072,
}

n_sample_per_pair = 30

output_list = []
for task in tasks:
    for length in lengths:
        path = os.path.join(root_dir, "eval_{}_{}.json".format(task, length_to_length_num[length]))
        with open(path) as f:
            data_list = json.load(f)
        assert len(data_list) >= n_sample_per_pair
        for data in data_list[:n_sample_per_pair]:
            data = copy.deepcopy(data)
            data["length"] = length
            data["task_type"] = "ruler"
            data["sub_task_type"]= task
            output_list.append(data)

with open(save_path, "w") as f:
    json.dump(output_list, f)