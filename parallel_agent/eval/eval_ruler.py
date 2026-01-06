import json
import os
import copy
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import time
import numpy as np
import re
from collections import Counter
import string


prompt_template = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

$Q$

Format your response as follows: "Therefore, the answer is (insert answer here)"."""


def extract_answer(response):
    response = response.replace('*', '')

    if "the answer is" in response:
        ans = response.rsplit("the answer is", 1)[-1].strip().replace("<｜Assistant｜>", '').replace("<｜end▁of▁sentence｜>", '').strip().strip('.').strip()
    else:
        ans = None
    return ans

def extract_solution(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
  
    # Extract final answer using XML-style tags
    if "</think>" not in solution_str:
        if not os.environ.get("FORCE_THINK"):
            return solution_str, solution_str
        else:
            print("[Error] No valid answer tags found")
            return None, solution_str 
    final_answer = solution_str.split("</think>")[-1].strip()
    return final_answer, solution_str

### From RULER
def string_match_all(pred, ref):
    return sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)

def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        metrics['sub_em'] += string_match_all(pred, gold)
    metrics['total_num'] = len(goldens)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def sub_exact_match_score(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    return (ground_truth in prediction) or (prediction in ground_truth)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    subem = sub_exact_match_score(prediction, gold)

    f1, prec, recall = f1_score(prediction, gold)
    metrics['sub_em'] += subem
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    metrics['total_num'] += 1
    return em, prec, recall


def calc_qa_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics


def score_func(task, answers, response):
    pred, _ = extract_solution(response)
    final_pred = extract_answer(pred) if pred else extract_answer(response)
    if "qa" in task:
        # TBD: mem agent uses only the first gold answer here. Not sure about the reason. Need to check
        return calc_qa_metrics([final_pred], [answers[0]])
    else:
        sub_em = calc_metrics([final_pred], [answers])['sub_em'] if final_pred else 0
        return {"sub_em": sub_em}

def fill_pred_for_sample(api_root_url, model, sample, temperature, top_p, max_new_tokens):
    prompt = sample["prompt"]
    r = requests.post(
        url=api_root_url + "/chat/completions",
        headers={"Authorization": f"Bearer dummy"},
        json=dict(model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens
        )
    )
    r.raise_for_status()
    data = r.json()
    sample["response"] = data['choices'][0]['message']['content']


base_dir = os.path.expanduser("~/ruler_from_memagent")
tasks = [
    # "niah_single_1",
    # "niah_single_2",
    # "niah_single_3",
    # "niah_multikey_1",
    # "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "fwe",
    "qa_1",
]
context_length_strs = [
    # "8K",
    "64K",
    # "128K",
    # "256K",
]
context_length_str_to_num = {
    "8K": 8192,
    "16K": 16384,
    "32K": 32768,
    "64K": 65536,
    "128K": 131072,
    "256K": 262144,
    "512K": 524288,
}



max_new_tokens = 8192
temperature = 0
top_p = 1
api_root_url = "http://localhost:8000/v1"
# model = os.path.expanduser("~/Qwen2.5-7B-Instruct-Yarn")
# model = "Qwen2.5-7B-Instruct-Yarn"
model = "Qwen/Qwen2.5-7B-Instruct-1M"
save_root_dir = "results/"

while True:
    print("try to conntect...")
    p = subprocess.run(["curl", "-m", "100000000", api_root_url + "/models"], capture_output=True)
    if p.returncode != 0:
        print("waiting...")
        time.sleep(5)
    elif rf'"id":"{model}"' not in p.stdout.decode():
        print("model not found, maybe shutting down previous server...")
        time.sleep(5)
    else:
        print("connected")
        # time.sleep(5)
        break

for task in tasks:
    for context_length_str in context_length_strs:
        print(f"task: {task} length: {context_length_str}")
        context_length_num = context_length_str_to_num[context_length_str]
        task_path = os.path.join(base_dir, f"eval_{task}_{context_length_num}.json")
        
        with open(task_path) as f:
            samples = json.load(f)
        
        model_save_dir = os.path.join(save_root_dir, model.split("/")[-1].lower())
        result_save_path = os.path.join(model_save_dir, "result_{}_{}.json".format(task, context_length_str))
        if os.path.exists(model_save_dir) is False:
            os.makedirs(model_save_dir)
        if os.path.exists(result_save_path):
            with open(result_save_path) as f:
                existing_results = json.load(f)
            if len(existing_results) == len(samples):
                # skip
                print("skip!")
                continue
            else:
                # clear
                with open(result_save_path, "w") as f:
                    pass
        
        for sample in samples:
            context = sample["context"]
            input = sample["input"]
            prompt = prompt_template.replace('$DOC$', context.strip()).replace('$Q$', input.strip())
            sample["prompt"] = prompt
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(
                    fill_pred_for_sample,
                    api_root_url,
                    model,
                    sample,
                    temperature,
                    top_p,
                    max_new_tokens
                )
                for sample in samples
            ]
        
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()
        
        for sample in samples:
            metrics = score_func(task, sample["outputs"], sample["response"])
            for k, v in metrics.items():
                sample[k] = v
        
        print(f"task: {task} length: {context_length_str}  sub_em: {np.mean([sample['sub_em'] for sample in samples]):.2f}")
        
        with open(result_save_path, "w") as f:
            for sample in samples:
                del sample["context"]
                del sample["prompt"]
                del sample["input"]
            json.dump(samples, f)