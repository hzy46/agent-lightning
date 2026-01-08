import json
import os
import copy
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import time
import numpy as np
from transformers import AutoTokenizer
import asyncio
import fire
from algorithms.normal import fill_in_response as normal_fill_in_response
from algorithms.memagent import async_fill_in_response_with_sem as memagent_async_fill_in_response_with_sem

import re

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False 

# evaluation function from gsm infinite
def score_func(response, solution):
    # solution 是 sample 中的，给答案提取出来
    idx_answer_start = solution.find("Answer: ") 
    idx_answer_end = solution.find(".", idx_answer_start) 
    answer_text = solution[idx_answer_start + len("Answer: ") : idx_answer_end] 
    answer_text = int(answer_text.lower()) 
        
    response = re.sub('.\x08', 'b', response)
    response = response.lower() 

    idx_generated_begin = -1
    idx_generated_conclude = -1
    keywords = ["answer: ", "solution: ", "oxed{", "**answer:** ", "**answer: ", "final answer: answer: ", "\nanswer: ", r"\text{answer: } ",  "is ", "answer: "] # updated 
    keywordsend = [".", ".", "}", ".", "**", ".", ".", None, ".", "\n"] 
    cnt = 0 

    while not (idx_generated_begin != -1 and idx_generated_conclude != -1) and cnt < len(keywords): 
        if keywords[cnt] in ["oxed{", "is "]: 
            idx_generated_begin = response.rfind(keywords[cnt]) # this relies on the generated is stopped before generated next question plus onwoards by stop 
        else: 
            idx_generated_begin = response.find(keywords[cnt]) 
        if idx_generated_begin != -1: 
            if keywordsend[cnt] is None: 
                idx_generated_conclude = idx_generated_begin + len(keywords[cnt]) 
                while response[idx_generated_conclude].isdigit() == True: 
                    idx_generated_conclude += 1 
            else: 
                idx_generated_conclude = response.find(keywordsend[cnt], idx_generated_begin + len(keywords[cnt])) 
            if idx_generated_conclude == -1: 
                idx_generated_conclude = len(response) 
        if not is_integer(response[idx_generated_begin + len(keywords[cnt]) : idx_generated_conclude]): 
            idx_generated_begin = -1 
            idx_generated_conclude = -1 
        cnt += 1 
    
    if idx_generated_begin == -1: 
        return 0
    else: 
        try: 
            answergenerated_text = int(response[idx_generated_begin + len(keywords[cnt - 1]) : idx_generated_conclude]) 
        except: 
            return 0
        return int(answergenerated_text == answer_text)


def main(
    model="Qwen/Qwen2.5-7B-Instruct-1M",
    method="normal",
    task_ops=[
        2,
        4,
        6,
        8,
    ],
    length_strs=[
        # "8K",
        # "16K",
        "32K",
        "64K",
        "128K",
    ],
    save_root_dir="results/",
    limit_n=None,
    max_workers=None,
):
    base_dir = os.path.expanduser("~/gsm_infinite_parsed")
    tokenizer = AutoTokenizer.from_pretrained(model)

    if os.path.exists(model):
        # it is a path
        model = model.split("/")[-1]

    api_root_url = "http://localhost:8000/v1"
    
    if max_workers is None:
        if method == "normal":
            max_workers = 50
        elif method == "memagent":
            max_workers = 50
        elif method == "parallel":
            max_workers = 5
        else:
            raise NotImplementedError


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

    for length_str in length_strs:
        for task_op in task_ops:
            print(f"task_op: {task_op} length: {length_str}")
            task_path = os.path.join(base_dir, f"hard_{length_str}.json")
            
            with open(task_path) as f:
                samples = json.load(f)

            # filter by task_op
            samples = [sample for sample in samples if sample["op"] == task_op]

            if limit_n is not None:
                samples = samples[:limit_n]
                print(f"limit samples to {limit_n}")

            model_save_dir = os.path.join(save_root_dir,  "{}_{}".format(method, model.split("/")[-1].lower()))
            result_save_path = os.path.join(model_save_dir, "gsm_result_{}_{}.json".format(task_op, length_str))
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
            if method == "normal":
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            normal_fill_in_response,
                            api_root_url,
                            model,
                            sample,
                            "gsm_infinite"
                        )
                        for sample in samples
                    ]
                
                    for future in tqdm(as_completed(futures), total=len(futures)):
                        future.result()
            elif method == "memagent":
                async def _run_memagent():
                    semaphore = asyncio.Semaphore(max_workers)
                    aio_tasks = [
                        asyncio.create_task(
                            memagent_async_fill_in_response_with_sem(
                                semaphore,
                                api_root_url,
                                sample,
                                model,
                                tokenizer,
                                "gsm_infinite"
                            )
                        )
                        for sample in samples
                    ]

                    for coro in tqdm(
                        asyncio.as_completed(aio_tasks),
                        total=len(aio_tasks),
                    ):
                        await coro

                asyncio.run(_run_memagent())
            else:
                raise NotImplementedError

            for sample in samples:
                sample["score"] = score_func(sample["response"], sample["solution"])

            
            print(f"task_op: {task_op} length_str: {length_str} score: {np.mean([sample['score'] for sample in samples]):.2f}")
            
            with open(result_save_path, "w") as f:
                for sample in samples:
                    del sample["context"]
                    del sample["query"]
                json.dump(samples, f)

if __name__ == "__main__":
    fire.Fire(main)