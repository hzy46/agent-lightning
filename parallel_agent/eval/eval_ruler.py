import json
import os
import copy
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import time
import numpy as np
from methods import normal_get_pred_for_sample, memagent_async_get_pred_for_sample
from parallel_methods import parallel_async_get_pred_for_sample
from transformers import AutoTokenizer
import asyncio
import fire
from utils import score_func

async def sem_memagent_call_memagent(
    semaphore,
    api_root_url,
    sample,
    model,
    tokenizer,
    temperature,
    top_p,
):
    async with semaphore:
        return await memagent_async_get_pred_for_sample(
            api_root_url,
            sample,
            model,
            tokenizer,
            temperature,
            top_p,
        )


async def sem_memagent_call_parallel(
    semaphore,
    api_root_url,
    sample,
    model,
    tokenizer,
    temperature,
    top_p,
):
    async with semaphore:
        return await parallel_async_get_pred_for_sample(
            api_root_url,
            sample,
            model,
            tokenizer,
            temperature,
            top_p,
        )


def main(
    model="Qwen/Qwen2.5-7B-Instruct-1M",
    method="normal",
    tasks = [
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery",
        "vt",
        "fwe",
        "qa_1",
    ],
    context_length_strs=[
        "32K",
        "64K",
        "128K",
        "256K",
    ],
    save_root_dir="results/",
    limit_n=None,
):
    base_dir = os.path.expanduser("~/ruler_from_memagent")
    context_length_str_to_num = {
        "8K": 8192,
        "16K": 16384,
        "32K": 32768,
        "64K": 65536,
        "128K": 131072,
        "256K": 262144,
        "512K": 524288,
    }
    tokenizer = AutoTokenizer.from_pretrained(model)

    temperature = 0
    top_p = 1
    api_root_url = "http://localhost:8000/v1"

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

    for context_length_str in context_length_strs:
        for task in tasks:
            print(f"task: {task} length: {context_length_str}")
            context_length_num = context_length_str_to_num[context_length_str]
            task_path = os.path.join(base_dir, f"eval_{task}_{context_length_num}.json")
            
            with open(task_path) as f:
                samples = json.load(f)
            if limit_n is not None:
                samples = samples[:limit_n]
                print(f"limit samples to {limit_n}")
            
            model_save_dir = os.path.join(save_root_dir,  "{}_{}".format(method, model.split("/")[-1].lower()))
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
            
            max_workers = 50
            if method == "normal":
                with ThreadPoolExecutor(max_workers=50) as executor:
                    futures = [
                        executor.submit(
                            normal_get_pred_for_sample,
                            api_root_url,
                            model,
                            sample,
                            temperature,
                            top_p,
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
                            sem_memagent_call_memagent(
                                semaphore,
                                api_root_url,
                                sample,
                                model,
                                tokenizer,
                                temperature,
                                top_p,
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
            elif method == "parallel":
                async def _run_parallel():
                    # use a small parallism for parallel agent
                    semaphore = asyncio.Semaphore(5)
                    aio_tasks = [
                        asyncio.create_task(
                            sem_memagent_call_parallel(
                                semaphore,
                                api_root_url,
                                sample,
                                model,
                                tokenizer,
                                temperature,
                                top_p,
                            )
                        )
                        for sample in samples
                    ]

                    for coro in tqdm(
                        asyncio.as_completed(aio_tasks),
                        total=len(aio_tasks),
                    ):
                        await coro

                asyncio.run(_run_parallel())
            else:
                raise NotImplementedError

            for sample in samples:
                metrics = score_func(task, sample["outputs"], sample["response"])
                for k, v in metrics.items():
                    sample[k] = v
            
            print(f"task: {task} length: {context_length_str}  sub_em: {np.mean([sample['sub_em'] for sample in samples]):.2f}")
            
            with open(result_save_path, "w") as f:
                for sample in samples:
                    del sample["context"]
                    del sample["input"]
                json.dump(samples, f)


if __name__ == '__main__':
    fire.Fire(main)
    


   