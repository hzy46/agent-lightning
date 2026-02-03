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
from utils import score_func
from algorithms.normal import fill_in_response as normal_fill_in_response
from algorithms.memagent import async_fill_in_response_with_sem as memagent_async_fill_in_response_with_sem
from algorithms.parallel import async_fill_in_response_with_sem as parallel_async_fill_in_response_with_sem
# use stream retrieval
from algorithms.stream_retrieval import async_fill_in_response_with_sem as stream_async_fill_in_response_with_sem, ModelConfig as StreamModelConfig, AlgorithmConfig as StreamAlgorithmConfig

def main(
    model="Qwen/Qwen2.5-7B-Instruct-1M",
    method="normal",
    doc_num_list=[
        100,
        800,
    ],
    save_root_dir="results/",
    limit_n=None,
    max_workers=None,
    keep_origin=False,
    max_rounds=6,
    chunk_size=5000,
    fix_chunk_num=None,
):
    base_dir = os.path.expanduser("~/qa_eval_more")
    tokenizer = AutoTokenizer.from_pretrained(model)

    if "global_step_" in model:
        model_save_name = model.strip("/").split("/")[-2].lower() + "_step" + model[model.find("global_step_") + len("global_step_"):].strip("/")
        model = model.strip("/").split("/")[-1] 
    elif os.path.exists(model):
        # it is a path
        model = model.strip("/").split("/")[-1]
        model_save_name = model.lower()
    else:
        model_save_name = model.split("/")[-1].lower()
    print("model_save_name", model_save_name)

    api_root_url = "http://localhost:8000/v1"
    
    if max_workers is None:
        if method == "normal":
            max_workers = 50
        elif method == "memagent":
            max_workers = 50
        elif method == "parallel":
            max_workers = 10
        elif method == "stream":
            max_workers = 10
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

    for doc_num in doc_num_list:
        print(f"doc_num: {doc_num}")
        task_path = os.path.join(base_dir, f"test_doc{doc_num}.json")
        
        with open(task_path) as f:
            samples = json.load(f)
        if limit_n is not None:
            samples = samples[:limit_n]
            print(f"limit samples to {limit_n}")
        
        if method == "stream":
            if fix_chunk_num is None:
                model_save_dir = os.path.join(save_root_dir,  "{}_round{}_chunk{}_{}".format(method, max_rounds, chunk_size, model_save_name))
            else:
                model_save_dir = os.path.join(save_root_dir,  "{}_fix_chunk_num{}_round{}_{}".format(method, fix_chunk_num, max_rounds, model_save_name))
        elif method == 'memagent':
            model_save_dir = os.path.join(save_root_dir,  "{}_chunk{}_{}".format(method, chunk_size, model_save_name))
        else:
            model_save_dir = os.path.join(save_root_dir,  "{}_{}".format(method, model_save_name))
        result_save_path = os.path.join(model_save_dir, "result_qa_doc{}.json".format(doc_num))
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

        start_time = time.time()
        if method == "normal":
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        normal_fill_in_response,
                        api_root_url,
                        model,
                        sample,
                        "qa"
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
                            "qa",
                            chunk_size=chunk_size,
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
        elif method == "stream":
            async def _run_stream():
                semaphore = asyncio.Semaphore(max_workers)
                aio_tasks = [
                    asyncio.create_task(
                        stream_async_fill_in_response_with_sem(
                            semaphore,
                            StreamModelConfig(
                                api_root_url=api_root_url,
                                model=model,
                            ),
                            tokenizer,
                            StreamAlgorithmConfig(
                                max_rounds=max_rounds, 
                                chunk_size=chunk_size, 
                                fix_chunk_num=fix_chunk_num,
                            ),
                            sample,
                            "qa",
                        )
                    )
                    for sample in samples
                ]

                for coro in tqdm(
                    asyncio.as_completed(aio_tasks),
                    total=len(aio_tasks),
                ):
                    await coro

            asyncio.run(_run_stream())
        else:
            raise NotImplementedError
        end_time = time.time()
        avg_task_time = (end_time - start_time) / len(samples)

        for sample in samples:
            metrics = score_func(task, sample["outputs"], sample["response"])
            for k, v in metrics.items():
                sample[k] = v
        
        print(f"task: {task} length: {context_length_str} sub_em: {np.mean([sample['sub_em'] for sample in samples]):.2f} avg_task_time (this is not latency unless max_workers=1): {avg_task_time:.2f}s")
        
        with open(result_save_path, "w") as f:
            if keep_origin is False:
                for sample in samples:
                    del sample["context"]
                    del sample["input"]
            json.dump(samples, f)


if __name__ == '__main__':
    fire.Fire(main)
    


   