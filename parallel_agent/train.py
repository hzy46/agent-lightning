import random
import re

import agentlightning as agl
import os
import json
import copy
from transformers import AutoTokenizer
from eval.algorithms.parallel import async_fill_in_response as parallel_async_fill_in_response
from eval.algorithms.normal import async_fill_in_response as normal_async_fill_in_response
from eval.utils import score_func_gsm_infinite, score_func as score_func_ruler

import traceback
import fire

verl_config = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_batch_size": 16,
        "max_prompt_length": 10240,
        "max_response_length": 1024,
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 8,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.8,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 128,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
            "entropy_from_logits_with_chunking": True # for oom issue
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-7B-Instruct-1M",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 4,
        "val_before_train": False,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "ParallelAgent",
        "experiment_name": "placeholder",
        "nnodes": 1,
        "save_freq": 100,
        "test_freq": 50,
        "total_epochs": 10,
    },
}

tokenizer = AutoTokenizer.from_pretrained(verl_config["actor_rollout_ref"]["model"]["path"])


@agl.rollout
async def solver_agent_parallel(task, llm) -> None:
    # Query LLM endpoint. All queries will be automatically tracked by LLM proxy
    try:
        model = llm.model
        api_root_url = llm.endpoint
        temperature = llm.sampling_parameters.get("temperature", 1.0)
        task = copy.deepcopy(task['data']) # workaround 因为 agl 似乎会强行 merge 不一样的 task 转成一样的 key
        # print(task["task_type"], task.keys())
        await parallel_async_fill_in_response(api_root_url, task, model, tokenizer, task["task_type"], temperature)
    except Exception as e:
        print("Failure:", traceback.format_exc())
        task["response"] = ""

    if task["task_type"] == "gsm_infinite":
        reward = int(score_func_gsm_infinite(task["response"], task["solution"]))
    elif task["task_type"] == "ruler":
        reward = score_func_ruler(task["sub_task_type"], task['outputs'], task['response'])['sub_em']
    elif task["task_type"] == "memagent_train":
        reward = score_func_ruler("qa", task['answers'], task['response'])['sub_em']
    else:
        raise NotImplementedError

    # This reward will be tracked automatically
    agl.emit_reward(reward)


@agl.rollout
async def solver_agent_normal(task, llm) -> None:
    # Query LLM endpoint. All queries will be automatically tracked by LLM proxy
    try:
        model = llm.model
        api_root_url = llm.endpoint
        temperature = llm.sampling_parameters.get("temperature", 1.0)
        task = copy.deepcopy(task['data']) # workaround 因为 agl 似乎会强行 merge 不一样的 task 转成一样的 key
        # print(task["task_type"], task.keys())
        await normal_async_fill_in_response(api_root_url, modelm task, task["task_type"], temperature)
    except Exception as e:
        print("Failure:", traceback.format_exc())
        task["response"] = ""

    if task["task_type"] == "gsm_infinite":
        reward = int(score_func_gsm_infinite(task["response"], task["solution"]))
    elif task["task_type"] == "ruler":
        reward = score_func_ruler(task["sub_task_type"], task['outputs'], task['response'])['sub_em']
    elif task["task_type"] == "memagent_train":
        reward = score_func_ruler("qa", task['answers'], task['response'])['sub_em']
    else:
        raise NotImplementedError

    # This reward will be tracked automatically
    agl.emit_reward(reward)


method_to_agent_func = {
    "normal": solver_agent_normal,
    "parallel": solver_agent_parallel,
}

def main(
    train_doc_nums=[],
    train_gsm_lengths=["16K"],
    method="parallel",
    eval_ruler=False,
):

    # set name according to paras
    experiment_name = f"train_qwen2.5-7b-1m_{method}_"
    if len(train_doc_nums) > 0:
        experiment_name += "docs_" + "-".join([str(doc_num) for doc_num in train_doc_nums]) + "_"
    if len(train_gsm_lengths) > 0:
        experiment_name += "gsm_" + "-".join([str(length) for length in train_gsm_lengths]) + "_"
    experiment_name = experiment_name.strip("_")
    verl_config["trainer"]["experiment_name"] = experiment_name


    # adjust parameter
    if method == "normal":
        # max context length is 13233
        verl_config["data"]["max_prompt_length"] = 14000
        verl_config["data"]["max_response_length"] = 2048
    elif method == "parallel":
        verl_config["data"]["max_prompt_length"] = 10240
        verl_config["data"]["max_response_length"] = 1024
    elif method == "memagent":
        verl_config["data"]["max_prompt_length"] = 10240
        verl_config["data"]["max_response_length"] = 1024


    rng = random.Random(42)
    train_sample_list = []
    # mem agent train data (may need regenerate)
    memagent_train_dataset_dir = os.path.expanduser("~/ruler_from_memagent")
    
    for doc_num in train_doc_nums:
        file_path = os.path.join(memagent_train_dataset_dir, f"eval_{doc_num}.json")
        with open(file_path) as f:
            data_list = json.load(f)
        for data in data_list:
            data["task_type"] = "memagent_train"
            data["doc_num"] = doc_num
        for data in data_list:
            train_sample_list.append({
                "data": data
            })


    gsm_train_dataset_dir = os.path.expanduser("~/gsm_infinite_parsed_train")
    for gsm_length in train_gsm_lengths:
        file_path = os.path.join(gsm_train_dataset_dir, f"hard_{gsm_length}.json")
        with open(file_path) as f:
            data_list = json.load(f)
        for data in data_list:
            train_sample_list.append({
                "data": data
            })

    rng.shuffle(train_sample_list)

    # prepare_test
    test_sample_list = []
    # gsm_infinite
    gsm_test_dataset_dir = os.path.expanduser("~/gsm_infinite_parsed_eval")
    for file_name in [
        # "hard_8K.json", 
        "hard_16K.json", 
        # "hard_32K.json"
    ]:
        file_path = os.path.join(gsm_test_dataset_dir, file_name)
        with open(file_path) as f:
            data_list = json.load(f)
        for data in data_list:
            test_sample_list.append({
                "data": data  # workaround 因为 agl 似乎会强行 merge 不一样的 task 转成一样的 key
            })
    # ruler
    if eval_ruler:
        ruler_test_file_path = os.path.expanduser("~/ruler_mini.json")
        with open(ruler_test_file_path) as f:
            data_list = json.load(f)
        for data in data_list:
            test_sample_list.append({
                "data": data
            })

    rng.shuffle(test_sample_list)

    algorithm = agl.VERL(verl_config)
    # Number of agents launched in parallel to query the LLM.
    # This parameter strongly affects throughput and efficiency:
    # higher parallelism improves utilization but increases GPU overhead.
    n_runners = 16
    # This tracer is a dummy one, as currently tracing is done in the llm proxy part
    tracer = agl.OtelTracer()
    adapter = agl.LlmProxyTraceToTriplet()
    # Set store=None to use managed store
    trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=None, tracer=tracer, adapter=adapter)

    agent_func = method_to_agent_func[method]
    trainer.fit(agent_func, train_sample_list, val_dataset=test_sample_list)


if __name__ == "__main__":
    fire.Fire(main)