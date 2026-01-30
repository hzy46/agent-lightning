import random
import re

import agentlightning as agl
import os
import json
import copy
from transformers import AutoTokenizer
from eval.algorithms.parallel import async_fill_in_response as parallel_async_fill_in_response
from eval.algorithms.normal import async_fill_in_response as normal_async_fill_in_response
from eval.algorithms.memagent import async_fill_in_response as memagent_async_fill_in_response
from eval.utils import score_func_gsm_infinite, score_func as score_func_ruler
from eval.utils import score_func_kv_retrieval
from eval.algorithms.stream import async_fill_in_response as stream_async_fill_in_response, ModelConfig as StreamModelConfig, AlgorithmConfig as StreamAlgorithmConfig
from eval.algorithms.stream_retrieval import async_fill_in_response as stream_retrieval_async_fill_in_response

import traceback
import fire
from functools import partial
import math


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
            # "path": os.path.expanduser("~/Qwen2.5-7B-Instruct-Yarn"),
            # "path": os.path.expanduser("Qwen/Qwen2.5-7B-Instruct"),
            "path": "placeholder",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 4,
        "val_before_train": True,
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

tokenizer_info = {
    "tokenizer": None
}


parallel_config = {
    "fix_chunk_num": None,
    "use_token_penalty": False,
    "token_penalty_L": 1024,
    "token_penalty_k": 0.0001,
    "max_rounds": 3,
}

@agl.rollout
async def solver_agent_parallel(task, llm) -> None:
    # Query LLM endpoint. All queries will be automatically tracked by LLM proxy
    fix_chunk_num = parallel_config['fix_chunk_num']
    tokenizer =  tokenizer_info["tokenizer"]

    try:
        model = llm.model
        api_root_url = llm.endpoint
        temperature = llm.sampling_parameters.get("temperature", 1.0)
        task = copy.deepcopy(task['data']) # workaround 因为 agl 似乎会强行 merge 不一样的 task 转成一样的 key
        # print(task["task_type"], task.keys())
        await parallel_async_fill_in_response(
            api_root_url,
            task,
            model,
            tokenizer,
            task["task_type"],
            temperature,
            fix_chunk_num=fix_chunk_num,
            max_rounds=parallel_config["max_rounds"],
        )
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

    # during training
    use_token_penalty = parallel_config['use_token_penalty']
    token_penalty_L = parallel_config['token_penalty_L']
    token_penalty_k = parallel_config['token_penalty_k']
    if temperature != 0 and use_token_penalty:
        output_token_num = task["output_token_num"]
        if output_token_num <= token_penalty_L:
            cost = 0
        else:
            cost = 1 - math.exp(-token_penalty_k * (output_token_num - token_penalty_L))
        # only apply on positive reward
        if reward > 0:
            print(f"reward: {reward}  output_token_num: {output_token_num} cost: {cost} reward - cost: {reward - cost}")
            reward = reward - cost

    # This reward will be tracked automatically
    agl.emit_reward(reward)




stream_config = {
    "use_token_penalty": False,
    "token_penalty_L": 1024,
    "token_penalty_k": 0.0001,
    "algorithm": StreamAlgorithmConfig(
        max_rounds=3,
        chunk_size=5000, 
        fix_chunk_num=None,
        only_answer_in_gsm=False,
    )
}


@agl.rollout
async def solver_agent_stream(task, llm) -> None:
    # Query LLM endpoint. All queries will be automatically tracked by LLM proxy
    tokenizer =  tokenizer_info["tokenizer"]
    temperature = llm.sampling_parameters.get("temperature", 1.0)
    try:
        stream_model_config = StreamModelConfig(
            api_root_url=llm.endpoint,
            model=llm.model,
            temperature=temperature,
        )
        task = copy.deepcopy(task['data']) # workaround 因为 agl 似乎会强行 merge 不一样的 task 转成一样的 key
        # model_config, tokenizer, algorithm_config, sample, task_type)
        # print(f"stream_algorithm_config: fix_chunk_num={stream_config['algorithm'].fix_chunk_num}")
        if task["task_type"] == "ruler" or task["task_type"] == "kv_retrieval":
            await stream_retrieval_async_fill_in_response(
                stream_model_config,
                tokenizer,
                stream_config['algorithm'],
                task,
                task["task_type"],
            )
        elif task["task_type"] == "gsm_infinite":
            await stream_async_fill_in_response(
                stream_model_config,
                tokenizer,
                stream_config['algorithm'],
                task,
                task["task_type"],
            )
    except Exception as e:
        print("Failure:", traceback.format_exc())
        task["response"] = ""

    if task["task_type"] == "gsm_infinite":
        reward = int(score_func_gsm_infinite(task["response"], task["solution"]))
    elif task["task_type"] == "ruler":
        reward = score_func_ruler(task["sub_task_type"], task['outputs'], task['response'])['sub_em']
    elif task["task_type"] == "memagent_train":
        reward = score_func_ruler("qa", task['answers'], task['response'])['sub_em']
    elif task["task_type"]  == "kv_retrieval":
        reward = score_func_kv_retrieval(task["response"], task["answers"])
    else:
        raise NotImplementedError

    # during training
    use_token_penalty = stream_config['use_token_penalty']
    token_penalty_L = stream_config['token_penalty_L']
    token_penalty_k = stream_config['token_penalty_k']
    if temperature != 0 and use_token_penalty:
        output_token_num = task["output_token_num"]
        if output_token_num <= token_penalty_L:
            cost = 0
        else:
            cost = 1 - math.exp(-token_penalty_k * (output_token_num - token_penalty_L))
        # only apply on positive reward
        if reward > 0:
            print(f"reward: {reward}  output_token_num: {output_token_num} cost: {cost} reward - cost: {reward - cost}")
            reward = reward - cost

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
        await normal_async_fill_in_response(api_root_url, model, task, task["task_type"], temperature)
    except Exception as e:
        print("Failure:", traceback.format_exc())
        task["response"] = ""

    if task["task_type"] == "gsm_infinite":
        reward = int(score_func_gsm_infinite(task["response"], task["solution"]))
    elif task["task_type"] == "ruler":
        reward = score_func_ruler(task["sub_task_type"], task['outputs'], task['response'])['sub_em']
    elif task["task_type"] == "memagent_train":
        reward = score_func_ruler("qa", task['answers'], task['response'])['sub_em']
    elif task["task_type"] == "kv_retrieval":
        reward = score_func_kv_retrieval(task["response"], task["answers"])
    else:
        raise NotImplementedError

    # This reward will be tracked automatically
    agl.emit_reward(reward)

@agl.rollout
async def solver_agent_memagent(task, llm) -> None:
    # Query LLM endpoint. All queries will be automatically tracked by LLM proxy
    tokenizer =  tokenizer_info["tokenizer"]
    try:
        model = llm.model
        api_root_url = llm.endpoint
        temperature = llm.sampling_parameters.get("temperature", 1.0)
        task = copy.deepcopy(task['data']) # workaround 因为 agl 似乎会强行 merge 不一样的 task 转成一样的 key
        # print(task["task_type"], task.keys())
        await memagent_async_fill_in_response(api_root_url, task, model, tokenizer, task["task_type"], temperature)
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
    "memagent": solver_agent_memagent,
    "stream": solver_agent_stream,
}

def main(
    train_doc_nums=[],
    train_gsm_lengths=["8K", "16K"],
    train_kv_lengths=[], # "8K", "16K"
    train_kv_subset="maxhop2_maxans2",
    from_model=os.path.expanduser("~/train_qwen2.5-7b_normal_gsm_8K/global_step_500"),
    method="parallel",
    eval_ruler=False,
    max_rounds=3,
    use_token_penalty=False,
    token_penalty_L=1024,
    token_penalty_k=0.0001,
    fix_chunk_num=None, # for parallel only
    agg_mode=False,
    use_new_gen_data=False,
    only_answer_in_gsm=False,
):

    # set name according to paras
    if "train_qwen2.5-7b_normal_gsm_8K/global_step_500" in from_model:
        experiment_name = f"train_from_gsm_8k_step500_qwen2.5-7b_{method}_"
    elif "qwen/qwen2.5-7b-instruct" == from_model.lower():
        experiment_name = f"train_qwen2.5-7b_{method}_"
    elif "train_qwen2.5-7b_normal_kv_8K/global_step_400" in from_model:
        experiment_name = f"train_from_kv_8k_step400_qwen2.5-7b_{method}_"
    else:
        raise NotImplementedError
    verl_config["actor_rollout_ref"]["model"]["path"] = from_model

    tokenizer_info["tokenizer"] = AutoTokenizer.from_pretrained(from_model)

    if len(train_doc_nums) > 0:
        experiment_name += "docs_" + "-".join([str(doc_num) for doc_num in train_doc_nums]) + "_"
    if len(train_gsm_lengths) > 0:
        experiment_name += "gsm_" + "-".join([str(length) for length in train_gsm_lengths]) + "_"
    if len(train_kv_lengths) > 0:
        experiment_name += f"kv_v2_{train_kv_subset}_" + "-".join([str(length) for length in train_kv_lengths]) + "_"
    if use_token_penalty:
        experiment_name = experiment_name + f"token_penalty_L{token_penalty_L}_k{token_penalty_k}_"
    if max_rounds != 3: # default = 3
         experiment_name = experiment_name + f"max_rounds_{max_rounds}_"
    if fix_chunk_num is not None:
        experiment_name = experiment_name + f"fix_chunk_num_{fix_chunk_num}_"
    if agg_mode:
        assert method == "stream"
        experiment_name = experiment_name + f"agg_"
    if use_new_gen_data:
        experiment_name = experiment_name + f"data_new_gen_"
    if only_answer_in_gsm and method == "stream":
        experiment_name = experiment_name + f"only_ans_"


    experiment_name = experiment_name.strip("_")
    verl_config["trainer"]["experiment_name"] = experiment_name

    # adjust parameter
    if method == "normal":
        # max context length is 13233
        verl_config["data"]["max_prompt_length"] = 15000
        verl_config["data"]["max_response_length"] = 2048
        verl_config["actor_rollout_ref"]["actor"]['ppo_mini_batch_size'] = 32
        verl_config["actor_rollout_ref"]["actor"]['ppo_micro_batch_size_per_gpu'] = 2
        assert use_token_penalty is False
    elif method == "parallel":
        verl_config["data"]["max_prompt_length"] = 10240
        verl_config["data"]["max_response_length"] = 1024
        parallel_config['use_token_penalty'] = use_token_penalty
        parallel_config['token_penalty_L'] = token_penalty_L
        parallel_config['token_penalty_k'] = token_penalty_k
        parallel_config['fix_chunk_num'] = fix_chunk_num
        parallel_config['max_rounds'] = max_rounds
    elif method == "stream":
        if agg_mode:
            verl_config["data"]["max_prompt_length"] = 10240
            verl_config["data"]["max_response_length"] = 1024
            verl_config["agentlightning"] =  {
                "trace_aggregator": {
                    "level": "trajectory",
                    "trajectory_max_prompt_length": 10240,
                    "trajectory_max_response_length": 5120,
                }
            }
        else:
            verl_config["data"]["max_prompt_length"] = 10240
            verl_config["data"]["max_response_length"] = 1024
        if max_rounds >= 6 and max_rounds <= 8:
            verl_config["actor_rollout_ref"]['actor']['ppo_micro_batch_size_per_gpu'] = 2
        elif max_rounds > 8:
            verl_config["actor_rollout_ref"]['actor']['ppo_micro_batch_size_per_gpu'] = 1

        stream_config['use_token_penalty'] = use_token_penalty
        stream_config['token_penalty_L'] = token_penalty_L
        stream_config['token_penalty_k'] = token_penalty_k
        stream_config['algorithm'].fix_chunk_num = fix_chunk_num
        stream_config['algorithm'].max_rounds = max_rounds
        stream_config['algorithm'].only_answer_in_gsm = only_answer_in_gsm

    elif method == "memagent":
        verl_config["data"]["max_prompt_length"] = 10240
        verl_config["data"]["max_response_length"] = 1024
        assert use_token_penalty is False

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

    if use_new_gen_data:
        gsm_train_dataset_dir = os.path.expanduser("~/new_gen_gsm_infinite_parsed")
    else:
        gsm_train_dataset_dir = os.path.expanduser("~/gsm_infinite_parsed_train")
    for gsm_length in train_gsm_lengths:
        file_path = os.path.join(gsm_train_dataset_dir, f"hard_{gsm_length}.json")
        with open(file_path) as f:
            data_list = json.load(f)
        for data in data_list:
            train_sample_list.append({
                "data": data
            })

    kv_train_dataset_dir = os.path.expanduser(f"~/multi_hop_kv_retrieval_v2_{train_kv_subset}")
    for kv_length in train_kv_lengths:
        file_path = os.path.join(kv_train_dataset_dir, f"train_{kv_length}.json")
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

    if len(train_gsm_lengths) > 0:
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
    elif len(train_kv_lengths) > 0:
        kv_test_dataset_dir = os.path.expanduser(f"~/multi_hop_kv_retrieval_v2_{train_kv_subset}")
        if method == "normal":
            kv_lengths = ["8K"]
        else:
            kv_lengths = ["8K", "16K"]
        for kv_length in kv_lengths:
            file_path = os.path.join(kv_test_dataset_dir, f"test_{kv_length}.json")
            with open(file_path) as f:
                data_list = json.load(f)
            for data in data_list:
                test_sample_list.append({
                    "data": data
                })
    else:
        raise NotImplementedError
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