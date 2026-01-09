import random
import re

import agentlightning as agl
import os
import json
import copy
from transformers import AutoTokenizer
from eval.algorithms.parallel import async_fill_in_response as parallel_async_fill_in_response
from eval.utils import score_func_gsm_infinite

verl_config = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_batch_size": 16,
        "max_prompt_length": 6192,
        "max_response_length": 2000,
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
            "path": "Qwen/Qwen2.5-7B-Instruct",
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
        "experiment_name": "train_new_qwen2.5-7b-instruct",
        "nnodes": 1,
        "save_freq": 500,
        "test_freq": 25,
        "total_epochs": 2,
    },
}

tokenizer = AutoTokenizer.from_pretrained(verl_config["actor_rollout_ref"]["model"]["path"])


@agl.rollout
async def solver_agent(task, llm) -> None:
    # Query LLM endpoint. All queries will be automatically tracked by LLM proxy
    try:
        model = llm.model
        api_root_url = llm.endpoint
        temperature = llm.sampling_parameters.get("temperature", 1.0)
        task = copy.deepcopy(task)
        await parallel_async_fill_in_response(api_root_url, task, model, tokenizer, task["task_type"], temperature)
    except Exception as e:
        print("Failure:", str(e))
        task["response"] = ""

    assert task["task_type"] == "gsm_infinite"

    reward = int(score_func_gsm_infinite(task["response"], task["solution"]))
    # This reward will be tracked automatically
    agl.emit_reward(reward)


if __name__ == "__main__":
    train_dataset_dir = os.path.expanduser("~/gsm_infinite_parsed_tail")
    test_dataset_dir = os.path.expanduser("~/gsm_infinite_parsed")
    rng = random.Random(42)

    train_sample_list = []
    for file_name in ["hard_8K.json", "hard_16K.json", "hard_32K.json"]:
        file_path = os.path.join(train_dataset_dir, file_name)
        with open(file_path) as f:
            data_list = json.load(f)
            train_sample_list.extend(data_list)
    rng.shuffle(train_sample_list)

    test_sample_list = []
    for file_name in ["hard_8K.json", "hard_16K.json", "hard_32K.json"]:
        file_path = os.path.join(test_dataset_dir, file_name)
        with open(file_path) as f:
            data_list = json.load(f)
            test_sample_list.extend(data_list)
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

    trainer.fit(solver_agent, train_sample_list, val_dataset=test_sample_list)
