import random
import re
from typing import TypedDict, cast, List

from datasets import load_dataset
from openai import AsyncOpenAI
import agentlightning as agl
from agent_pipeline import run_query_pipeline
import jsonlines
import os

verl_config = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_batch_size": 16,
        "max_prompt_length": 4096,
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
            "ppo_mini_batch_size": 64,
            "ppo_micro_batch_size_per_gpu": 8,
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
        "val_before_train": False,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "ParallelAgent",
        "experiment_name": "train_easy_qwen2.5-7b-instruct",
        "nnodes": 1,
        "save_freq": 500,
        "test_freq": 25,
        "total_epochs": 2,
    },
}


class KVProblem(TypedDict):
    chunks: List[str]
    ground_truth: str
    query: str


@agl.rollout
async def kv_agent(task: KVProblem, llm: agl.LLM) -> None:
    # Query LLM endpoint. All queries will be automatically tracked by LLM proxy
    try:
        max_rounds = 3
        answer = await run_query_pipeline(llm, task["chunks"], task["query"], max_rounds)
        answer = answer.strip()
    except Exception as e:
        print("Failure:", str(e))
        answer = ""

    if task["ground_truth"] == answer:
        reward = 1
    else:
        reward = 0

    # This reward will be tracked automatically
    agl.emit_reward(reward)


if __name__ == "__main__":
    dataset_dir = os.path.expanduser("~/parallel_agent_easy")
    train_path = os.path.join(dataset_dir, "train.jsonl")
    test_path = os.path.join(dataset_dir, "test.jsonl")

    train_sample_list = []
    with jsonlines.open(train_path) as reader:
        for j in reader:
            train_sample_list.append(j)

    test_sample_list = []
    with jsonlines.open(test_path) as reader:
        for j in reader:
            test_sample_list.append(j)

    train_dataset = cast(agl.Dataset[KVProblem], train_sample_list)
    val_dataset = cast(agl.Dataset[KVProblem], test_sample_list[:100])

    algorithm = agl.VERL(verl_config)
    # Number of agents launched in parallel to query the LLM.
    # This parameter strongly affects throughput and efficiency:
    # higher parallelism improves utilization but increases GPU overhead.
    n_runners = 32
    # This tracer is a dummy one, as currently tracing is done in the llm proxy part
    tracer = agl.OtelTracer()
    adapter = agl.LlmProxyTraceToTriplet()
    # Set store=None to use managed store
    trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=None, tracer=tracer, adapter=adapter)

    trainer.fit(kv_agent, train_dataset, val_dataset=val_dataset)
