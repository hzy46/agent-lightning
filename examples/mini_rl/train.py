import re
from typing import TypedDict, cast

from datasets import load_dataset
from openai import AsyncOpenAI

import agentlightning as agl

verl_config = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_batch_size": 32,
        "max_prompt_length": 512,
        "max_response_length": 1024,
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.6,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 32,
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
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "Qwen/Qwen2.5-1.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "mini_rl_gsm8k",
        "nnodes": 1,
        "save_freq": 64,
        "test_freq": 32,
        "total_epochs": 2,
    },
}


class Gsm8kProblem(TypedDict):
    question: str
    answer: str


ds = load_dataset("openai/gsm8k", "main")

prompt_template = """
You are given the following question:

{}

Please think step by step and put your final answer after ####.

Output example:

<thinking process>
#### <your answer>
""".strip()

train_dataset = cast(agl.Dataset[Gsm8kProblem], ds["train"].to_list())
val_dataset = cast(agl.Dataset[Gsm8kProblem], ds["test"].to_list())


@agl.rollout
async def gsm8k_agent(task: Gsm8kProblem, llm: agl.LLM) -> None:
    model = llm.model
    openai_base_url = llm.endpoint
    temperature = llm.sampling_parameters.get("temperature", 1.0)
    client = AsyncOpenAI(
        api_key="dummy",
        base_url=openai_base_url,
    )
    regex_pattern = r"####\s*ANSWER:\s*(.+?)(\s*####|$)"
    try:
        prompt = prompt_template.format(task["question"])
        messages = [{"role": "user", "content": prompt}]
        response = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )
        last_message = response.choices[0].message.content

        answer = re.search(regex_pattern, last_message)
        if answer:
            answer = answer.group(1)
        else:
            answer = last_message
    except Exception as e:
        print("Failure:", str(e))
        answer = "None"
    gt_answer = re.search(regex_pattern, task["answer"]).group(1)
    if gt_answer == answer:
        reward = 1
    else:
        reward = 0
    agl.emit_reward(reward)


algorithm = agl.VERL(verl_config)
n_runners = 24

# this tracer is a dummy one, as currently tracing is done in the proxy part
tracer = agl.OtelTracer()
adapter = agl.LlmProxyTraceToTriplet()
# set store=None to use
trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=None, tracer=tracer, adapter=adapter)

trainer.fit(gsm8k_agent, train_dataset, val_dataset=val_dataset)
