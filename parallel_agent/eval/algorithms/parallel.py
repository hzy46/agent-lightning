import asyncio
import re
import random
import aiohttp


async def get_async_client():
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400))


MAX_ROUNDS = 3


chunk_prompt_template = """
You are a chunk-level agent collaborating with a central long-context processing agent to complete a task.

Due to length limitations, we cannot process the entire context in full. However, you can process your assigned chunk and report relevant findings back to the central agent.

Your assigned chunk:
- Chunk index: {chunk_index}
- Total chunks: {chunk_total}
- Content:
{chunk_text}

The central agent's query to you is:
{query}

Instructions:
- If the query explicitly targets a specific chunk index and you are NOT that chunk, output an empty report exactly as: <report></report>
- Otherwise, read your chunk and provide the most useful information you can for answering the query.
 - Do NOT provide the full chunk content. Keep your report concise and only include relevant information.

Output format:
- Put BOTH your thinking and your findings together inside a single block: <report>...</report>
- Do not output anything outside <report></report>
"""


central_prompt_template_ruler = """
You are a central long-context processing agent working on a task.

Because of context length limitations, you cannot directly process the entire context yourself; instead, the context is divided into multiple chunks, and you must communicate with individual chunk agents to request and gather the information you need. 

When doing so, do not ask chunk agents to provide the full chunk content, as this is time-consuming; instead, request only the specific parts that are relevant.

You are given the key you need to find, as well as reports from all previously processed chunks. Based on this information, you must decide whether to answer or to update the query.

First, think through the problem and put your reasoning inside <thinking></thinking>.  
If you can answer, put the final value inside <answer></answer>, and you must format your response as follows "Therefore, the answer is (insert answer here)".  
If you need more information, put your reasoning, the required information, and the corresponding chunk identifiers inside <query></query>.

Current Query:
{query}

Reports from Chunks:
{round_report}

Output exactly two blocks, in this order:
<thinking>...</thinking>
and exactly one of the following:
<answer>Therefore, the answer is (insert answer here).</answer>
or
<query>...</query>
""".strip()

central_prompt_inter_template_ruler = """
Updated Query:
{query}

Reports from Chunks:
{round_report}

Output exactly two blocks, in this order:
<thinking>...</thinking>
and exactly one of the following:
<answer>Therefore, the answer is (insert answer here)</answer>
or
<query>...</query>
""".strip()



central_prompt_template_gsm_infinite = """
You are a central long-context processing agent working on a task.

Because of context length limitations, you cannot directly process the entire context yourself; instead, the context is divided into multiple chunks, and you must communicate with individual chunk agents to request and gather the information you need. 

When doing so, do not ask chunk agents to provide the full chunk content, as this is time-consuming; instead, request only the specific parts that are relevant.

You are given the key you need to find, as well as reports from all previously processed chunks. Based on this information, you must decide whether to answer or to update the query.

First, think through the problem and put your reasoning inside <thinking></thinking>.  
If you can answer, put the final value inside <answer></answer>, and you must format your response as follows "Answer: <the answer here>."
If you need more information, put your reasoning, the required information, and the corresponding chunk identifiers inside <query></query>.

Current Query:
{query}

Reports from Chunks:
{round_report}

Output exactly two blocks, in this order:
<thinking>...</thinking>
and exactly one of the following:
<answer>Answer: <the answer here>.</answer>
or
<query>...</query>
""".strip()

central_prompt_inter_template_gsm_infinite = """
Updated Query:
{query}

Reports from Chunks:
{round_report}

Output exactly two blocks, in this order:
<thinking>...</thinking>
and exactly one of the following:
<answer>Answer: <the answer here>.</answer>
or
<query>...</query>
""".strip()


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return match.group(1).strip() if match else None


async def call_llm(api_root_url, model, temperature, messages, log_dict) -> str:
    MAX_NEW = 1024
    top_p = 1
    session = await get_async_client()
    async with session:
        async with session.post(
            url= api_root_url + "/chat/completions",
            headers={"Authorization": f"Bearer dummy"},
            json=dict(model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=MAX_NEW,
            )
        ) as resp:
            status = resp.status
            if status!= 200:
                print(f"{status=}, {model=}")
                return ""
            data = await resp.json()
            response = data['choices'][0]['message']['content']
            log_dict["all_responses"].append(response)
            return response

async def run_chunk_agent(api_root_url, model, temperature, chunk_text: str, query: str, idx: int, total: int, log_dict) -> str:
    prompt = chunk_prompt_template.format(
        chunk_index=idx,
        chunk_total=total,
        chunk_text=chunk_text,
        query=query
    )
    messages = [{"role": "user", "content": prompt}]
    output = await call_llm(api_root_url, model, temperature, messages, log_dict)

    # if random.random() <= 0.001:
        # print("\n------chunk agent log starts-----\n", prompt, "\n*\n", output, "\n-------chunk agent log ends------\n")

    report = extract_tag(output, "report")
    if report:
        return f"[Chunk {idx} Report]\n{report}"
    return ""

async def run_central_agent(api_root_url, model, temperature, query, round_report, history_messages, task_type, log_dict) -> dict:

    if task_type == "ruler" or task_type == "memagent_train":
        central_prompt_template = central_prompt_template_ruler
        central_prompt_inter_template = central_prompt_inter_template_ruler
    elif task_type == "gsm_infinite":
        central_prompt_template = central_prompt_template_gsm_infinite
        central_prompt_inter_template = central_prompt_inter_template_gsm_infinite
    if len(history_messages) == 0:
        history_messages.append({"role": "user", "content": central_prompt_template.format(
            query=query,
            round_report=round_report
        )})
    else:
        history_messages.append({"role": "user", "content": central_prompt_inter_template.format(
            query=query,
            round_report=round_report
        )})
    output = await call_llm(api_root_url, model, temperature, history_messages, log_dict)

    # if random.random() <= 0.01:
        # print("\n-------central agent log starts-------\n", "\n*\n".join([entry["content"] for entry in history_messages]), "\n*\n", output, "\n-------central agent log ends-----\n")

    history_messages.append({"role": "assistant", "content": output})

    answer = extract_tag(output, "answer")
    if answer is not None:
        return {"type": "answer", "content": answer}

    new_query = extract_tag(output, "query")
    if new_query is not None:
        return {"type": "query", "content": new_query}

    return {"type": "answer", "content": ""}


async def run_query_pipeline(api_root_url, model, temperature, chunks: list[str], query: str, task_type, max_rounds=3) -> tuple[str, list]:
    current_query = query
    history_messages = []
    log_dict = {
        "all_responses": []
    }
    for round_idx in range(1, max_rounds + 1):
        tasks = [
            run_chunk_agent(api_root_url, model, temperature, chunk, current_query, i + 1, len(chunks), log_dict)
            for i, chunk in enumerate(chunks)
        ]
        chunk_reports = await asyncio.gather(*tasks)
        # 汇总本轮 report
        round_report = "\n\n".join([r for r in chunk_reports if r])

        # 跑 query agent
        result = await run_central_agent(api_root_url, model, temperature, current_query, round_report, history_messages, task_type, log_dict)

        if result["type"] == "answer":
            return result["content"], history_messages

        # 更新 query
        current_query = result["content"]

    return "", history_messages, log_dict["all_responses"]


async def async_fill_in_response(api_root_url, sample, model, tokenizer, task_type, temperature=0, chunk_size=5000, max_rounds=3):
    if task_type == "ruler":
        context = sample["context"].strip()
        query = sample['input'].strip()
    elif task_type == "gsm_infinite":
        context = sample["context"].strip()
        query = sample['query'].strip()
    elif task_type == "memagent_train":
        context = sample["context"].strip()
        query = sample['input'].strip()
    else:
        raise NotImplementedError

    input_ids = tokenizer.encode(context, add_special_tokens=False)
    chunks = []
    # print("chunk_size", chunk_size)
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i + chunk_size]
        chunk = tokenizer.decode(chunk_ids)
        chunks.append(chunk)

    response, history_messages, all_responses = await run_query_pipeline(api_root_url, model, temperature, chunks, query, task_type, max_rounds)
    sample["history_messages"] = history_messages
    sample["response"] = response

    # for token penalty, this may affect timing
    output_token_num = 0
    for response in all_responses:
        output_token_num += len(tokenizer.encode(response, add_special_tokens=False))
    sample["output_token_num"] = output_token_num

async def async_fill_in_response_with_sem(semaphore, api_root_url, sample, model, tokenizer, task_type, temperature=0, chunk_size=5000, max_rounds=3):
    async with semaphore:
        await async_fill_in_response(api_root_url, sample, model, tokenizer, task_type, temperature, chunk_size, max_rounds)

