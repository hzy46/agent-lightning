import openai
import jsonlines
import asyncio
import re
from openai import AsyncOpenAI
import random


chunk_prompt_template = """
You are a chunk-level agent collaborating with a central long-context processing agent to complete a key-value retrieval task.

The key and its corresponding value may be fully contained within a single chunk, or they may be split across multiple chunks. For example, if "key: <xxxx> value:" appears at the end of chunk 2, the corresponding value may start at the beginning of chunk 3.

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
  - If you detect that a key/value pair is split across chunk boundaries, report the relevant beginning or ending fragments that would help the central agent stitch the value together.
  - Do NOT provide the full chunk content. Keep your report concise and only include information relevant to the query and the key-value retrieval task. Use abbreviations or short excerpts where appropriate.

Output format:
- Put BOTH your thinking and your findings together inside a single block: <report>...</report>
- Do not output anything outside <report></report>
"""


central_prompt_template = """
You are a central long-context processing agent working on a key-value retrieval task.

Because of context length limitations, you cannot directly process the entire context yourself; instead, the context is divided into multiple chunks, and you must communicate with individual chunk agents to request and gather the information you need. When doing so, do not ask chunk agents to provide the full chunk content, as this is time-consuming; instead, request only the specific parts that are relevant (for example, the beginning or the end of a chunk).

The key and its corresponding value may be fully contained within a single chunk, or they may be split across multiple chunks. For example, if "key: <xxxx> value:" appears at the end of chunk 2, the corresponding value may start at the beginning of chunk 3. In such cases, you must correctly combine information from multiple chunks to retrieve the complete value.

You are given the key you need to find, as well as reports from all previously processed chunks. Based on this information, you must decide whether to answer or to update the query.

If you answer, you must provide the complete and correct value that you have found.  
If you update the query, you may request specific chunks and specify which part of the chunk is needed, such as the beginning or the end.

First, think through the problem and put your reasoning inside <thinking></thinking>.  
If you can answer, put the final value inside <answer></answer> without anything else.  
If you need more information, put your reasoning, the required information, and the corresponding chunk identifiers inside <query></query>.

Current Query:
{query}

Reports from Chunks:
{round_report}

Output exactly two blocks, in this order:
<thinking>...</thinking>
and exactly one of the following:
<answer>...</answer>
or
<query>...</query>
""".strip()

central_prompt_inter_template = """
Updated Query:
{query}

Reports from Chunks:
{round_report}

Output exactly two blocks, in this order:
<thinking>...</thinking>
and exactly one of the following:
<answer>...</answer>
or
<query>...</query>
""".strip()



def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return match.group(1).strip() if match else None


async def call_llm(llm, messages) -> str:
    # llm is from agl, contains endpoint, model name, and temperature
    model = llm.model
    openai_base_url = llm.endpoint
    temperature = llm.sampling_parameters.get("temperature", 1.0)

    client = AsyncOpenAI(
        api_key="dummy",
        base_url=openai_base_url,
    )

    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


async def run_chunk_agent(llm, chunk_text: str, query: str, idx: int, total: int) -> str:
    prompt = chunk_prompt_template.format(
        chunk_index=idx,
        chunk_total=total,
        chunk_text=chunk_text,
        query=query
    )
    messages = [{"role": "user", "content": prompt}]
    output = await call_llm(llm, messages)

    if random.random() <= 0.001:
        print("\n------chunk agent log starts-----\n", prompt, "\n*\n", output, "\n-------chunk agent log ends------\n")

    report = extract_tag(output, "report")
    if report:
        return f"[Chunk {idx} Report]\n{report}"
    return ""

async def run_central_agent(llm, query, round_report, history_messages) -> dict:
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
    output = await call_llm(llm, history_messages)

    if random.random() <= 0.01:
        print("\n-------central agent log starts-------\n", "\n*\n".join([entry["content"] for entry in history_messages]), "\n*\n", output, "\n-------central agent log ends-----\n")

    history_messages.append({"role": "assistant", "content": output})

    answer = extract_tag(output, "answer")
    if answer is not None:
        return {"type": "answer", "content": answer}

    new_query = extract_tag(output, "query")
    if new_query is not None:
        return {"type": "query", "content": new_query}

    return {"type": "answer", "content": ""}


async def run_query_pipeline(llm, chunks: list[str], query: str, max_rounds: int) -> str:
    current_query = query
    history_messages = []

    for round_idx in range(1, max_rounds + 1):
        tasks = [
            run_chunk_agent(llm, chunk, current_query, i + 1, len(chunks))
            for i, chunk in enumerate(chunks)
        ]
        chunk_reports = await asyncio.gather(*tasks)
        # 汇总本轮 report
        round_report = "\n\n".join([r for r in chunk_reports if r])

        # 跑 query agent
        result = await run_central_agent(llm, current_query, round_report, history_messages)

        if result["type"] == "answer":
            return result["content"]

        # 更新 query
        current_query = result["content"]

    return ""