import asyncio
import re
import random
import aiohttp


async def get_async_client():
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400))


start_prompt_template = """
You are a chunk-level agent collaborating with other chunk-level agents to process a long-text task.

Due to length limitations, we cannot process the entire context at once. However, you can process your assigned chunk and then broadcast the relevant information to other agents so that everyone can work together to solve the problem.

You are currently processing chunk {chunk_index} out of a total of {chunk_total} chunks.

Chunk content:
{chunk_text}

The current long-text question is:

{query}

You may output in one of the following formats:

1. <thinking>...</thinking>
Use this when you do not find any information related to the query in this chunk, but you are willing to wait for new information broadcast by other agents.

2. <thinking>...</thinking> <broadcast>...</broadcast>
Use this when you find information in this chunk that is related to the query but does not directly answer it. In this case, you choose to broadcast this information to other agents.

3. <thinking>...</thinking> <answer>...</answer>
Use this when you find information in this chunk that is directly relevant to the answer. You choose to answer the query and end the current stream. The content inside <answer></answer> will also be broadcast to other agents.

Note: There may be more than one piece of information related to the query. Try to find as many as possible and make full use of the broadcast mechanism to handle multi-hop reasoning. Broadcasted information should be as concise and accurate as possible. Try not to answer too early.
"""


turn_prompt_template = """
You are currently processing chunk {chunk_index} out of a total of {chunk_total} chunks.

In this round, you have received the following broadcast information:

{received_broadcast_information}

You have also received the following answer information:

{received_answer_information}

Continue and output in one of the following three formats:
1. <thinking>...</thinking> 
2. <thinking>...</thinking> <broadcast>...</broadcast> 
3. <thinking>...</thinking> <answer>...</answer>
"""


one_broadcast_info_template = """
Information from chunk {chunk_index}:

{information}
""".strip()


one_answer_info_template = """
Answer from chunk {chunk_index}:

{information}
""".strip()


answer_prompt_template_gsm_inifinite = """
You are working on a long-text task. The full text has been split into {chunk_total} chunks, and {chunk_total} agents are collaborating to answer the following question.

Query:
{query}

Agent Collaboration Stream:
{stream_content}

You need to extract all content enclosed in <answer></answer> from the agentic stream, synthesize them into a final answer, and output the result.

Your output format should be:
<answer>Answer: <the answer here>.</answer>
"""


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return match.group(1).strip() if match else None


class ModelConfig(object):

    def __init__(self, api_root_url, model, temperature=0, max_new=1024):
        self.api_root_url = api_root_url
        self.model = model
        self.temperature = temperature
        self.max_new = max_new

class AlgorithmConfig(object):

    def __init__(self, max_rounds, chunk_size, fix_chunk_num):
        self.max_rounds = max_rounds
        self.chunk_size = chunk_size
        self.fix_chunk_num = fix_chunk_num


class Stream(object):

    def __init__(self, context, query, chunk_index, chunk_total):
        self.context = context
        self.query = query
        self.chunk_index = chunk_index
        self.chunk_total = chunk_total
        self.is_finished = False
        self.messages = []
        self.broadcast = ""
        self.answer = ""
        self.received_broadcast_information = ""
        self.received_answer_information = ""

    async def run(self, model_config, log_dict):
        self.broadcast = ""
        self.answer = ""
        if self.is_finished:
            return
        if len(self.messages) == 0:
            self.messages.append({
                "role": "user",
                "content": start_prompt_template.format(
                    chunk_index=self.chunk_index,
                    chunk_total=self.chunk_total,
                    chunk_text=self.context,
                    query=self.query,
                )
            })
        else:
            if self.received_broadcast_information == "" and self.received_answer_information == "":
                # 此时也没有跑的必要，因为没有任何新信息，如果所有 stream 都是这个状态则直接结束
                # 注意，这个会导致在有些 round 某些 stream 处于暂停状态
                return
            if self.received_broadcast_information == "":
                received_broadcast_information = "N/A"
            else:
                received_broadcast_information = self.received_broadcast_information

            if self.received_answer_information == "":
                received_answer_information = "N/A"
            else:
                received_answer_information = self.received_answer_information
            self.messages.append({
                "role": "user",
                "content": turn_prompt_template.format(
                    chunk_index=self.chunk_index,
                    chunk_total=self.chunk_total,
                    received_broadcast_information=received_broadcast_information,
                    received_answer_information=received_answer_information,
                )
            })
            # clear as it is used
            self.received_broadcast_information = ""
            self.received_answer_information = ""

        response = await call_llm(model_config, self.messages, log_dict)
        self.messages.append({
            "role": "assistant",
            "content": response,
        })

        # core status
        answer = extract_tag(response, "answer")
        broadcast = extract_tag(response, "broadcast")
        if answer is None:
            self.answer = ""
        else:
            self.answer = answer
        # if the stream decides to answer, end it
        if self.answer != "":
            self.is_finished = True
            return
        if broadcast is None:
            self.broadcast = ""
        else:
            self.broadcast = broadcast


    def update(self, received_broadcast_information, received_answer_information):
        if self.is_finished:
            return
        self.received_broadcast_information = received_broadcast_information
        self.received_answer_information = received_answer_information

    def format_content(self):
        content_list = []
        for i in range(1, len(self.messages)):
            message = self.messages[i]
            if message["role"] == "user":
                content_list.append("User: {}".format(message["content"]))
            else:
                content_list.append("Assistant: {}".format(message["content"]))
        return '\n'.join(content_list)      


async def call_llm(model_config, messages, log_dict) -> str:
    top_p = 1
    session = await get_async_client()
    async with session:
        async with session.post(
            url= model_config.api_root_url + "/chat/completions",
            headers={"Authorization": f"Bearer dummy"},
            json=dict(model=model_config.model,
                messages=messages,
                temperature=model_config.temperature,
                top_p=top_p,
                max_tokens=model_config.max_new,
            )
        ) as resp:
            status = resp.status
            if status!= 200:
                print(f"{status=}, model={model_config.model}")
                return ""
            data = await resp.json()
            response = data['choices'][0]['message']['content']
            log_dict["all_responses"].append(response)
            return response



async def run_query_pipeline(model_config, algorithm_config, chunks, query, task_type) :
    log_dict = {
        "all_responses": []
    }
    streams = [
        Stream(context=chunk, query=query, chunk_index=chunk_i + 1, chunk_total=len(chunks))
        for chunk_i, chunk in enumerate(chunks)
    ]
    for round_idx in range(1, algorithm_config.max_rounds + 1):
        tasks = [
            stream.run(model_config, log_dict)
            for stream in streams
        ]
        await asyncio.gather(*tasks)
        # for each stream, update the received information
        # 结束条件：所有 stream 都 finished 或 无 broadcast 和 answer
        is_end = True
        for stream in streams:
            if not(stream.is_finished):
                broadcast_info_text_list = []
                answer_info_text_list = []
                for other_stream in streams:
                    if other_stream.chunk_index == stream.chunk_index:
                        continue
                    if other_stream.broadcast != "":
                        broadcast_info_text_list.append(
                            one_broadcast_info_template.format(chunk_index=other_stream.chunk_index, information=other_stream.broadcast)
                        )
                    if other_stream.answer != "":
                        answer_info_text_list.append(
                            one_answer_info_template.format(chunk_index=other_stream.chunk_index, information=other_stream.answer)
                        )
                broadcast_info = "\n\n".join(broadcast_info_text_list)
                answer_info = "\n\n".join(answer_info_text_list)
                if broadcast_info != "" or answer_info != "":
                    is_end = False
                stream.update(broadcast_info, answer_info)
        if is_end:
            break

    stream_content_list = []
    for stream in streams:
        stream_content_list.append("Chunk {}\nLog:{}\n".format(
            stream.chunk_index,
            stream.format_content()
        ))
    stream_content = "\n\n".join(stream_content_list)

    if task_type == "gsm_infinite":
        answer_prompt_template = answer_prompt_template_gsm_inifinite
    else:
        raise NotImplementedError

    response = await call_llm(
        model_config,
        [{"role": "user", "content": answer_prompt_template.format(
            chunk_total=len(chunks),
            stream_content=stream_content,
            query=query
        )}],
        log_dict,
    )
    answer = extract_tag(response, "answer")
    if answer is None:
        answer = ""

    return answer, stream_content, log_dict["all_responses"]



async def async_fill_in_response(model_config, tokenizer, algorithm_config, sample, task_type):
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
    if algorithm_config.fix_chunk_num is not None:
        chunk_size = len(input_ids) // algorithm_config.fix_chunk_num + 1
    else:
        chunk_size = algorithm_config.chunk_size

    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i:i + chunk_size]
        chunk = tokenizer.decode(chunk_ids)
        chunks.append(chunk)

    response, history_messages, all_responses = await run_query_pipeline(model_config, algorithm_config, chunks, query, task_type)
    sample["history_messages"] = history_messages
    sample["response"] = response

    # for token penalty, this may affect timing
    output_token_num = 0
    for response in all_responses:
        output_token_num += len(tokenizer.encode(response, add_special_tokens=False))
    sample["output_token_num"] = output_token_num

async def async_fill_in_response_with_sem(semaphore, model_config, tokenizer, algorithm_config, sample, task_type):
    async with semaphore:
        await async_fill_in_response(model_config, tokenizer, algorithm_config, sample, task_type)

