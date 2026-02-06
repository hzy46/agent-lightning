import requests
import aiohttp
import os

async def get_async_client():
    return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400))

def extract_solution(solution_str):
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
  
    # Extract final answer using XML-style tags
    if "</think>" not in solution_str:
        if not os.environ.get("FORCE_THINK"):
            return solution_str, solution_str
        else:
            print("[Error] No valid answer tags found")
            return None, solution_str 
    final_answer = solution_str.split("</think>")[-1].strip()
    return final_answer, solution_str



template = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

template_final_ruler = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and format your response as follows "Therefore, the answer is (insert answer here)".

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""



template_final_gsm_infinite = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>
"""

template_final_qa = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Format your answer in \\boxed{{}}, e.g. \\boxed{{your answer}}.
"""



no_memory = "No previous memory"


async def _async_fill_in_response(api_root_url, sample, model, tokenizer, task_type, temperature=0, chunk_size=5000):
    RECURRENT_CHUNK_SIZE = chunk_size
    RECURRENT_MAX_NEW = 1024
    top_p = 1
    API_KEY = "dummy"

    if task_type == "ruler":
        template_final = template_final_ruler
        context = sample["context"].strip()
        prompt = sample['input'].strip()
    elif task_type == "gsm_infinite":
        template_final = template_final_gsm_infinite
        context = sample["context"].strip()
        prompt = sample['query'].strip()
    elif task_type == 'qa':
        template_final = template_final_qa
        context = sample["context"].strip()
        prompt = sample['query'].strip()
    else:
        raise NotImplementedError

    session = await get_async_client()
    history_memory_list = []
    async with session:
        input_ids = tokenizer.encode(context)
        memory = no_memory
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            chunk = input_ids[i:i + RECURRENT_CHUNK_SIZE]
            msg = template.format(prompt=prompt, chunk=tokenizer.decode(chunk), memory=memory)
            try:
                async with session.post(
                    url= api_root_url + "/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json=dict(model=model,
                        messages=[{"role": "user", "content": msg}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=RECURRENT_MAX_NEW
                    )
                ) as resp:
                    status = resp.status
                    if status!= 200:
                        print(f"{status=}, {model=}")
                        return '', history_memory_list
                    data = await resp.json()
                    memory, _ = extract_solution(data['choices'][0]['message']['content'])
                    history_memory_list.append(memory)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                return '', history_memory_list
        msg = template_final.format(prompt=prompt, memory=memory)
        try:
            async with session.post(
                url= api_root_url + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=dict(model=model,
                    messages=[{"role": "user", "content": msg}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=RECURRENT_MAX_NEW
                )
            ) as resp:
                status = resp.status
                if status!= 200:
                    print(f"{status=}, {model=}")
                    return '', history_memory_list
                data = await resp.json()
                return data['choices'][0]['message']['content'], history_memory_list
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback
            traceback.print_exc()
        return '', history_memory_list

async def async_fill_in_response_with_sem(semaphore, api_root_url, sample, model, tokenizer, task_type, temperature=0, chunk_size=5000):
    async with semaphore:
        response, history_memory_list = await _async_fill_in_response(api_root_url, sample, model, tokenizer, task_type, temperature, chunk_size)
        sample["history_memory_list"] = history_memory_list
        sample["response"] = response


async def async_fill_in_response(api_root_url, sample, model, tokenizer, task_type, temperature=0, chunk_size=5000):
    response, history_memory_list = await _async_fill_in_response(api_root_url, sample, model, tokenizer, task_type, temperature, chunk_size)
    sample["history_memory_list"] = history_memory_list
    sample["response"] = response


