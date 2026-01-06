import requests
import aiohttp

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

normal_prompt_template = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

$Q$

Format your response as follows: "Therefore, the answer is (insert answer here)"."""



memagent_template = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

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

memagent_template_final = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and format your response as follows "Therefore, the answer is (insert answer here)".

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

mem_agent_no_memory = "No previous memory"


async def _memagent_async_get_pred_for_sample(api_root_url, sample, model, tokenizer, temperature, top_p):
    RECURRENT_CHUNK_SIZE = 5000
    RECURRENT_MAX_NEW = 1024
    API_KEY = "dummy"

    context = sample["context"].strip()
    prompt = sample['input'].strip()
    session = await get_async_client()
    async with session:
        input_ids = tokenizer.encode(context)
        memory = mem_agent_no_memory
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            chunk = input_ids[i:i+RECURRENT_CHUNK_SIZE]
            msg = memagent_template.format(prompt=prompt, chunk=tokenizer.decode(chunk), memory=memory)
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
                        return ''
                    data = await resp.json()
                    memory, _ = extract_solution(data['choices'][0]['message']['content'])
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                return ''
        msg = memagent_template_final.format(prompt=prompt, memory=memory)
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
                    return ''
                data = await resp.json()
                return data['choices'][0]['message']['content']
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback
            traceback.print_exc()
        return ''

async def memagent_async_get_pred_for_sample(api_root_url, sample, model, tokenizer, temperature, top_p):
    response = await _memagent_async_get_pred_for_sample(api_root_url, sample, model, tokenizer, temperature, top_p)
    sample["response"] = response


def normal_get_pred_for_sample(api_root_url, model, sample, temperature, top_p):
    max_new_tokens = 8192
    context = sample["context"]
    input = sample["input"]
    prompt = normal_prompt_template.replace('$DOC$', context.strip()).replace('$Q$', input.strip())
    r = requests.post(
        url=api_root_url + "/chat/completions",
        headers={"Authorization": f"Bearer dummy"},
        json=dict(model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens
        )
    )
    r.raise_for_status()
    data = r.json()
    sample["response"] = data['choices'][0]['message']['content']


