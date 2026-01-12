import requests
import aiohttp
import os


prompt_template_ruler = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

$Q$

Format your response as follows: "Therefore, the answer is (insert answer here)"."""




prompt_template_gsm_infinite = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

$Q$"""



def fill_in_response(api_root_url, model, sample, task_type):
    if task_type == "ruler":
        max_new_tokens = 8192
        temperature = 0
        top_p = 1
        context = sample["context"]
        query = sample["input"]
        prompt_template = prompt_template_ruler
    elif task_type == "gsm_infinite":
        max_new_tokens = 2048
        temperature = 0
        top_p = 1
        context = sample["context"]
        query = sample["query"]
        prompt_template = prompt_template_gsm_infinite
    else:
        raise NotImplementedError

    prompt = prompt_template.replace('$DOC$', context.strip()).replace('$Q$', query.strip())
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



async def _async_fill_in_response(api_root_url, model, sample, task_type, temperature=0):
    if task_type == "ruler":
        max_new_tokens = 8192
        top_p = 1
        context = sample["context"]
        query = sample["input"]
        prompt_template = prompt_template_ruler
    elif task_type == "gsm_infinite":
        max_new_tokens = 2048
        top_p = 1
        context = sample["context"]
        query = sample["query"]
        prompt_template = prompt_template_gsm_infinite
    else:
        raise NotImplementedError

    prompt = prompt_template.replace('$DOC$', context.strip()).replace('$Q$', query.strip())
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400))
    async with session:
        try:
            async with session.post(
                url= api_root_url + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=dict(model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                )
            ) as resp:
                status = resp.status
                if status!= 200:
                    print(f"{status=}, {model=}")
                    return ''
                data = await resp.json()
                return data['choices'][0]['message']['content']
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ''
            

async def async_fill_in_response(api_root_url, model, sample, task_type, temperature=0):
    response = await _async_fill_in_response(api_root_url, model, sample, task_type, temperature)
    sample["response"] = response
