import requests


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
        max_new_tokens = 4096
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


