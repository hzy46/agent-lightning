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




def fill_response_for_sample_ruler(api_root_url, model, sample):
    max_new_tokens = 8192
    temperature = 0
    top_p = 1
    context = sample["context"]
    input = sample["input"]
    prompt = prompt_template_ruler.replace('$DOC$', context.strip()).replace('$Q$', input.strip())
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


def fill_response_for_sample_gsm_infinite(api_root_url, model, sample):
    max_new_tokens = 4096
    temperature = 0
    top_p = 1
    context = sample["context"]
    query = sample["query"]
    prompt = prompt_template_gsm_infinite.replace('$DOC$', context.strip()).replace('$Q$', input.strip())
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


