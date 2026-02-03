import os
import requests
import sys
import json
import random
from multiprocessing import Pool
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path

# code adapted from memagent
# Global variables
QAS = None
DOCS = None
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# SQuAD dataset processing
def read_squad(file):
    with open(file) as f:
        data = json.load(f)
        
    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]]
                    })
    return total_qas, total_docs

# HotpotQA dataset processing
def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)
    
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    
    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })
    return total_qas, total_docs

def generate_input_output(index, num_docs):
    global QAS, DOCS
    curr_q = QAS[index]['query']
    curr_a = QAS[index]['outputs']
    curr_docs = QAS[index]['context']
    curr_more = QAS[index].get('more_context', [])
    
    if num_docs < len(DOCS):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        all_docs = DOCS

    # fix issue    
    rnd = random.Random(index)
    rnd.shuffle(all_docs)
    # random.Random(4).shuffle(all_docs)

    DOCUMENT_PROMPT = "Document {i}:\n{document}"
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])


    # new format
    formatted_output = {
        "task_type": "qa",
        "query": curr_q,
        "context": context,
        # for multiple answer
        "ground_truths": [str(a) for a in curr_a.tolist()],
        "num_docs": num_docs,
    }
    return formatted_output


def generate_json(num_samples: int, incremental: int = 10, qas=None, docs=None):
    global QAS, DOCS
    if qas is None or docs is None:
        raise ValueError("QAS and DOCS must be provided.")
    
    QAS = qas
    DOCS = docs
    
    length = min(num_samples, len(QAS))
    print("start")
    
    from utils import TqdmExecutor
    write_jsons = TqdmExecutor(max_workers=os.cpu_count()).run(generate_input_output, range(length), num_docs=incremental)

    save_dir = os.path.expanduser("~/qa_eval_more")
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"test_doc{incremental}.json")
    with open(save_path, "w") as f:
        json.dump(write_jsons, f)



def download_hotpotqa():
    files = {
        "hotpotqa_dev.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        # "hotpotqa_train.json": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
    }

    chunk_size = 1024 * 1024  # 1 MB

    for filename, url in files.items():
        if os.path.exists(filename):
            print(f"[OK] {filename} 已存在，跳过下载")
            continue

        print(f"[DOWNLOAD] {filename}")

        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length", 0))
            downloaded = 0

            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = downloaded * 100 / total_size
                        bar_len = 30
                        filled = int(bar_len * percent / 100)
                        bar = "=" * filled + "-" * (bar_len - filled)
                        sys.stdout.write(
                            f"\r  [{bar}] {percent:6.2f}% ({downloaded/1024/1024:.1f} MB)"
                        )
                        sys.stdout.flush()

        print("\n[DONE]", filename)





if __name__ == "__main__":
    download_hotpotqa()
    random.seed(42)
    num_samples = 200
    # 100 is roughly 16K
    doc_num_list = [100]
    QAS_dev, DOCS_dev = read_hotpotqa('hotpotqa_dev.json')
    print("overall dev doc num:", len(DOCS_dev))
    for doc_num in doc_num_list:
        generate_json(num_samples, doc_num, QAS_dev, DOCS_dev)