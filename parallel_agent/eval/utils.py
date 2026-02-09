import re
from collections import Counter
import string
import os

def extract_answer(response):
    response = response.replace('*', '')

    if "the answer is" in response:
        ans = response.rsplit("the answer is", 1)[-1].strip().replace("<｜Assistant｜>", '').replace("<｜end▁of▁sentence｜>", '').strip().strip('.').strip()
    else:
        # ans = None will be too strict here, as sometimes the answer will be formulated as "Therefore, the special magic numbers for XXX are"
        # we keep the same test criterion for all methods.
        ans = response.strip()
    return ans

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

### From RULER
def string_match_all(pred, ref):
    return sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)

def calc_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        metrics['sub_em'] += string_match_all(pred, gold)
    metrics['total_num'] = len(goldens)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if s is None:
        return ""

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def sub_exact_match_score(prediction, ground_truth):
    ground_truth = normalize_answer(ground_truth)
    prediction = normalize_answer(prediction) 
    return (ground_truth in prediction) or (prediction in ground_truth)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    subem = sub_exact_match_score(prediction, gold)

    f1, prec, recall = f1_score(prediction, gold)
    metrics['sub_em'] += subem
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    metrics['total_num'] += 1
    return em, prec, recall


def calc_qa_metrics(predictions, goldens):
    assert len(predictions) == len(goldens)
    metrics = {'f1': 0, 'prec': 0, 'recall': 0, 'em': 0, 'sub_em': 0, 'total_num': 0}
    for pred, gold in zip(predictions, goldens):
        update_answer(metrics, pred, gold)
    for k, _ in metrics.items():
        if k == 'total_num':
            continue
        metrics[k] = round((metrics[k]/metrics['total_num']), 2)
    return metrics

def score_func(task, answers, response):
    pred, _ = extract_solution(response)
    final_pred = extract_answer(pred) if pred else extract_answer(response)
    if "qa" in task:
        # TBD: mem agent uses only the first gold answer here. Not sure about the reason. Need to check
        return calc_qa_metrics([final_pred], [answers[0]])
    else:
        sub_em = calc_metrics([final_pred], [answers])['sub_em'] if final_pred else 0
        return {"sub_em": sub_em}


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False 

# evaluation function from gsm infinite
def score_func_gsm_infinite(response, solution):
    # solution 是 sample 中的，给答案提取出来
    idx_answer_start = solution.find("Answer: ") 
    idx_answer_end = solution.find(".", idx_answer_start) 
    answer_text = solution[idx_answer_start + len("Answer: ") : idx_answer_end] 
    answer_text = int(answer_text.lower()) 
        
    response = re.sub('.\x08', 'b', response)
    response = response.lower() 

    idx_generated_begin = -1
    idx_generated_conclude = -1
    keywords = ["answer: ", "solution: ", "oxed{", "**answer:** ", "**answer: ", "final answer: answer: ", "\nanswer: ", r"\text{answer: } ",  "is ", "answer: "] # updated 
    keywordsend = [".", ".", "}", ".", "**", ".", ".", None, ".", "\n"] 
    cnt = 0 

    while not (idx_generated_begin != -1 and idx_generated_conclude != -1) and cnt < len(keywords): 
        if keywords[cnt] in ["oxed{", "is "]: 
            idx_generated_begin = response.rfind(keywords[cnt]) # this relies on the generated is stopped before generated next question plus onwoards by stop 
        else: 
            idx_generated_begin = response.find(keywords[cnt]) 
        if idx_generated_begin != -1: 
            if keywordsend[cnt] is None: 
                idx_generated_conclude = idx_generated_begin + len(keywords[cnt]) 
                while response[idx_generated_conclude].isdigit() == True: 
                    idx_generated_conclude += 1 
            else: 
                idx_generated_conclude = response.find(keywordsend[cnt], idx_generated_begin + len(keywords[cnt])) 
            if idx_generated_conclude == -1: 
                idx_generated_conclude = len(response) 
        if not is_integer(response[idx_generated_begin + len(keywords[cnt]) : idx_generated_conclude]): 
            idx_generated_begin = -1 
            idx_generated_conclude = -1 
        cnt += 1 
    
    if idx_generated_begin == -1: 
        return 0
    else: 
        try: 
            answergenerated_text = int(response[idx_generated_begin + len(keywords[cnt - 1]) : idx_generated_conclude]) 
        except: 
            return 0
        return int(answergenerated_text == answer_text)


def score_func_kv_retrieval(response, answers):
    response = response.strip()
    response = response.replace(" ", "")
    response = response.replace("<answer>", "")
    response = response.replace("</answer>", "")
    response = response.strip()
    response_list = response.split(",")
    response_list = [res.strip() for res in response_list if res.strip()]
    return len(set(response_list).intersection(set(answers))) / len(set(response_list).union(set(answers)))
