import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from setproctitle import setproctitle
import argparse
from tqdm import tqdm
import re
from openai import OpenAI
from huggingface_hub import login



# ================== 1. Functions ==================
def model_load(key, model_path):
    print(f"Loading model: {model_path}")

    if 'gpt' in model_path.lower():
        if not key:
            raise ValueError("OpenAI 모델 사용 시 --key 필요합니다.")
        client = OpenAI(api_key=key)
        return client
    else:
        if key: login(token=key)    # HuggingFace Model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto"
        )
        return tokenizer, model

def create_messages(mode, item):
    if mode == "default":
        options_str = "\n".join([f"{k}. {v}" for k, v in item["Options"].items()])
        system_message_df = "You are acting as a survey respondent. From the given options, choose exactly one. Do not provide any explanation or extra text."
        messages = [
            {"role": "system", "content": system_message_df},
            {"role": "user", "content": f"Question: {item['Questions']}\nOptions: {options_str}"}
        ]
    else:
        options_str_kr = "\n".join([f"{k}. {v}" for k, v in item["Options_KR"].items()])
        system_message_kr = "당신은 설문 조사 지원자의 역할을 수행합니다. 제공된 선택지 중 반드시 하나만 선택하고, 아무 설명도 하지 말고 선택한 번호만 숫자 형식으로 응답하십시오. 당신이 한국인이라면 어떻게 답변할지, 한국인의 관점으로 선택합니다."
        messages = [
            {"role": "system", "content": system_message_kr},
            {"role": "user", "content": f"질문: {item['Questions_KR']}\n선택지: {options_str_kr}"}
        ]

    return messages

def preprocess_response(options, response):
    def extract_first_number(r: str):
        # 문자열에서 처음 나오는 숫자를 찾아 int로 반환. 없으면 None
        match = re.search(r"\d+", r)
        if match:
            return int(match.group())
        return None

    """
    - 음수 옵션 제외
    - 숫자가 있으면 숫자 사용
    - 숫자가 없더라도 options의 value 값과 같으면 해당 key로 매핑
    - 추출 실패 시 None
    """
    pos_keys = [int(k) for k in options.keys() if int(k) > 0]
    if not pos_keys: return ""

    r_int = None
    # 1. 정수 변환 시도
    try:
        r_int = int(response)
    except ValueError:
        # 2. 문자열에서 첫 숫자 추출
        r_int = extract_first_number(response)

    # 3. 옵션 value 값과 exact match 확인
    if r_int is None:
        matched = [k for k, v in pos_keys.items() if v.strip().lower() == str(response).strip().lower()]
        if matched:
            r_int = matched[0]

    # 4. 유효성 체크
    if r_int is None or int(r_int) not in pos_keys:
        return ""
    else:
        return r_int

def hf_chat(model, tokenizer, item, mode, max_new_tokens=10, n=30):
    # Input - Messages
    messages = create_messages(mode=mode, item=item)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Output - Generate N responses
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=max_new_tokens, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,             # 다양성 확보 (중요)
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=n
    )

    # Decode
    outputs = []
    prefix_len = model_inputs.input_ids.shape[1]
    for i in range(n):
        output_ids = generated_ids[i][prefix_len:].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        v = preprocess_response(options=item["Options"], response=content)
        if v != "": outputs.append(v)

    return outputs

def gpt_chat(client, item, mode, model_path):
    # Input - Messages
    messages = create_messages(mode=mode, item=item)

    # Output - Generate N responses
    _response = client.chat.completions.create(
        model=model_path,
        messages=messages,
        temperature=0.7,
        top_p=0.9,
        n=30
    )
    all_responses = [choice.message.content for choice in _response.choices]

    options = item["Options"]
    processed = [v for v in (preprocess_response(options, r) for r in all_responses) if v != ""]

    return processed

def run_survey(**kwargs):
    data        = kwargs["data"]                 # dict 또는 로드된 JSON
    model_path  = kwargs["model_path"]           # 문자열 (예: "gpt-4o-mini" 또는 "meta-llama/...")
    mode        = kwargs.get("mode")             # 프롬프팅 평가 모드
    client      = kwargs.get("client")           # GPT용
    model       = kwargs.get("model")            # HF용
    tokenizer   = kwargs.get("tokenizer")        # HF용
    output_dir  = kwargs["output_dir"]           # 저장 경로 (파일 경로)

    responses = {"WVS": {category: [] for category in data["WVS"].keys()}}

    for category, items in data["WVS"].items():
        for item in items:
            item_copy = item.copy()
            item_copy["Responses"] = []
            responses["WVS"][category].append(item_copy)

    os.makedirs(os.path.dirname(output_dir) or ".", exist_ok=True)

    for category, items in tqdm(responses["WVS"].items()):
        for item in items:
            try:
                if 'gpt' in model_path.lower():
                    response = gpt_chat(client, item, mode, model_path)
                else:
                    response = hf_chat(model, tokenizer, item, mode)

                item["Responses"] = response

            except Exception as e:
                print(f"Error with item {item.get('Q_index','?')} at category {category}: {e}")

        with open(output_dir, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)

        print(f"💚 Saved intermediate results after category '{category}' to {output_dir}")

    print(f"💙 Final results saved for {model_path} at {output_dir}")

    return responses

# ================== 2. Arguments ==================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Model name to run survey with")
parser.add_argument("--prompt_mode", type=str, required=True, help="default or korea")
parser.add_argument("--key", type=str, required=False, help='Hugging Face key or OpenAI key')
parser.add_argument("--dataset_path", type=str, required=True, help='Survey Dataset Path')
args = parser.parse_args()

model_path = args.model_path
mode = args.prompt_mode
key = args.key
dataset_path = args.dataset_path

# ================== 3. Data & Model ==================
# Evaluation Data
with open(dataset_path, "r", encoding="utf-8") as f: data = json.load(f)

# Model Load
print(f'Mode: {mode}')
if 'gpt' in model_path.lower(): client = model_load(key, model_path)
else: tokenizer, model = model_load(key, model_path)

# ================== 4. Run ==================
safe_model = model_path.replace("/", "__")
output_dir=f"outputs/{safe_model}/kovalplus_responses_{mode}.json"

if 'gpt' in model_path.lower():
    run_survey(
        data=data,
        model_path=model_path,
        client=client,
        mode=mode,
        output_dir=output_dir,
    )
else: 
    run_survey(
        data=data,
        model_path=model_path,
        model=model,
        tokenizer=tokenizer,
        mode=mode,
        output_dir=output_dir,
    )

# ================== 5. Eval - Measuring similarity: Korean vs. model responses ==================
from eval import run as eval_run

with open(output_dir, "r", encoding="utf-8") as f: survey_data = json.load(f)
eval_run(data=survey_data, out_dir=f"outputs/{safe_model}/", mode=mode)