from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.gpt import GPT


TOKEN_NUM: int = 128
TEMPERATURE: float = 0.1
TOP_P: float = 1
SYSTEM_PROMPT = "以下に、日本の妖怪に関する質問をする指示があります。質問に対する回答を記述してください。"


def func_gpt4o(prompt: str) -> str:
    return _func_gpt(prompt, "gpt-4o", "2024-08-06")


def func_gpt4o_mini(prompt: str) -> str:
    return _func_gpt(prompt, "gpt-4o-mini", "2024-08-01-preview")


def func_gemma_2_2b_it(prompt: str) -> str:
    return _func_gemma(prompt, "google/gemma-2-2b-it")


def func_gemma_2_9b_it(prompt: str) -> str:
    return _func_gemma(prompt, "google/gemma-2-9b-it")


def func_gemma_2_27b_it(prompt: str) -> str:
    return _func_gemma(prompt, "google/gemma-2-27b-it")


def func_llm_jp_3_1p8b_instruct(prompt: str) -> str:
    return _func_default(prompt, "llm-jp/llm-jp-3-1.8b-instruct")


def func_llm_jp_3_3p7b_instruct(prompt: str) -> str:
    return _func_default(prompt, "llm-jp/llm-jp-3-3.7b-instruct")


def func_llm_jp_3_13b_instruct(prompt: str) -> str:
    return _func_default(prompt, "llm-jp/llm-jp-3-13b-instruct")


def func_llm_jp_3_172b_instruct3(prompt: str) -> str:
    return _func_default(prompt, "llm-jp/llm-jp-3-172b-instruct3")


def func_calm3_22b_chat(prompt: str) -> str:
    return _func_default(prompt, "cyberagent/calm3-22b-chat")


def func_qwen2_7b_instruction(prompt: str) -> str:
    return _func_qwen(prompt, "Qwen/Qwen2-7B-Instruct")


def func_qwen2p5_3b_instruct(prompt: str) -> str:
    return _func_qwen(prompt, "Qwen/Qwen2.5-3B-Instruct")


def func_qwen2p5_7b_instruct(prompt: str) -> str:
    return _func_qwen(prompt, "Qwen/Qwen2.5-7B-Instruct")


def func_qwen2p5_14b_instruct(prompt: str) -> str:
    return _func_qwen(prompt, "Qwen/Qwen2.5-14B-Instruct")


def func_qwen2p5_32b_instruct(prompt: str) -> str:
    return _func_qwen(prompt, "Qwen/Qwen2.5-32B-Instruct")


def func_eurollm_1p7b_instruct(prompt: str) -> str:
    return _func_default(prompt, "utter-project/EuroLLM-1.7B-Instruct")


def func_eurollm_9b_instruct(prompt: str) -> str:
    return _func_default(prompt, "utter-project/EuroLLM-9B-Instruct")


def func_swallow_7b_instruct_v0p1(prompt: str) -> str:
    return _func_default(prompt, "tokyotech-llm/Swallow-7b-instruct-v0.1")


def func_swallow_13b_instruct_v0p1(prompt: str) -> str:
    return _func_default(prompt, "tokyotech-llm/Swallow-13b-instruct-v0.1")


def func_llama_3p1_swallow_8b_instruct_v0p2(prompt: str) -> str:
    return _func_default(prompt, "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2")


def func_llama_3p1_swallow_70b_instruct_v0p1(prompt: str) -> str:
    return _func_default(prompt, "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.1")


def func_tanuki_8x8b_dpo_v1p0(prompt: str) -> str:
    return _func_default(prompt, "weblab-GENIAC/Tanuki-8x8B-dpo-v1.0")


def func_tanuki_8b_dpo_v1p0(prompt: str) -> str:
    return _func_default(prompt, "weblab-GENIAC/Tanuki-8B-dpo-v1.0")


def func_llama_3p1_70b_japanese_instruct_2407(prompt: str) -> str:
    return _func_default(prompt, "cyberagent/Llama-3.1-70B-Japanese-Instruct-2407")


def func_meta_llama_3p1_8b_instruct(prompt: str) -> str:
    return _func_default(prompt, "meta-llama/Meta-Llama-3.1-8B-Instruct")


def func_llama_3p2_3b_instruct(prompt: str) -> str:
    return _func_default(prompt, "meta-llama/Llama-3.2-3B-Instruct")


def func_meta_llama_3_8b_instruct(prompt: str) -> str:
    return _func_default(prompt, "meta-llama/Meta-Llama-3-8B-Instruct")


def func_meta_llama_3p3_70b_instruct(prompt: str) -> str:
    return _func_default(prompt, "meta-llama/Llama-3.3-70B-Instruct")


def func_mistral_7b_instruct_v0p2(prompt: str) -> str:
    return _func_default(prompt, "mistralai/Mistral-7B-Instruct-v0.2")


def func_mistral_nemo_instruct_2407(prompt: str) -> str:
    return _func_default(prompt, "mistralai/Mistral-Nemo-Instruct-2407")


def func_llama_3p1_swallow_8b_instruct_v0p1(prompt: str) -> str:
    return _func_default(prompt, "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1")


def func_llama_3_swallow_8b_instruct_v0p1(prompt: str) -> str:
    return _func_default(prompt, "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1")


def func_llama_3_elyza_jp_8b(prompt: str) -> str:
    return _func_default(prompt, "elyza/Llama-3-ELYZA-JP-8B")


def func_llama_3p1_swallow_8b_instruct_v0p3(prompt: str) -> str:
    return _func_default(prompt, "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3")


def func_llama_3_swallow_70b_instruct_v0p1(prompt: str) -> str:
    return _func_default(prompt, "tokyotech-llm/Llama-3-Swallow-70B-Instruct-v0.1")


def func_swallow_chatbotarena_dpo(prompt: str) -> str:
    return _func_swallow_dpo(prompt, "ddyuudd/Swallow-chatbotarena-dpo")


# tokyotech-llm/Swallow-7b-hf
def func_swallow_7b_hf(prompt: str) -> str:
    return _func_swallow_hf(prompt, "tokyotech-llm/Swallow-7b-hf")


# tokyotech-llm/Swallow-7b-NVE-hf
def func_swallow_7b_NVE_hf(prompt: str) -> str:
    return _func_swallow_hf(prompt, "tokyotech-llm/Swallow-7b-NVE-hf")


# llm-jp/llm-jp-13b-instruct-full-jaster-v1.0
def func_llm_jp_13b_instruct_full_jaster_v1p0(prompt: str) -> str:
    return _func_llmjp(prompt, "llm-jp/llm-jp-13b-instruct-full-jaster-v1.0")


# ddyuudd/llmjp-chatbotarena-dpo
def func_llmjp_chatbotarena_dpo(prompt: str) -> str:
    return _func_dpo_llm_jp(prompt, "ddyuudd/llmjp-chatbotarena-dpo")


# cyberagent/calm2-7b-chat
def func_calm2_7b_chat(prompt: str) -> str:
    return _func_calm2(prompt, "cyberagent/calm2-7b-chat")


# cyberagent/calm2-7b-chat-dpo-experimental
def func_calm2_7b_chat_dpo_experimental(prompt: str) -> str:
    return _func_calm2(prompt, "cyberagent/calm2-7b-chat-dpo-experimental")


def _func_gpt(prompt: str, model: str, api_version: str) -> str:
    gpt = GPT(model=model, api_version=api_version)

    # 3回までリトライ
    for _ in range(3):
        try:
            response = gpt.send_user_prompt(prompt)
            if type(response) is not str:
                raise Exception("Response is None")
        except Exception as e:
            print(e)
            pass
        else:
            break
    else:
        response = ""
        print("Failed to generate response 3 times!")

    return response


def _func_default(prompt: str, model_name: str) -> str:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # 3回までリトライ
    for _ in range(3):
        try:
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
                "cuda"
            )

            output_ids = model.generate(
                input_ids,
                max_new_tokens=TOKEN_NUM,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )[0]
            generated_ids = output_ids[len(input_ids[0]) :]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if type(response) is not str:
                raise Exception("Response is None")
        except Exception as e:
            print(e)
            pass
        else:
            break
    else:
        response = "Missing response"
        print("Failed to generate response 3 times!")
    print(response)
    return response


def _func_gemma(prompt: str, model_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # 10回までリトライ
    for _ in range(10):
        try:
            input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=TOKEN_NUM,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )[0]
            generated_ids = output_ids[len(input_ids[0]) :]
            response = tokenizer.decode(generated_ids)
            if type(response) is not str:
                raise Exception("Response is None")
            if response == "":
                raise Exception("Response is empty")
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_lineno)
        else:
            break
    else:
        response = "Missing response"
        print("Failed to generate response 10 times!")
    print(response)
    return response


def _func_qwen(prompt: str, model_name: str) -> str:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # 3回までリトライ
    for _ in range(3):
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=TOKEN_NUM,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )[0]
            generated_ids = output_ids[len(model_inputs[0]) :]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if type(response) is not str:
                raise Exception("Response is None")
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_lineno)
            pass
        else:
            break
    return response


def _func_calm2(prompt: str, model_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    # 10回までリトライ
    for _ in range(10):
        try:
            msg = f"""{SYSTEM_PROMPT}\nUSER: {prompt}"""
            input_ids = tokenizer(msg, return_tensors="pt").to("cuda")
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=TOKEN_NUM,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )[0]
            generated_ids = output_ids[len(input_ids[0]) :]
            response = tokenizer.decode(generated_ids)
            if type(response) is not str:
                raise Exception("Response is None")
            if response == "":
                raise Exception("Response is empty")
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_lineno)
        else:
            break
    else:
        response = "Missing response"
        print("Failed to generate response 10 times!")
    print(response)
    return response


def _func_llmjp(prompt: str, model_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 10回までリトライ
    for _ in range(10):
        try:
            msg = f"""{SYSTEM_PROMPT}\n{prompt}### 回答："""
            input_ids = tokenizer(
                msg, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **input_ids,
                    max_new_tokens=TOKEN_NUM,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                )[0]
            generated_ids = output_ids[len(input_ids[0]) :]
            response = tokenizer.decode(generated_ids)
            if type(response) is not str:
                raise Exception("Response is None")
            if response == "":
                raise Exception("Response is empty")
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_lineno)
        else:
            break
    else:
        response = "Missing response"
        print("Failed to generate response 10 times!")
    print(response)
    return response


def _func_dpo_llm_jp(prompt: str, model_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 10回までリトライ
    for _ in range(10):
        try:
            msg = f"""{SYSTEM_PROMPT}\n{prompt}### 回答："""
            input_ids = tokenizer(
                msg, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **input_ids,
                    max_new_tokens=TOKEN_NUM,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                )[0]
            generated_ids = output_ids[len(input_ids[0]) :]
            response = tokenizer.decode(generated_ids)
            if type(response) is not str:
                raise Exception("Response is None")
            if response == "":
                raise Exception("Response is empty")
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_lineno)
        else:
            break
    else:
        response = "Missing response"
        print("Failed to generate response 10 times!")
    print(response)
    return response


def _func_swallow_hf(prompt: str, model_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 10回までリトライ
    for _ in range(10):
        try:
            msg = (SYSTEM_PROMPT, f"""### 指示：\n{prompt}\n\n### 応答：""")
            input_ids = tokenizer(
                msg, add_special_tokens=False, return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **input_ids,
                    max_new_tokens=TOKEN_NUM,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                )[0]
            generated_ids = output_ids[len(input_ids[0]) :]
            response = tokenizer.decode(generated_ids)
            if type(response) is not str:
                raise Exception("Response is None")
            if response == "":
                raise Exception("Response is empty")
        except Exception as e:
            print(e)
            print(e.__traceback__.tb_lineno)
        else:
            break
    else:
        response = "Missing response"
        print("Failed to generate response 10 times!")
    print(response)
    return response


def _func_swallow_dpo(prompt: str, model_name: str) -> str:
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # 3回までリトライ
    for _ in range(3):
        try:
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
                model.device
            )

            output_ids = model.generate(
                input_ids,
                max_new_tokens=TOKEN_NUM,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )[0]
            generated_ids = output_ids[len(input_ids[0]) :]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if type(response) is not str:
                raise Exception("Response is None")
        except Exception as e:
            print(e)
            pass
        else:
            break
    else:
        response = "Missing response"
        print("Failed to generate response 3 times!")
    print(response)
    return response
