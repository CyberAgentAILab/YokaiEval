import argparse
import os

from src.llm_functions import (
    func_gpt4o,
    func_gpt4o_mini,
    func_gemma_2_2b_it,
    func_gemma_2_9b_it,
    func_gemma_2_27b_it,
    func_llm_jp_3_1p8b_instruct,
    func_llm_jp_3_3p7b_instruct,
    func_llm_jp_3_13b_instruct,
    func_calm3_22b_chat,
    func_qwen2p5_7b_instruct,
    func_qwen2p5_14b_instruct,
    func_qwen2p5_32b_instruct,
    func_eurollm_1p7b_instruct,
    func_swallow_7b_instruct_v0p1,
    func_eurollm_9b_instruct,
    func_swallow_13b_instruct_v0p1,
    func_llama_3p1_70b_japanese_instruct_2407,
    func_llama_3p1_swallow_70b_instruct_v0p1,
    func_llama_3p1_swallow_8b_instruct_v0p2,
    func_meta_llama_3p1_8b_instruct,
    func_qwen2_7b_instruction,
    func_tanuki_8x8b_dpo_v1p0,
    func_qwen2p5_3b_instruct,
    func_llm_jp_3_172b_instruct3,
    func_tanuki_8b_dpo_v1p0,
    func_llama_3p2_3b_instruct,
    func_meta_llama_3_8b_instruct,
    func_meta_llama_3p3_70b_instruct,
    func_mistral_7b_instruct_v0p2,
    func_mistral_nemo_instruct_2407,
    func_llama_3p1_swallow_8b_instruct_v0p1,
    func_llama_3_swallow_8b_instruct_v0p1,
    func_llama_3_elyza_jp_8b,
    func_llama_3p1_swallow_8b_instruct_v0p3,
    func_llama_3_swallow_70b_instruct_v0p1,
    func_calm2_7b_chat,
    func_calm2_7b_chat_dpo_experimental,
    func_llm_jp_13b_instruct_full_jaster_v1p0,
    func_llmjp_chatbotarena_dpo,
    func_swallow_7b_hf,
    func_swallow_7b_NVE_hf,
    func_swallow_chatbotarena_dpo,
)

from src.eval import evaluation
from src.question import generate_verify_questions
from src.get_yokai_info import get_yokai_list
from src.visualize_result import (
    totaling_llm_answers,
    load_answers_dir,
    export_df_to_csv,
    calc_accuracy,
)
from src.judge import re_judge_responses_gpt, re_judge_responses_algo
from src.generate_datasets import (
    generate_datasets,
    export_datasets,
    export_datasets_json,
)
from src.check_verify_question import (
    is_verify,
    filter_verify_questions,
    filter_llm_responses_by_checked_questions,
)
from src.gpt import GPT


from src.models.common import save_verify_questions, save_yokai_list


parser = argparse.ArgumentParser(
    description="Generate verification questions or Get yokai list or Evaluate the LLM with generated questions or Visualize the results"
)

subparsers = parser.add_subparsers(
    dest="task", required=True, help="generate, scrape, or eval"
)

# Generate verification questions
generate_parser = subparsers.add_parser(
    "generate", help="Generate verification questions"
)
generate_parser.add_argument(
    "--num", type=int, required=True, help="Number of questions to generate"
)
generate_parser.add_argument(
    "--yokai_list", type=str, required=True, help="Path to yokai list json file"
)
generate_parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Path to save verification questions json file",
)

# Get Yokai List from local pages
scrape_parser = subparsers.add_parser(
    "scrap", help="Get Yokai List from local pages"
)
scrape_parser.add_argument(
    "--num", type=int, required=True, help="Number of yokai to scrape"
)
scrape_parser.add_argument(
    "--output", type=str, required=True, help="Path to save yokai list json file"
)

# Evaluate LLMs
eval_parser = subparsers.add_parser(
    "eval", help="Evaluate the LLM with generated questions"
)
eval_parser.add_argument(
    "--verify_questions",
    type=str,
    required=True,
    help="Path to verification questions json file",
)
eval_parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to output directory"
)

# Visualize the results
visualize_parser = subparsers.add_parser(
    "visualize", help="Visualize the results of the LLMs"
)
visualize_parser.add_argument(
    "--answers_dir",
    type=str,
    required=True,
    help="Path to directory containing answers",
)
visualize_parser.add_argument(
    "--output", type=str, required=True, help="Path to output csv file"
)

# Rejudge the results
rejudge_parser = subparsers.add_parser(
    "rejudge", help="Rejudge the results of the LLMs"
)
rejudge_parser.add_argument(
    "--mode", type=str, required=True, help="Mode of rejudge, algo or gpt"
)
rejudge_parser.add_argument(
    "--answers_dir",
    type=str,
    required=True,
    help="Path to directory containing answers",
)
rejudge_parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to output directory"
)

# Generate datasets
generate_datasets_parser = subparsers.add_parser(
    "generate_datasets", help="Generate datasets for training"
)
generate_datasets_parser.add_argument(
    "--verify_questions",
    type=str,
    required=True,
    help="Path to verification questions json file",
)
generate_datasets_parser.add_argument(
    "--output", type=str, required=True, help="Path to output directory"
)

# Check is verify
check_verify_parser = subparsers.add_parser("check_verify", help="Check is verify")
check_verify_parser.add_argument(
    "--verify_questions",
    type=str,
    required=True,
    help="Path to verification questions json file",
)

# filter verify questions
filter_verify_parser = subparsers.add_parser(
    "filter_verify", help="Filter verify questions"
)
filter_verify_parser.add_argument(
    "--verify_questions",
    type=str,
    required=True,
    help="Path to verification questions json file",
)
filter_verify_parser.add_argument(
    "--output", type=str, required=True, help="Path to output json file"
)

# filter responses by checked questions
filter_responses_parser = subparsers.add_parser(
    "filter_responses", help="Filter responses by checked questions"
)
filter_responses_parser.add_argument(
    "--llm_response_dir",
    type=str,
    required=True,
    help="Path to directory containing responses",
)
filter_responses_parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to output directory"
)


args = parser.parse_args()

# Generate verification questions
if args.task == "generate":
    num = args.num
    yokai_list = args.yokai_list
    output_dir = args.output

    gpt = GPT()

    responses = generate_verify_questions(gpt, yokai_list, num)
    save_verify_questions(responses, output_dir)

    print(f"Total cost: {gpt.total_cost:.4f} $")

# Generate datasets (verification questionsをcsvに変換する)
if args.task == "generate_datasets":
    verify_question = args.verify_questions
    output = args.output

    datasets = generate_datasets(verify_question)
    export_datasets(datasets, output)
    export_datasets_json(datasets, output.replace(".csv", ".json"))
    print(f"Datasets saved to {output}")
    print(f"Datasets saved to {output.replace('.csv', '.json')}")


# Get Yokai List
if args.task == "scrap":
    num = args.num
    output_dir = args.output

    yokai_list = get_yokai_list(num)
    save_yokai_list(yokai_list, output_dir)


# Evaluate LLMs
if args.task == "eval":
    verify_question = args.verify_questions
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ============================
    # 評価したいモデルをここに記述
    # ============================

    # eg. API
    # evaluation(func_gpt4o, os.path.join(output_dir, "gpt4o.json"), verify_question)

    # eg. OPEN LLM
    # evaluation(func_calm3_22b_chat, os.path.join(output_dir, "calm3-chat.json"), verify_question)
    evaluation(func_llm_jp_3_1p8b_instruct, os.path.join(output_dir, "llm-jp-test.json"), verify_question)

    print(f"Results saved to {output_dir}")

# Visualize the results
if args.task == "visualize":
    answers_dir = args.answers_dir
    output_path = args.output

    params = load_answers_dir(answers_dir)
    df = totaling_llm_answers(params)
    export_df_to_csv(df, f"{output_path}-result.csv")
    df = calc_accuracy(df)
    export_df_to_csv(df, f"{output_path}-acurency.csv")
    print(f"Results saved to {output_path}-result.csv")
    print(f"Results saved to {output_path}-acurency.csv")

# Rejudge the results
if args.task == "rejudge":
    if args.mode == "algo":
        answers_dir = args.answers_dir
        output_dir = args.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        re_judge_responses_algo(answers_dir, output_dir)
    elif args.mode == "gpt":
        answers_dir = args.answers_dir
        output_dir = args.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        re_judge_responses_gpt(answers_dir, output_dir)
    else:
        print("Error: Invalid mode!")

# Check is verify
if args.task == "check_verify":
    verify_question = args.verify_questions
    is_verify(verify_question)

# filter verify questions
if args.task == "filter_verify":
    verify_question = args.verify_questions
    output = args.output
    filter_verify_questions(verify_question, output)
    print(f"Results saved to {output}")

# filter responses by checked questions
if args.task == "filter_responses":
    llm_response_dir = args.llm_response_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    filter_llm_responses_by_checked_questions(llm_response_dir, output_dir)
    print(f"Results saved to {output_dir}")
