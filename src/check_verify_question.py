import os
from typing import List
from src.models.common import (
    VerifyQuestion,
    load_verify_questions,
    LLMAnswer,
    save_answers,
    save_verify_questions,
    load_answers,
)
from src.gpt import GPT


def is_verify(verify_question_json_path: str):
    gpt = GPT()
    questions = load_verify_questions(verify_question_json_path)
    answers: List[LLMAnswer] = []
    for q in questions:
        yokai_name = q.yokai.name
        yokai_detail = q.yokai.detail
        question = q.format_question()
        prompt = f"""
###Task###
これは{yokai_name}の説明です。説明を参考にして、質問に答えてください。
###説明###
{yokai_detail}。
###質問###
{question}
        """
        for _ in range(3):
            response = gpt.send_user_prompt(prompt)
            if response is not None:
                break
            else:
                print("Retry")
        if response is not None:
            judge = q.answer in response
            answers.append(
                LLMAnswer(
                    question=q, question_prompt=prompt, response=response, judge=judge
                )
            )
            print(judge, response, q.answer)
        else:
            print("Failed to get response")
            print(yokai_name)
    save_answers(answers, "data/gpt4o-mini-check-question-is-verify.json")


def filter_verify_questions(
    verify_question_json_path: str,
    output_json_path: str,
    check_json_path: str = "data/gpt4o-mini-check-question-is-verify.json",
):
    checked_answers = load_answers(check_json_path)
    questions = load_verify_questions(verify_question_json_path)

    checked_questions: List[VerifyQuestion] = []
    for q in questions:
        checked_answer_question = [a for a in checked_answers if a.question == q]
        if len(checked_answer_question) == 0:
            continue
        else:
            checked_answer_question = checked_answer_question[0]
        if checked_answer_question.judge:
            checked_questions.append(q)

    save_verify_questions(checked_questions, output_json_path)


def filter_llm_responses_by_checked_questions(
    llm_response_dir: str,
    output_dir: str,
    checked_questions_json_path: str = "data/checked_verify_questions.json",
):
    checked_questions = load_verify_questions(checked_questions_json_path)
    llm_response_json_paths = [
        os.path.join(llm_response_dir, f)
        for f in os.listdir(llm_response_dir)
        if f.endswith(".json")
    ]
    for llm_response_json_path in llm_response_json_paths:
        responses = load_answers(llm_response_json_path)

        filtered_responses: List[LLMAnswer] = []
        for r in responses:
            if r.question.yokai.name in [q.yokai.name for q in checked_questions]:
                filtered_responses.append(r)

        output_json_path = os.path.join(
            output_dir, os.path.basename(llm_response_json_path)
        )
        save_answers(filtered_responses, output_json_path)
