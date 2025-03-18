from typing import List, Callable

from src.models.common import (
    LLMAnswer,
    save_answers,
    VerifyQuestion,
    load_verify_questions,
)


def ask_question_llm(question: VerifyQuestion, func: Callable[[str], str]) -> LLMAnswer:
    question_prompt = question.format_question()

    response = func(question_prompt)  # 関数渡しの関数を実行

    judge = False

    return LLMAnswer(
        question=question,
        question_prompt=question_prompt,
        response=response,
        judge=judge,
    )


def ask_questions_llm(
    questions: List[VerifyQuestion], func: Callable[[str], str]
) -> List[LLMAnswer]:
    return [ask_question_llm(question, func) for question in questions]


def evaluation(func: Callable[[str], str], save_path: str, verify_question_path: str):
    """Run the evaluation

    Args:
        func (Callable[[str], str]): Function wrapper for the query to the LLM
    """
    questions = load_verify_questions(verify_question_path)
    answers = ask_questions_llm(questions, func)
    save_answers(answers, save_path)
