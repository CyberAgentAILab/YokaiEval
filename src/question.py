from typing import List

from src.prompt import generate_prompts
from src.gpt import GPT
from src.models.common import VerifyQuestion


def generate_verify_questions(
    gpt: GPT, yokai_list_json_path: str, num: int
) -> List[VerifyQuestion]:
    """Generate a number of verification questions

    Args:
        gpt (GPT): GPT object
        yokai_list_json_path (str): Path to yokai list json file
        num (int): Number of questions to generate

    Returns:
        List[str]: List of responses from GPT
    """
    generate_question_prompts = generate_prompts(yokai_list_json_path, num)

    verify_questions: List[VerifyQuestion] = []
    for generate_question_prompt in generate_question_prompts:
        gpt_response = gpt.send_user_prompt(generate_question_prompt.prompt)

        verify_question = VerifyQuestion.from_llm_response(
            gpt_response, generate_question_prompt.yokai
        )
        if verify_question is not None:
            verify_questions.append(verify_question)

    return verify_questions
