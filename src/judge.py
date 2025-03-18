import pylcs
from rapidfuzz.distance.Levenshtein import distance
from typing import List

from src.question import VerifyQuestion
from src.visualize_result import load_answers_dir
from src.models.common import save_answers, LLMAnswer
from src.gpt import GPT

CORRECT_LCS_THRESHOLD = 0.8


def re_judge_responses_gpt(answers_dir: str, output_dir: str) -> None:
    """Rejudge the answers of the LLMs

    Args:
        answers_dir (str): Path to directory containing answers json files

    Returns:
        List[LLMAnswer]: List of LLMAnswer objects
    """
    all_model_answers = load_answers_dir(answers_dir)

    gpt = GPT(model="gpt-4o", api_version="2024-08-06")

    for model_answers in all_model_answers:
        answers: List[LLMAnswer] = []
        print(f"Rejudging {model_answers.model_name}...")
        for answer in model_answers.llm_answer:
            verify_question = answer.question
            response = answer.response
            question = answer.question_prompt
            correct = answer.question.answer

            judge = gpt_judge(gpt, question, response, correct)

            answers.append(
                LLMAnswer(
                    question=verify_question,
                    question_prompt=question,
                    response=response,
                    judge=judge,
                )
            )

        save_answers(answers, f"{output_dir}/{model_answers.model_name}.json")
        print(f"Rejudged {model_answers.model_name}!")


def gpt_judge(gpt: GPT, question: str, response: str, correct: str) -> bool:
    with open("prompts/judge_prompt.txt", "r", encoding="utf-8") as f:
        judge_prompt_template = f.read()

    prompt = judge_prompt_template.format(
        question=question, response=response, correct=correct
    )

    judge = gpt.send_user_prompt(prompt)
    print(judge, response, correct)

    # もっと柔軟性を持たせても良いが、gpt4は優秀なので、これで十分だと思う
    if judge in ["true", "True", "TRUE"]:
        return True
    elif judge in ["false", "False", "FALSE"]:
        return False
    elif judge in ["null", "Null", "None", "none"]:
        return None
    else:
        print("Error: Can't judge!")
        return None


def re_judge_responses_algo(answers_dir: str, output_dir: str) -> None:
    """Rejudge the answers of the LLMs

    Args:
        answers_dir (str): Path to directory containing answers json files

    Returns:
        List[LLMAnswer]: List of LLMAnswer objects
    """
    all_model_answers = load_answers_dir(answers_dir)

    for model_answers in all_model_answers:
        answers: List[LLMAnswer] = []
        for answer in model_answers.llm_answer:
            verify_question = answer.question
            response = answer.response
            question = answer.question_prompt
            correct = answer.question.answer

            judge = algo_judge(verify_question, response, correct)

            answers.append(
                LLMAnswer(
                    question=verify_question,
                    question_prompt=question,
                    response=response,
                    judge=judge,
                )
            )

        save_answers(answers, f"{output_dir}/{model_answers.model_name}.json")
        print(f"Rejudged {model_answers.model_name}!")


def algo_judge(question: VerifyQuestion, response: str, correct: str) -> bool:
    choices = question.choices
    in_corrects = [choice for choice in choices if choice != correct]
    in_correct1 = in_corrects[0]
    in_correct2 = in_corrects[1]
    in_correct3 = in_corrects[2]

    # 正解を出力に含んでいるか判定
    if pylcs.lcs2(response, correct) / len(correct) > CORRECT_LCS_THRESHOLD:

        # 誤答を出力に含んでいないか判定
        if not (
            in_correct1 in response
            or in_correct2 in response
            or in_correct3 in response
        ):
            # 1つの誤答も含んでいない場合は True
            return True

        # 質問の反復を出力に含んでいる場合を検出
        # 編集距離が問題文の長さの1/2以下の場合は、質問の反復と判定
        elif (
            distance(question.format_question(), response)
            < len(question.format_question()) / 2
        ):
            # 質問の反復部分は取り除いて再度判定

            # `？`を含む質問文の場合、`？`より前を取り除く
            if "？" in response:  # 全角
                cut_response = (
                    "".join(response.split("?")[1:]) if "？" in response else response
                )
            elif "?" in response:  # 半角
                cut_response = (
                    "".join(response.split("?")[1:]) if "?" in response else response
                )
            else:
                # それ以外の場合は、問題文の長さ文字数を取り除く
                cut_response = response[len(question.format_question()) :]

            # 一つ以上解答を含んでいるか判定
            if (
                pylcs.lcs2(cut_response, correct) / len(correct) > CORRECT_LCS_THRESHOLD
                or pylcs.lcs2(cut_response, in_correct1) / len(in_correct1)
                > CORRECT_LCS_THRESHOLD
                or pylcs.lcs2(cut_response, in_correct2) / len(in_correct2)
                > CORRECT_LCS_THRESHOLD
                or pylcs.lcs2(cut_response, in_correct3) / len(in_correct3)
                > CORRECT_LCS_THRESHOLD
            ):
                # 含んでいる場合は、正解を含んでいる and 誤答を含んでいないかで正誤判定
                if (
                    pylcs.lcs2(cut_response, correct) / len(correct)
                    > CORRECT_LCS_THRESHOLD
                    and pylcs.lcs2(cut_response, in_correct1) / len(in_correct1)
                    < CORRECT_LCS_THRESHOLD
                    and pylcs.lcs2(cut_response, in_correct2) / len(in_correct2)
                    < CORRECT_LCS_THRESHOLD
                    and pylcs.lcs2(cut_response, in_correct3) / len(in_correct3)
                    < CORRECT_LCS_THRESHOLD
                ):
                    return True
                else:
                    # 正解を含んでおらず、誤答を含んでいる場合は False
                    return False
            else:  # 一つも含んでいない場合
                # 出力の中で最初の選択肢を解答として判定
                try:
                    # 曖昧な場合を検出するのは大変なので、解答がそのまま出力に含まれている場合のみ対応
                    correct_index = response.index(correct)
                except ValueError:
                    # 正解が含まれていない場合は、False
                    return False
                for in_correct in [in_correct1, in_correct2, in_correct3]:
                    try:
                        in_correct_index = response.index(in_correct)
                    except ValueError:
                        continue
                    if correct_index > in_correct_index:
                        return False
                return True
        else:  # 質問の反復を含んでいない場合
            # 出力の中で最初の選択肢を解答として判定
            try:
                # 曖昧な場合を検出するのは大変なので、解答がそのまま出力に含まれている場合のみ対応
                correct_index = response.index(correct)
            except ValueError:
                # 正解が含まれていない場合は、False
                return False
            for in_correct in [in_correct1, in_correct2, in_correct3]:
                try:
                    in_correct_index = response.index(in_correct)
                except ValueError:
                    continue
                if correct_index > in_correct_index:
                    return False
            return True
    else:
        # どの選択肢も含んでいない場合は None
        if (
            pylcs.lcs2(response, in_correct1) / len(in_correct1) < CORRECT_LCS_THRESHOLD
            and pylcs.lcs2(response, in_correct2) / len(in_correct2)
            < CORRECT_LCS_THRESHOLD
            and pylcs.lcs2(response, in_correct3) / len(in_correct3)
            < CORRECT_LCS_THRESHOLD
        ):
            return None
        else:
            # 正解を含んでいないが、誤答を含んでいる場合は False
            return False
