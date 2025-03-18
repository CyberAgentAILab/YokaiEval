import pandas as pd
import os
from typing import List
from dataclasses import dataclass

from src.models.common import LLMAnswer, load_answers


@dataclass(frozen=True)
class TotalingLLMAnswersParam:
    model_name: str
    llm_answer: List[LLMAnswer]


def load_answers_dir(answers_dir: str) -> List[TotalingLLMAnswersParam]:
    """Load all answers from a directory

    Args:
        answers_dir (str): Path to directory containing answers json files

    Returns:
        List[TotalingLLMAnswersParam]: List of TotalingLLMAnswersParam objects
    """
    all_answers_model_name: List[TotalingLLMAnswersParam] = []
    for file in os.listdir(answers_dir):
        if file.endswith(".json"):
            answers = load_answers(os.path.join(answers_dir, file))
            model_name = file.replace(".json", "")
            all_answers_model_name.append(
                TotalingLLMAnswersParam(model_name=model_name, llm_answer=answers)
            )
    return all_answers_model_name


def sort_model_names_by_accuracy(params: List[TotalingLLMAnswersParam]) -> List[str]:
    """Sort model names by accuracy

    Args:
        params (List[TotalingLLMAnswersParam]): List of TotalingLLMAnswersParam objects

    Returns:
        List[str]: List of model names sorted by accuracy
    """
    model_names = [param.model_name for param in params]

    # Calculate accuracy for each model
    accuracy = []
    for param in params:
        correct = 0
        for answer in param.llm_answer:
            if answer.judge:
                correct += 1
        accuracy.append(correct / len(param.llm_answer))

    # Sort model names by accuracy
    model_names_sorted = [
        x for _, x in sorted(zip(accuracy, model_names), reverse=True)
    ]
    return model_names_sorted


def totaling_llm_answers(params: List[TotalingLLMAnswersParam]) -> pd.DataFrame:
    """Totaling the answers of the LLMs

    Args:
        answers (List[TotalingLLMAnswersParam]): List of TotalingLLMAnswersParam objects

    Returns:
        pd.DataFrame: |
            DataFrame containing the total of the answers.
            Columns: yokai, question, correct, gpt4o, gpt4o-mini, gemma-2-2b-it, ...
    """
    sorted_model_names = sort_model_names_by_accuracy(params)

    columns = ["yokai", "question", "correct"] + sorted_model_names

    df = pd.DataFrame(columns=columns)

    for param in params:
        model_name = param.model_name
        for idx, answer in enumerate(param.llm_answer):
            df.loc[idx, "yokai"] = answer.question.yokai.name
            df.loc[idx, "question"] = answer.question_prompt
            df.loc[idx, "correct"] = answer.question.answer
            df.loc[idx, model_name] = answer.judge

    return df


def export_df_to_csv(df: pd.DataFrame, output_path: str):
    """Export DataFrame to csv

    Args:
        df (pd.DataFrame): DataFrame to export
        output_dir (str): Path to output directory
    """
    df.to_csv(output_path)


def calc_accuracy(df: pd.DataFrame):
    """Calculate accuracy of each model

    Args:
        df (pd.DataFrame): DataFrame containing the total of the answers.

    Returns:
        pd.Series: Series containing the accuracy of each model
    """
    model_names = df.columns[3:]
    accuracy = {}
    for model_name in model_names:
        correct = df[model_name].sum()
        accuracy[model_name] = correct / len(df)
        none_count = df[model_name].isnull().sum()
        print(f"{model_name}'s none count: {none_count}")
    print(accuracy)
    return pd.Series(accuracy)
