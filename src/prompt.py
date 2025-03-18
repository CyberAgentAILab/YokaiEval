from typing import List
import json
from dataclasses import dataclass

from src.get_yokai_info import Yokai


@dataclass(frozen=True)
class GenerateQuestionPrompt:
    yokai: Yokai
    prompt: str

    @classmethod
    def from_yokai(
        cls, yokai: Yokai, prompt_template: str = "prompts/prompt_template.txt"
    ) -> "GenerateQuestionPrompt":
        prompt = cls._generate_prompt(yokai, prompt_template)
        return cls(yokai, prompt)

    @staticmethod
    def _generate_prompt(yokai: Yokai, prompt_template: str) -> str:
        with open(prompt_template, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        prompt = prompt_template.format(name=yokai.name, detail=yokai.detail)

        return prompt


def _load_yokai_list(json_file_path: str) -> List[Yokai]:
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data is None:
        print(f"Can't load {json_file_path}!")

    yokai_list: List[Yokai] = [
        Yokai(name=yokai["name"], detail=yokai["detail"]) for yokai in data
    ]

    return yokai_list


def generate_prompts(json_file_path: str, num: int) -> List[GenerateQuestionPrompt]:
    """Generate a list of prompts for generate verify questions

    Args:
        json_file_path (str): Path to json file containing yokai list
        num (int): Number of prompts to create

    Returns:
        List[GenerateQuestionPrompt]: List of GenerateQuestionPrompt objects
    """
    all_yokai_list = _load_yokai_list(json_file_path)

    if num > len(all_yokai_list):
        print(f"Number of yokai is less than {num}!")
        print(f"Number of yokai is set to {len(all_yokai_list)}.")
        num = len(all_yokai_list)

    yokai_list = all_yokai_list[:num]
    prompts = [GenerateQuestionPrompt.from_yokai(yokai) for yokai in yokai_list]

    return prompts


if __name__ == "__main__":
    print(generate_prompts("data/yokai_list.json", 2))
