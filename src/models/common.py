from typing import List
import json
import sys

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Yokai:
    name: str
    detail: str


def save_yokai_list(yokai_list: List[Yokai], json_file_path: str):
    """Save yokai list to json file

    Args:
        yokai_list (List[Yokai]): List of Yokai ojects
        json_file_path (str): Path to save json file
    """
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(yokai) for yokai in yokai_list], f, ensure_ascii=False, indent=4
        )
    print(f"Saved Yokai json {json_file_path}!")


@dataclass(frozen=True)
class VerifyQuestion:
    yokai: Yokai
    question: str
    choices: List[str]
    answer: str

    @classmethod
    def from_llm_response(self, response: str, yokai: Yokai):
        try:
            response_json = json.loads(response)
            question = response_json["question"]
            choices = response_json["choices"]
            answer = response_json["answer"]
            return VerifyQuestion(yokai, question, choices, answer)

        except json.JSONDecodeError:
            print(response)
            print("Can't parse json GPT response", file=sys.stderr)
            return None
        except KeyError:
            print(response)
            print(
                "Can't find question, choices, or answer in GPT response",
                file=sys.stderr,
            )
            return None

    def format_question(self) -> str:
        return f"""{self.question}
- {self.choices[0]}
- {self.choices[1]}
- {self.choices[2]}
- {self.choices[3]}"""


def save_verify_questions(questions: List[VerifyQuestion], export_json_path: str):
    """Export questions to a json file

    Args:
        questions (List[VerifyQuestion]): List of VerifyQuestion objects
    """
    with open(export_json_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(question) for question in questions],
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(f"Exported {len(questions)} questions to verify_questions.json")


def load_verify_questions(json_file_path: str) -> List[VerifyQuestion]:
    """Load verification questions from a json file

    Args:
        json_file_path (str): Path to json file

    Returns:
        List[VerifyQuestion]: List of VerifyQuestion objects
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data is None:
        print(f"Can't load {json_file_path}!")

    yokai_list: List[VerifyQuestion] = [
        VerifyQuestion(
            yokai=Yokai(**verify_question.get("yokai", {"name": "", "detail": ""})),
            question=verify_question["question"],
            choices=verify_question["choices"],
            answer=verify_question["answer"],
        )
        for verify_question in data
    ]

    return yokai_list


@dataclass(frozen=True)
class LLMAnswer:
    question: VerifyQuestion
    question_prompt: str
    response: str
    judge: bool


def save_answers(answers: List[LLMAnswer], json_file_path: str) -> None:
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(answer) for answer in answers], f, ensure_ascii=False, indent=4
        )
    print(f"Saved answers json {json_file_path}!")


def load_answers(json_file_path: str) -> List[LLMAnswer]:
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data is None:
        print(f"Can't load {json_file_path}!")
    return [
        LLMAnswer(
            question=VerifyQuestion(
                yokai=Yokai(**answer["question"]["yokai"]),
                question=answer["question"]["question"],
                choices=answer["question"]["choices"],
                answer=answer["question"]["answer"],
            ),
            question_prompt=answer["question_prompt"],
            response=answer["response"],
            judge=answer["judge"],
        )
        for answer in data
    ]


@dataclass(frozen=True)
class JFolktalesDataset:
    yokai_name: str
    prompt: str
    answer: str
