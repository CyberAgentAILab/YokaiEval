import pandas as pd
from dataclasses import asdict
from typing import List
from src.models.common import load_verify_questions, JFolktalesDataset
import json


def generate_datasets(
    verify_question_path: str,
) -> List[JFolktalesDataset]:
    datasets: List[JFolktalesDataset] = []
    questions = load_verify_questions(verify_question_path)

    for q in questions:
        question = q.format_question()
        datasets.append(JFolktalesDataset(q.yokai.name, question, q.answer))
    return datasets


def export_datasets(datasets: List[JFolktalesDataset], save_path: str) -> None:
    df = pd.DataFrame([asdict(dataset) for dataset in datasets])
    df.to_csv(save_path, index=False)


def export_datasets_json(datasets: List[JFolktalesDataset], save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(dataset) for dataset in datasets], f, ensure_ascii=False, indent=4
        )
    print(f"Saved datasets json {save_path}!")
