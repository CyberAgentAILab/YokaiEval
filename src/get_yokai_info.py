import requests
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import urljoin
import time
from dataclasses import dataclass

from src.models.common import Yokai


@dataclass(frozen=True)
class WikipediaTd:
    label: str
    url: str


def _get_wikipedia_table() -> List[WikipediaTd]:
    """Get Wikipedia table of yokai

    Returns:
        List[WikipediaTd]: List of yokai name and URL
    """

    # 事前にダウンロードしたファイルを用いる
    base_url = "data/wikipedia/"
    table_file_path = "data/wikipedia/日本の妖怪一覧.html"
    with open(table_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    tables = soup.find_all("table", attrs={"class": "wikitable"})

    wikipedia_table: List[WikipediaTd] = []
    for table in tables:
        for _, tr_element in enumerate(table.find_all("tr")):
            if tr_element.find_all("td") and tr_element.find_all("td")[0].find("a"):
                # 妖怪名を取得
                labl = tr_element.find_all("td")[0].text

                # 妖怪のページのURLを取得
                relative_url = tr_element.find_all("td")[0].find("a")["href"]

                url = base_url + relative_url

                if "redlink" not in url:  # 未作成のページは除外
                    wikipedia_table.append(WikipediaTd(label=labl, url=url))
                else:
                    print(f"Skip {labl} because the page is not created yet")

    return wikipedia_table


def get_yokai_list(num=20) -> List[Yokai]:
    """Get yokai detail information

    Args:
        count (int, optional): Number of yokai to get. Defaults to 20.

    Returns:
        List[Yokai]: List of yokai name and detail information
    """
    wikipedia_table = _get_wikipedia_table()

    yokai_list: List[Yokai] = []
    for wikipedia_td in wikipedia_table:

        # 事前にダウンロードしたファイルを用いる
        with open(wikipedia_td.url, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        yokai_name = wikipedia_td.label

        yokai_detail = " ".join([tag.text for tag in soup.find_all("p")])

        yokai_list.append(Yokai(name=yokai_name, detail=yokai_detail))

        if len(yokai_list) >= num:
            break
        print(yokai_list[-1])
        time.sleep(1)

    return yokai_list
