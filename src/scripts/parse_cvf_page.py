"""

This script generate json file which includes all CVPR 2023 papers info.

"""
import json
import pathlib
from typing import Final

from src.parser import get_paper_page_urls, parse_paper_page

urls: Final = get_paper_page_urls()

papers = list()
for i, url in enumerate(urls):
    paper = parse_paper_page(url)
    papers.append(paper.dict())

output_path: Final = pathlib.Path("./data/papers.json")
with output_path.open("w") as f:
    json.dump(papers, f)
