"""

This script download all CVPR 2023 papers under ./data directory.

"""
import json
import logging
import pathlib
from typing import Final

import requests

from src.parser import Paper

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

paper_info_path: Final = pathlib.Path("./data/papers.json")

# Check JSON file existence.
if not paper_info_path.exists():
    error_message: Final = f"This scripts requires `{str(paper_info_path)}`. \
        Please run `parse_cvf_page.py` frist to generate JSON file."
    raise FileNotFoundError(error_message)

# Load JSON and validate by Pydantic model.
with paper_info_path.open("r") as f:
    papers: Final = [Paper.parse_obj(p) for p in json.load(f)]

# Loop over all papers and save PDF under ./data directory
download_root_path: Final = pathlib.Path("./data/papers/")
for i, paper in enumerate(papers):
    response = requests.get(paper.pdf)

    # filename is like: <family_name>_<paper_title>_CVPR_2023_paper.pdf
    filename = paper.pdf.split("/")[-1]
    directory_path = download_root_path / filename.removesuffix("_CVPR_2023_paper.pdf")
    file_path = directory_path / filename

    logger.info(f"[{i+1}/{len(papers)}] Downloading paper `{paper.title}`.")

    # If directory already exists, skip it.
    if directory_path.exists():
        continue

    # Create directory to save PDF.
    directory_path.mkdir(parents=True)
    with file_path.open("wb") as f:
        f.write(response.content)
