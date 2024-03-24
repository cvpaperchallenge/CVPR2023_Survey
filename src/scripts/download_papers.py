"""

This script download all CVPR 2023 papers under ./data directory.

"""
import argparse
import json
import logging
import pathlib
from typing import Final

import requests

from src.parser import Paper

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

paper_info_path: Final = pathlib.Path("./data/papers.json")

def download_paper_pdfs(output_root_dir: pathlib.Path, paper_info_path: pathlib.Path) -> None:
    """Download all papers PDF files.

    Args:
        output_root_dir (pathlib.Path): Output root directory to save the PDF files.
        paper_info_path (pathlib.Path): Path to the JSON file which contains paper information

    """
    # Check JSON file existence.
    if not paper_info_path.exists():
        error_message: Final = f"The file `{str(paper_info_path)}` does not exist. \
            Please run `parse_cvf_page.py` first to generate JSON file."
        raise FileNotFoundError(error_message)

    # Load JSON and validate by Pydantic model.
    with paper_info_path.open("r") as f:
        papers: Final = [Paper.model_validate(p) for p in json.load(f)]

    # Loop over all papers and save PDF under ./data/paper directory
    for i, paper in enumerate(papers):
        response = requests.get(str(paper.pdf))

        # filename is like: <family_name>_<paper_title>_<conference_name>_<year>_paper.pdf
        filename = str(paper.pdf).split("/")[-1]
        core, conference_name, year, _ = filename.rsplit("_", 3)
        family_name, paper_title = core.split("_", 1)
        directory_path = output_root_dir / pathlib.Path(conference_name + year) / pathlib.Path(f"{i:04}_{paper_title}")
        file_path = directory_path / filename

        logger.info(f"[{i+1}/{len(papers)}] Downloading paper `{paper.title}`.")

        # If directory already exists, skip it.
        if directory_path.exists():
            continue

        # Create directory to save PDF.
        directory_path.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as f:
            f.write(response.content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root-dir",
        "-o",
        type=pathlib.Path,
        default="./data/papers",
        help="Output root directory to save the PDF files.",
    )
    parser.add_argument(
        "--paper-info",
        "-p",
        type=pathlib.Path,
        required=True,
        help="Path to the JSON file which contains paper information.",
    )
    args = parser.parse_args()

    download_paper_pdfs(output_root_dir=args.output_root_dir, paper_info_path=args.paper_info)