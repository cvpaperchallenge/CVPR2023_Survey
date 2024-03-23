"""

This script generate json file which includes all CVPR 2023 papers info.

"""
import argparse
import json
import pathlib
from typing import Final

from src.parser import ConferencePath, get_paper_page_urls, parse_paper_page


def extract_paper_info(output_dir: pathlib.Path, conference: str) -> None:
    """
    Extract paper information from CVF page and save it as JSON file.

    Args:
        output_dir (str): Output directory to save the JSON file.
        conference (str): Conference name where papers information is extracted.

    """
    if conference == "cvpr":
        conference_path = ConferencePath.CVPR
    elif conference == "iccv":
        conference_path = ConferencePath.ICCV
    urls: Final = get_paper_page_urls(conference_path)

    papers = list()
    for i, url in enumerate(urls):
        print(f"Processing {i+1}/{len(urls)}: {url}")
        paper = parse_paper_page(url)
        papers.append(paper.dict())

    output_path: Final = output_dir / f"{conference}_papers.json"
    with output_path.open("w") as f:
        json.dump(papers, f)

    print(f"Successfully parsed {len(papers)} papers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        default="./data",
        help="Output directory to save the JSON file.",
    )
    parser.add_argument(
        "--conference",
        "-c",
        choices=["cvpr", "iccv"],
        type=str,
        required=True,
        help="Conference name where papers information is extracted.",
    )
    args = parser.parse_args()

    extract_paper_info(output_dir=args.output_dir, conference=args.conference)
