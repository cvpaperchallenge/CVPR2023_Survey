"""

This script generate json file which includes all papers information of the selected conference.
The code estimates Computer Vision Foundation(CVF) supported conferences such as CVPR and ICCV.

"""
import argparse
import json
import logging
import pathlib
from typing import Any, Final

from pydantic_core import Url

from src.cvf_page_parser import get_paper_page_urls, parse_paper_page

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def url_serializer_for_json_dump(object: Any) -> str:
    if isinstance(object, Url):
        return str(object)
    raise TypeError(
        f"Object of type {object.__class__.__name__} is not JSON serializable"
    )


def extract_paper_info(output_dir: pathlib.Path, conference: str, year: int) -> None:
    """
    Extract paper information from CVF page and save it as JSON file.

    Args:
        output_dir (str): Output directory to save the JSON file.
        conference (str): The conference name.
        year (int): The year of the conference.

    """
    urls: Final = get_paper_page_urls(conference=conference, year=year)

    papers = list()
    for i, url in enumerate(urls):
        logger.info(f"Processing {i+1}/{len(urls)}: {url}")
        paper = parse_paper_page(url)
        papers.append(paper.model_dump())

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path: Final = output_dir / f"{conference}{year}_papers.json"
    with output_path.open("w") as f:
        json.dump(papers, f, indent=4, default=url_serializer_for_json_dump)

    logger.info(f"Successfully parsed {len(papers)} papers.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        default="./data/json",
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
    parser.add_argument(
        "--year",
        "-y",
        type=int,
        required=True,
        help="The year of the conference.",
    )
    args = parser.parse_args()

    extract_paper_info(
        output_dir=args.output_dir, conference=args.conference, year=args.year
    )
