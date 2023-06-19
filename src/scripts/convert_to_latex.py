"""

This script convert PDF into Latex format text. This script uses Mathpix
API and it requires followings as environmental variables.

- MATHPIX_API_ID
- MATHPIX_API_KEY

"""
import json
import logging
import pathlib
from typing import Final, cast

from src.loader import CustomMathpixLoader
from src.parser import Paper

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Check JSON file existence.
paper_info_path: Final = pathlib.Path("./data/papers.json")
if not paper_info_path.exists():
    error_message: Final = f"This scripts requires `{str(paper_info_path)}`. \
        Please run `parse_cvf_page.py` frist to generate JSON file."
    raise FileNotFoundError(error_message)

# Load JSON and validate by Pydantic model.
with paper_info_path.open("r") as f:
    papers: Final = [Paper.parse_obj(p) for p in json.load(f)]

# Loop over all papers.
paper_root_path: Final = pathlib.Path("./data/papers/")
for i, paper in enumerate(papers):
    # stem is like: <family_name>_<paper_title>_CVPR_2023_paper
    stem = str(pathlib.Path(paper.pdf).stem)
    directory_path = paper_root_path / stem.removesuffix("_CVPR_2023_paper")

    # Check if PDF file exists or not.
    pdf_file_path = directory_path / (stem + ".pdf")
    if not pdf_file_path.exists():
        raise FileNotFoundError(
            f"`{str(pdf_file_path)}` does not exist. Please run `download_papers.py` frist to download PDF file."
        )

    # If mathpix file already exists, continue loop.
    mathpix_file_path = directory_path / (stem + "_mathpix.txt")
    if mathpix_file_path.exists():
        logger.info(f"`{str(mathpix_file_path)}` already exists.")
        continue

    # Send request to Mathpix.
    logger.info(f"[{i+1}/{len(papers)}] `{paper.title}` is sent to Mathpix API.")
    latex_text = CustomMathpixLoader(
        file_path=str(pdf_file_path),
        output_path_for_tex=directory_path,
        processed_file_format=["mmd", "tex.zip"],
        other_request_parameters={
            "math_inline_delimiters": ["$", "$"],
            "math_display_delimiters": ["$$", "$$"],
        },
        output_langchain_document=False,
    ).load()["mmd"]

    # Save latex format text.
    with mathpix_file_path.open("w") as f:
        f.write(cast(str, latex_text))
