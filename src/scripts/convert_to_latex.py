"""

This script convert PDF into Latex format text. This script uses Mathpix
API and it requires the following environment variables.

- MATHPIX_API_ID
- MATHPIX_API_KEY

"""
import argparse
import logging
import pathlib
from typing import Final, cast

from src.loader import CustomMathpixPDFLoader

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def convert_pdf_to_latex(paper_root_dir: pathlib.Path) -> None:
    """Convert PDF files into Latex format text.

    Args:
        paper_root_dir (pathlib.Path): Path to the directory containing PDF files.
    """
    # Loop over all papers.
    pdf_file_paths = sorted(list(paper_root_dir.glob("**/*.pdf")))
    for i, pdf_file_path in enumerate(pdf_file_paths):
        directory_path = pdf_file_path.parent
        stem = pdf_file_path.stem

        # Check if PDF file exists or not.
        pdf_file_path = directory_path / (stem + ".pdf")
        if not pdf_file_path.exists():
            raise FileNotFoundError(
                f"`{str(pdf_file_path)}` does not exist. Please run `download_papers.py` first to download PDF file."
            )

        # If mathpix file already exists, skip the conversion.
        mathpix_file_path = directory_path / (stem + "_mathpix.txt")
        if mathpix_file_path.exists():
            logger.info(
                f"Skip converting `{str(pdf_file_path)}` as the Mathpix file already exists."
            )
            continue

        # Send request to Mathpix.
        logger.info(f"[{i+1}/{len(pdf_file_paths)}] `{stem}` is sent to Mathpix API.")
        latex_text = (
            CustomMathpixPDFLoader(
                file_path=str(pdf_file_path),
                processed_file_format="md",
                extra_request_data={
                    "math_inline_delimiters": ["$", "$"],
                    "math_display_delimiters": ["$$", "$$"],
                },
            )
            .load_mmd()[0]
            .page_content
        )

        # Save latex format text.
        with mathpix_file_path.open("w") as f:
            f.write(latex_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-pdf-dir",
        "-i",
        type=pathlib.Path,
        required=True,
        help="Path to the directory containing PDF files.",
    )

    args = parser.parse_args()
    convert_pdf_to_latex(args.input_dir)
