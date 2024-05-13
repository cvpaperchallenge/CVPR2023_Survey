# ruff: noqa: D200, D212
"""

This script copies target files to the output directory.

"""
import argparse
import logging
import pathlib
import shutil
from typing import Final, Literal

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_outputs(
  paper_root_dir: pathlib.Path,
  output_root_dir: pathlib.Path,
  target: Literal["mmd", "summary"]
) -> None:
    """Copy target files to the output directory.

    Args:
        paper_root_dir (pathlib.Path): Path to the root directory containing PDF files.
        output_root_dir (pathlib.Path): Path to the root directory for outputs.
        target (Literal["mmd", "summary"]): Target file type to copy.
    """
    # Loop over all papers.
    pdf_file_paths = sorted(list(paper_root_dir.glob("**/*.pdf")))
    for _, pdf_file_path in enumerate(pdf_file_paths):
        directory_path: pathlib.Path = pdf_file_path.parents[0]

        conference_name: str = pdf_file_path.parents[1].name
        pdf_file_name: str = pdf_file_path.stem

        paper_id: int = int(directory_path.name.split("_")[0])

        target_file_path: pathlib.Path
        if target == "mmd":
            target_file_path = directory_path / (pdf_file_name + "_mathpix.txt")
            # Check if mathpix file exists or not.
            if not target_file_path.exists():
                raise FileNotFoundError(
                    f"`{str(target_file_path)}` does not exist. Please run `convert_to_mmd.py` first to get mmd format text file."
                )
        elif target == "summary":
            target_file_path = directory_path / (pdf_file_name + "_summary.json")
            # Check if summary file exists or not.
            if not target_file_path.exists():
                raise FileNotFoundError(
                    f"`{str(target_file_path)}` does not exist. Please run `generate_summaries.py` first to get summary JSON file."
                )
        else:
            raise ValueError(f"Invalid target: {target}")

        output_directory_path: pathlib.Path = output_root_dir / conference_name / target
        output_directory_path.mkdir(parents=True, exist_ok=True)
        output_file_name: str = f"{paper_id:04d}_{target_file_path.name}"
        output_file_path: pathlib.Path = output_directory_path / output_file_name

        if output_file_path.exists():
            logger.info(f"`{str(output_file_path)}` already exists. Skip copying the file.")
            continue

        # Copy target_file_path to output_file_path
        logger.info(f"Copying `{str(target_file_path)}` to `{str(output_file_path)}`")
        shutil.copyfile(target_file_path, output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--paper-root-dir",
        "-p",
        type=pathlib.Path,
        help="Path to the root directory containing PDF files.",
    )
    parser.add_argument(
        "--output-root-dir",
        "-o",
        type=pathlib.Path,
        help="Path to the root directory for outputs.",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["mmd", "summary"],
        help="Target file type to copy.",
    )

    args = parser.parse_args()
    create_outputs(
        paper_root_dir=args.paper_root_dir,
        output_root_dir=args.output_root_dir,
        target=args.target
    )