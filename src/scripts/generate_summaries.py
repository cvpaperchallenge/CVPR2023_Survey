"""

This script generates summaries in ochiai format. This script requires
the following environmental variable.

- OPENAI_API_KEY

"""
import argparse
import json
import logging
import pathlib
from typing import Final

from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.cvf_page_parser import Paper
from src.mmd_text_parser import parse_mmd_text, structure_latex_documents
from src.summarizer import OchiaiFormatPaperSummarizer

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Note: list config
llm_model_name: str = "gpt-3.5-turbo-0125"  # "gpt-4-0125-preview"
temperature: float = 0.9
chunk_size: int = 200
chunk_overlap: int = 40


def generate_summaries_in_ochiai_format(
    paper_root_dir: pathlib.Path,
    paper_info_path: pathlib.Path,
    prompt_template_dir: pathlib.Path,
    verbose: bool = False,
) -> None:
    """Generate summaries of all papers in Ochiai format.

    Args:
        paper_root_dir (pathlib.Path): Path to the directory containing PDF files.
        paper_info_path (pathlib.Path): Path to the JSON file which contains paper information.
        prompt_template_dir (pathlib.Path): Path to the directory containing prompt templates.
        verbose (bool): Print used prompts, generated summaries, and token usage.
    """
    # Check JSON file existence.
    if not paper_info_path.exists():
        error_message: Final = f"The file `{str(paper_info_path)}` does not exist. \
            Please run `parse_cvf_page.py` first to generate JSON file."
        raise FileNotFoundError(error_message)

    # Load JSON and validate by Pydantic model.
    with paper_info_path.open("r") as f:
        papers: Final = [Paper.model_validate(p) for p in json.load(f)]

    # Loop over all papers.
    pdf_file_paths = sorted(list(paper_root_dir.glob("**/*.pdf")))
    for i, pdf_file_path in enumerate(pdf_file_paths):
        directory_path = pdf_file_path.parent
        stem = pdf_file_path.stem
        paper_id = int(directory_path.name.split("_")[0])

        # Check if PDF file exists or not.
        pdf_file_path = directory_path / (stem + ".pdf")
        if not pdf_file_path.exists():
            raise FileNotFoundError(
                f"`{str(pdf_file_path)}` does not exist. Please run `download_papers.py` first to download PDF file."
            )

        # If there is no mathpix file, send PDF to mathpix API.
        mathpix_file_path = directory_path / (stem + "_mathpix.txt")
        if not mathpix_file_path.exists():
            raise FileNotFoundError(
                f"`{str(mathpix_file_path)}` does not exist. Please run `convert_to_latex.py` first to get latex format text file."
            )

        # If summary already exists, continue the loop.
        summary_file_path = directory_path / (stem + "_summary.json")
        if summary_file_path.exists():
            logger.info(
                f"`{str(summary_file_path)}` already exists. Continue the loop."
            )
            continue

        # Parse Latex format text.
        raw_paper = TextLoader(file_path=str(mathpix_file_path)).load()[0]
        parsed_paper = parse_mmd_text(raw_paper.page_content)

        # Convert text into to structured documents
        text_splitter = TokenTextSplitter.from_tiktoken_encoder(
            model_name=llm_model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        documents = structure_latex_documents(
            parsed_paper,
            text_splitter,
            papers[paper_id].abstract,
        )

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        # Load vector database if it exists.
        if (directory_path / "index").exists() and (
            directory_path / "index_wo_abstract"
        ).exists():
            vectorstore = FAISS.load_local(
                str(directory_path / "index"),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            vectorstore_wo_abstract = FAISS.load_local(
                str(directory_path / "index_wo_abstract"),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            # Embed documents and store into vector database.
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=embeddings,
            )
            vectorstore_wo_abstract = FAISS.from_documents(
                documents=[
                    document
                    for document in documents
                    if document.metadata["section"] != "abstract"
                ],
                embedding=embeddings,
            )

            # Save vector database.
            vectorstore.save_local(str(directory_path / "index"))
            vectorstore_wo_abstract.save_local(
                str(directory_path / "index_wo_abstract")
            )

        # Generate summary.
        llm_model = ChatOpenAI(model_name=llm_model_name, temperature=temperature)  # type: ignore
        summarizer = OchiaiFormatPaperSummarizer(
            llm_model=llm_model,
            vectorstore={
                "all": vectorstore,
                "wo_abstract": vectorstore_wo_abstract,
            },
            prompt_template_dir=prompt_template_dir,
            verbose=verbose,
        )
        summary = summarizer.summarize()

        # Save summary.
        with summary_file_path.open("w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-pdf-dir",
        "-i",
        type=pathlib.Path,
        required=True,
        help="Path to the directory containing PDF files.",
    )
    parser.add_argument(
        "--paper-info-path",
        "-j",
        type=pathlib.Path,
        required=True,
        help="Path to the JSON file which contains paper information.",
    )
    parser.add_argument(
        "--prompt-template-dir",
        "-p",
        type=pathlib.Path,
        default="./src/prompts",
        help="Path to the directory containing prompt templates.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print used prompts, generated summaries, and token usage.",
    )

    args = parser.parse_args()
    generate_summaries_in_ochiai_format(
        paper_root_dir=args.input_pdf_dir,
        paper_info_path=args.paper_info_path,
        prompt_template_dir=args.prompt_template_dir,
        verbose=args.verbose,
    )
