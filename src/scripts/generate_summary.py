"""

This script generates summaries of all CVPR 2023 papers.

"""
import json
import logging
import pathlib
from typing import Final

import requests

from langchain.embeddings.openai import OpenAIEmbeddings
from src.parser import Paper
from langchain.document_loaders import TextLoader
from src.latex_parser import parse_latex_text, structure_latex_documents
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Note: list config 
paper_info_path: Final = pathlib.Path("./data/papers.json")
paper_root_path: Final = pathlib.Path("./data/papers/")
llm_model_name: str = "gpt-3.5-turbo"
temperature: float = 0.9
chunk_size: int = 200
chunk_overlap: int = 40

# Check JSON file existence.
if not paper_info_path.exists():
    error_message: Final = f"This scripts requires `{str(paper_info_path)}`. \
        Please run `parse_cvf_page.py` frist to generate JSON file."
    raise FileNotFoundError(error_message)

# Load JSON and validate by Pydantic model.
with paper_info_path.open("r") as f:
    papers: Final = [Paper.parse_obj(p) for p in json.load(f)]

text_splitter: Final = TokenTextSplitter.from_tiktoken_encoder(
    model_name=llm_model_name,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
embeddings: Final = OpenAIEmbeddings()
llm_model: Final = ChatOpenAI(model_name=llm_model_name, temperature=temperature)

# Loop over all papers.
for i, paper in enumerate(papers):
    # stem is like: <family_name>_<paper_title>_CVPR_2023_paper
    stem = str(pathlib.Path(paper.pdf).stem)
    directory_path = paper_root_path / stem.removesuffix("_CVPR_2023_paper")

    # Check if PDF file exists or not.
    pdf_file_path = directory_path / (stem + ".pdf")
    if not pdf_file_path.exists():
        raise FileNotFoundError(f"`{str(pdf_file_path)}` does not exist. Please run `download_papers.py` frist to download PDF file.")

    # If there is no mathpix file, send PDF to mathpix API.
    mathpix_file_path = directory_path / (stem + "_mathpix.txt")
    if not mathpix_file_path.exists():
        # Call mathpix API here.
        pass

    # Parse Latex format text and create
    raw_paper = TextLoader(file_path=str(mathpix_file_path)).load()[0]
    parsed_paper = parse_latex_text(raw_paper.page_content)
    documents = structure_latex_documents(
        parsed_paper,
        text_splitter,
        paper.abstract,
    )

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    vectorstore_witout_abstract = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # summarizer = OchiaiFormatPaperSummarizer(
    #     llm_model=llm_model,
    #     vectorstore=vectorstore,
    #     prompt_template_dir_path=pathlib.Path("./src/prompts")
    # )