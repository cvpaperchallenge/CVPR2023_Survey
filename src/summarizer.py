import logging
import pathlib
from abc import ABC
from typing import Any, Dict, Final

from jinja2 import Environment, FileSystemLoader
from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FormatOchiai(BaseModel):
    # outline: str = Field(description="どんなもの？")
    contribution: str = Field(description="先行研究と比べてどこがすごい？")
    method: str = Field(description="技術や手法のキモはどこ？")
    evaluation: str = Field(description="どうやって有効だと検証した？")
    discussion: str = Field(description="議論はある？")


class BasePaperSummarizer(ABC):
    """ """

    def __init__(
        self,
        llm_model: BaseLanguageModel,
        vectorstore: VectorStore,
        prompt_template_dir_path: pathlib.Path,
    ) -> None:
        self.llm_model = llm_model
        self.vectorstore = vectorstore
        self.template_env = Environment(
            loader=FileSystemLoader(str(prompt_template_dir_path))
        )

    def summarize(self) -> Any:
        raise NotImplementedError


class OchiaiFormatPaperSummarizer(BasePaperSummarizer):
    def __init__(
        self,
        llm_model: BaseLanguageModel,
        vectorstore: VectorStore,
        prompt_template_dir_path: pathlib.Path,
    ) -> None:
        super().__init__(
            llm_model=llm_model,
            vectorstore=vectorstore,
            prompt_template_dir_path=prompt_template_dir_path,
        )
        pass

    def summarize(self) -> FormatOchiai:
        """"""
        # outline = self._summarize_outline()
        contribution = self._summarize_contribution()
        method = self._summarize_method()
        evaluation = self._summarize_evaluation()
        discussion = self._summarize_discussion()
        return FormatOchiai(
            # outline=outline,
            contribution=contribution,
            method=method,
            evaluation=evaluation,
            discussion=discussion,
        )

    def _summarize_outline(self) -> str:
        """`どんなもの？`"""
        pass

    def _summarize_contribution(self) -> str:
        """`先行研究と比べてどこがすごい？`"""
        contribution_query: Final = "The contribution of this study"
        problem_query: Final = "The problems with previous studies"
        contribution = self._run_combine_document_chain(
            query=contribution_query,
            prompt_template_filename="contribution_ja.jinja2",
            prompt_input_variable="contribution_text",
        )
        problem = self._run_combine_document_chain(
            query=problem_query,
            prompt_template_filename="problem_ja.jinja2",
            prompt_input_variable="problem_text",
        )

        combine_template: Final = self.template_env.get_template(
            "combination_ja.jinja2"
        ).render()
        overall_prompt = PromptTemplate(
            input_variables=["contribution", "problem"],
            template=combine_template,
        )
        overall_chain = LLMChain(
            llm=self.llm_model, prompt=overall_prompt, verbose=True
        )
        return overall_chain.run(
            {
                "contribution": contribution,
                "problem": problem,
            }
        )

    def _summarize_method(self) -> str:
        """`技術や手法のキモはどこ？`"""
        query: Final = "The proposed method and dataset in this study"

        return self._run_combine_document_chain(
            query=query,
            prompt_template_filename="method_ja.jinja2",
            prompt_input_variable="text",
        )

    def _summarize_evaluation(self) -> str:
        """`どうやって有効だと検証した？`"""
        query: Final = "The experiments conducted in this study and their evaluation"

        return self._run_combine_document_chain(
            query=query,
            prompt_template_filename="evaluation_ja.jinja2",
            prompt_input_variable="text",
        )

    def _summarize_discussion(self) -> str:
        """`議論はある？`"""
        query: Final = "The authors' analysis and future prospects based on the results of the evaluation of this study"

        return self._run_combine_document_chain(
            query=query,
            prompt_template_filename="discussion_ja.jinja2",
            prompt_input_variable="text",
        )

    def _run_combine_document_chain(
        self,
        query: str,
        prompt_template_filename: str,
        prompt_input_variable: str,
        search_type: str = "similarity",
        search_kwargs: Dict = {"k": 5},
        verbose: bool = True,
    ) -> str:
        """ """
        prompt_template: Final = self.template_env.get_template(
            prompt_template_filename
        ).render()
        prompt: Final = PromptTemplate(
            template=prompt_template, input_variables=[prompt_input_variable]
        )

        chain: Final = LLMChain(llm=self.llm_model, prompt=prompt, verbose=verbose)
        combine_document_chain: Final = StuffDocumentsChain(
            llm_chain=chain,
            document_variable_name=prompt_input_variable,
            verbose=verbose,
        )

        retriever = self.vectorstore.as_retriever(
            serch_type=search_type,
            search_kwargs=search_kwargs,
        )
        result: Final = retriever.get_relevant_documents(query)
        return combine_document_chain.run(result)


if __name__ == "__main__":
    from langchain.docstore.document import Document
    from langchain.document_loaders import TextLoader
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.text_splitter import TokenTextSplitter
    from langchain.vectorstores import FAISS

    from src.latex_parser import parse_latex_text

    txt_path = pathlib.Path("./tests/data/visual_atoms.txt")
    raw_papers = TextLoader(file_path=txt_path).load()
    parsed_paper = parse_latex_text(raw_papers[0].page_content)

    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",  # "text-embedding-ada-002"
        chunk_size=200,
        chunk_overlap=40,
    )

    documents = []
    documents_for_search = []
    abstract_document = Document(
        page_content=parsed_paper["abstract"], metadata={"section": "abstract"}
    )
    documents.append(abstract_document)
    for each_section in parsed_paper["section"]:
        section_title = each_section["section_title"]
        if section_title == "References":
            continue
        section_id = each_section["section_id"]
        section_text = each_section["section_text"]
        if section_text != "":
            metadata = {"section_id": "{section_id}", "section": "{section_title}"}
            for each_section_text in text_splitter.split_text(section_text):
                documents_for_search.append(
                    Document(page_content=each_section_text, metadata=metadata)
                )
        for each_subsection in each_section["subsection_list"]:
            subsection_title = each_subsection["subsection_title"]
            subsection_id = each_subsection["subsection_id"]
            subsection_text = each_subsection["subsection_text"]
            metadata = {
                "section_id": "{section_id}.{subsection_id}",
                "section": "{section_title}/{subsection_title}"
            }
            for each_subsection_text in text_splitter.split_text(subsection_text):
                documents_for_search.append(
                    Document(page_content=each_subsection_text, metadata=metadata)
                )

    documents.extend(documents_for_search)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

    summarizer = OchiaiFormatPaperSummarizer(
        llm_model=llm_model,
        vectorstore=vectorstore,
        prompt_template_dir_path=pathlib.Path("./src/prompts"),
    )

    result = summarizer.summarize()
    print(result)
