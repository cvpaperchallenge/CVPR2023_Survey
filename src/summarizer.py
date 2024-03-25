import logging
import pathlib
from abc import ABC
from typing import Any, Final

from jinja2 import Environment, FileSystemLoader
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStore
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FormatOchiai(BaseModel):
    outline: str = Field(description="どんなもの？")
    contribution: str = Field(description="先行研究と比べてどこがすごい？")
    method: str = Field(description="技術や手法のキモはどこ？")
    evaluation: str = Field(description="どうやって有効だと検証した？")
    discussion: str = Field(description="議論はある？")


class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)
        # Log prompts in green color
        logger.info(f"\033[92mPrompt:\n{formatted_prompts}\033[0m")


class BasePaperSummarizer(ABC):
    """ """

    def __init__(
        self,
        llm_model: BaseLanguageModel,
        vectorstore: dict[str, VectorStore],
        prompt_template_dir: pathlib.Path,
        verbose: bool,
    ) -> None:
        self.llm_model = llm_model
        self.vectorstore = vectorstore
        self.template_env = Environment(
            loader=FileSystemLoader(str(prompt_template_dir))
        )
        self.verbose = verbose

    def _summarize(self) -> Any:
        raise NotImplementedError

    def summarize(self) -> FormatOchiai:
        """"""
        if self.verbose:
            with get_openai_callback() as cb:
                summary = self._summarize()
                # Log the token usage in red color
                logger.info(f"\033[91mToken Usage:\n{cb}\033[0m")
                # logger.info(cb)
                return summary
        else:
            return self._summarize()


class OchiaiFormatPaperSummarizer(BasePaperSummarizer):
    def __init__(
        self,
        llm_model: BaseLanguageModel,
        vectorstore: dict[str, VectorStore],
        prompt_template_dir: pathlib.Path,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            llm_model=llm_model,
            vectorstore=vectorstore,
            prompt_template_dir=prompt_template_dir,
            verbose=verbose,
        )

    def _summarize(self) -> FormatOchiai:
        outline = self._summarize_outline()
        contribution = self._summarize_contribution()
        method = self._summarize_method()
        evaluation = self._summarize_evaluation()
        discussion = self._summarize_discussion()
        return FormatOchiai(
            outline=outline,
            contribution=contribution,
            method=method,
            evaluation=evaluation,
            discussion=discussion,
        )

    def _summarize_outline(self) -> str:
        """`どんなもの？`"""
        prompt_template: Final = self.template_env.get_template(
            "outline_ja.jinja2"
        ).render()
        outline_prompt = PromptTemplate.from_template(
            template=prompt_template,
        )
        outline_chain = (
            RunnablePassthrough.assign(
                text=(
                    lambda inputs: "\n\n".join(
                        doc.page_content for doc in inputs["selected_documents"]
                    )
                )
            ).with_config(run_name="combine_documents")
            | outline_prompt
            | self.llm_model
            | StrOutputParser()
        ).with_config(run_name="outline_chain")

        retriever = self.vectorstore["wo_abstract"].as_retriever(
            serch_type="similarity",
            search_kwargs={"k": 1},
        )

        selected_documents = []
        # Get top-k relevant documents
        proposed_method: Final = retriever.get_relevant_documents("Proposed method")
        experiments: Final = retriever.get_relevant_documents("Experiments")
        resutls: Final = retriever.get_relevant_documents("Results")

        abstract_docstore_id = self.vectorstore["all"].index_to_docstore_id[0]
        abstract_document = self.vectorstore["all"].docstore._dict[abstract_docstore_id]
        selected_documents.append(abstract_document)
        selected_documents.extend(proposed_method)
        selected_documents.extend(experiments)
        selected_documents.extend(resutls)
        outline_sumamry = outline_chain.invoke(
            {"selected_documents": selected_documents},
            config={"callbacks": [CustomHandler()]} if self.verbose else None,
        )
        if self.verbose:
            # Log the outline summary in blue color
            logger.info(f"\033[94mOutline Summary:\n{outline_sumamry}\033[0m")
        return outline_sumamry

    def _summarize_contribution(self) -> str:
        """`先行研究と比べてどこがすごい？`"""
        contribution_query: Final = "The contribution of this study"
        problem_query: Final = "The problems with previous studies"
        contribution = self._run_combine_document_chain(
            query=contribution_query,
            prompt_template_filename="contribution_ja.jinja2",
        )
        problem = self._run_combine_document_chain(
            query=problem_query,
            prompt_template_filename="problem_ja.jinja2",
        )

        combine_template: Final = self.template_env.get_template(
            "combination_ja.jinja2"
        ).render()
        overall_prompt = PromptTemplate.from_template(
            template=combine_template,
        )
        overall_chain = overall_prompt | self.llm_model | StrOutputParser()
        contribution_summary = overall_chain.invoke(
            {
                "contribution": contribution,
                "problem": problem,
            },
            config={"callbacks": [CustomHandler()]} if self.verbose else None,
        )

        if self.verbose:
            # Log the contribution summary in blue color
            logger.info(f"\033[94mContribution Summary:\n{contribution_summary}\033[0m")
        return contribution_summary

    def _summarize_method(self) -> str:
        """`技術や手法のキモはどこ？`"""
        query: Final = "The proposed method and dataset in this study"

        method_summary = self._run_combine_document_chain(
            query=query,
            prompt_template_filename="method_ja.jinja2",
        )
        if self.verbose:
            # Log the method summary in blue color
            logger.info(f"\033[94mMethod Summary:\n{method_summary}\033[0m")
        return method_summary

    def _summarize_evaluation(self) -> str:
        """`どうやって有効だと検証した？`"""
        query: Final = "The experiments conducted in this study and their evaluation"

        evaluation_summary = self._run_combine_document_chain(
            query=query,
            prompt_template_filename="evaluation_ja.jinja2",
        )
        if self.verbose:
            # Log the evaluation summary in blue color
            logger.info(f"\033[94mEvaluation Summary:\n{evaluation_summary}\033[0m")
        return evaluation_summary

    def _summarize_discussion(self) -> str:
        """`議論はある？`"""
        query: Final = "The authors' analysis and future prospects based on the results of the evaluation of this study"

        discussion_summary = self._run_combine_document_chain(
            query=query,
            prompt_template_filename="discussion_ja.jinja2",
        )
        if self.verbose:
            # Log the discussion summary in blue color
            logger.info(f"\033[94mDiscussion Summary:\n{discussion_summary}\033[0m")
        return discussion_summary

    def _run_combine_document_chain(
        self,
        query: str,
        prompt_template_filename: str,
        search_type: str = "similarity",
        search_kwargs: dict[str, int] = {"k": 5},
    ) -> str:
        """ """
        prompt_template: Final = self.template_env.get_template(
            prompt_template_filename
        ).render()
        prompt: Final = PromptTemplate.from_template(
            template=prompt_template,
        )

        retriever = self.vectorstore["all"].as_retriever(
            serch_type=search_type,
            search_kwargs=search_kwargs,
        )

        combine_document_chain = (
            RunnablePassthrough.assign(
                selected_documents=(
                    (lambda inputs: inputs["query"]) | retriever
                ).with_config(run_name="retrieve_documents"),
            )
            | RunnablePassthrough.assign(
                text=(
                    lambda inputs: "\n\n".join(
                        doc.page_content for doc in inputs["selected_documents"]
                    )
                )
            ).with_config(run_name="combine_documents")
            | prompt
            | self.llm_model
            | StrOutputParser()
        )

        return combine_document_chain.invoke(
            {"query": query},
            config={"callbacks": [CustomHandler()]} if self.verbose else None,
        )


if __name__ == "__main__":
    from langchain.text_splitter import TokenTextSplitter
    from langchain_community.document_loaders.text import TextLoader
    from langchain_community.vectorstores.faiss import FAISS
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings

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
            metadata = {"section_id": f"{section_id}", "section": f"{section_title}"}
            for each_section_text in text_splitter.split_text(section_text):
                documents_for_search.append(
                    Document(page_content=each_section_text, metadata=metadata)
                )
        for each_subsection in each_section["subsection_list"]:
            subsection_title = each_subsection["subsection_title"]
            subsection_id = each_subsection["subsection_id"]
            subsection_text = each_subsection["subsection_text"]
            metadata = {
                "section_id": f"{section_id}.{subsection_id}",
                "section": f"{section_title}/{subsection_title}",
            }
            for each_subsection_text in text_splitter.split_text(subsection_text):
                documents_for_search.append(
                    Document(page_content=each_subsection_text, metadata=metadata)
                )

    documents.extend(documents_for_search)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    vectorstore_for_search = FAISS.from_documents(
        documents=documents_for_search,
        embedding=embeddings,
    )

    llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

    summarizer = OchiaiFormatPaperSummarizer(
        llm_model=llm_model,
        vectorstore={"all": vectorstore, "wo_abstract": vectorstore_for_search},
        prompt_template_dir=pathlib.Path("./src/prompts"),
        verbose=True,
    )

    result = summarizer.summarize()
    print(result)
