import logging
import pathlib
from abc import ABC
from typing import Dict, Final, List
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from jinja2 import Template, Environment, FileSystemLoader

logger: Final = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class BasePaperSummarizer(ABC):
    """
    """
    def __init__(
        self,
        llm_model,
        vectorstore,
        prompt_template_dir_path: pathlib.Path,
    ) -> None:
        self.llm_model=llm_model
        self.vectorstore=vectorstore
        self.template_env = Environment(loader=FileSystemLoader(str(prompt_template_dir_path)))

    def summarize(self):
        raise NotImplementedError


class OchiaiFormatPaperSummarizer(BasePaperSummarizer):
    def __init__(
        self,
        llm_model,
        vectorstore,
        prompt_template_dir_path: pathlib.Path,
    ) -> None:
        super().__init__(
            llm_model=llm_model,
            vectorstore=vectorstore,
            prompt_template_dir_path=prompt_template_dir_path,
        )
        pass

    def summarize(self):
        """"""
        pass
        # call all _summarize_* methods here.

    def _summarize_outline(self):
        """`どんなもの？`"""
        pass

    def _summarize_contribution(self):
        """`先行研究と比べてどこがすごい？`"""
        pass

    def _summarize_method(self):
        """`技術や手法のキモはどこ？`"""
        query: Final = "The proposed method and dataset in this study"

        return self._run_combine_document_chain(
            query=query,
            prompt_template_filename="method_ja.jinja2",
            prompt_input_variable="text",
        )


    def _summarize_evaluation(self):
        """`どうやって有効だと検証した？`"""
        pass

    def _summarize_discussion(self):
        """`議論はある？`"""
        pass

    def _run_combine_document_chain(
        self,
        query: str,
        prompt_template_filename: str,
        prompt_input_variable: str,
        search_type: str = "similarity",
        search_kwargs: Dict = {"k": 5},
        verbose: bool = True,
    ):
        """
        """
        prompt_template: Final = self.template_env.get_template(prompt_template_filename).render()
        prompt: Final = PromptTemplate(template=prompt_template, input_variables=[prompt_input_variable])

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


if __name__== "__main__":
    from langchain.document_loaders import TextLoader
    from src.latex_parser import parse_latex_text
    from langchain.text_splitter import TokenTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore.document import Document

    txt_path = pathlib.Path("./tests/data/visual_atoms.txt")
    raw_papers = TextLoader(file_path=txt_path).load()
    parsed_paper = parse_latex_text(raw_papers[0].page_content)

    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo", # "text-embedding-ada-002"
        chunk_size = 200,
        chunk_overlap = 40
    )

    documents = []
    documents_for_search = []
    abstract_document = Document(page_content=parsed_paper["abstract"], metadata={"section": "abstract"})
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
        prompt_template_dir_path=pathlib.Path("./src/prompts")
    )

    result = summarizer._summarize_method()
    print(result)