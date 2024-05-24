import re
from typing import Any

from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from pydantic import (
    BaseModel,
    HttpUrl,
    ValidationInfo,
    field_validator,
    model_validator,
)


class Paper(BaseModel):
    """Pydantic model which stores single paper infomation.

    Attributes:
        title (str): The title of the paper.
        author (str): The author of the paper.
        abstract (str): The abstract of the paper.
        cvf (HttpUrl): The URL of the CVF page.
        pdf (HttpUrl): The URL of the PDF file.
    """

    title: str
    author: str
    abstract: str
    cvf: HttpUrl
    pdf: HttpUrl


class Subsection(BaseModel):
    """Pydantic model which stores subsection infomation.

    Attributes:
        subsection_id (int): The id of the subsection.
        subsection_title (str): The title of the subsection.
        subsection_text (str): The contents text of the subsection.
    """

    subsection_id: int
    subsection_title: str
    subsection_text: str

    @model_validator(mode="before")
    @classmethod
    def check_subsection_text_contains_subsection_id(cls, data: Any) -> Any:
        """Check if the subsection text contains the subsection id.

        Returns:
            Subsection: The subsection object.
        """
        if not isinstance(data, dict):
            raise ValueError("Data should be a dictionary.")
        subsection_number_part = data["subsection_title"].split(" ", 1)[0]
        try:
            subsection_number_list = subsection_number_part.split(".")
            assert len(subsection_number_list) > 1
            for subsection_number in subsection_number_list:
                if subsection_number == "":
                    continue
                int(subsection_number)
        except ValueError as e:
            raise ValueError(
                f"The subsection number part should be an integer, but it is {subsection_number_part}."
            ) from e
        except AssertionError as e:
            raise ValueError(
                f"The subsection number part should contain more than one number, but it is {subsection_number_part}."
            ) from e

        return data


class Section(BaseModel):
    """Pydantic model which stores section infomation.

    Attributes:
        section_id (int): The id of the section.
        section_title (str): The title of the section.
        section_text (str): The contents text belonging directly to the section.
        subsection_list (list[Subsection]): The list of subsections in the section.
    """

    section_id: int
    section_title: str
    section_text: str
    subsection_list: list[Subsection]

    @model_validator(mode="before")
    @classmethod
    def check_section_text_contains_section_id(cls, data: Any) -> Any:
        """Check if the section text contains the section id.

        Returns:
            Section: The section object.
        """
        if not isinstance(data, dict):
            raise ValueError("Data should be a dictionary.")
        section_number_part = data["section_title"].split(" ", 1)[0]

        reference_pattern = re.compile(r"references?", re.IGNORECASE)
        acknowledge_pattern = re.compile(r"acknowledg(?:e?ment)s?", re.IGNORECASE)
        conclusion_pattern = re.compile(r"conclusion", re.IGNORECASE)
        limitation_pattern = re.compile(r"limitation", re.IGNORECASE)
        appendix_pattern = re.compile(r"appendix", re.IGNORECASE)
        reference_matches = reference_pattern.findall(section_number_part)
        acknowledgment_matches = acknowledge_pattern.findall(section_number_part)
        conclusion_matches = conclusion_pattern.findall(section_number_part)
        limtation_matches = limitation_pattern.findall(section_number_part)
        appendix_matches = appendix_pattern.findall(section_number_part)
        if (
            len(reference_matches) != 0
            or len(acknowledgment_matches) != 0
            or len(conclusion_matches) != 0
            or len(limtation_matches) != 0
            or len(appendix_matches) != 0
            or data["section_title"] == "Ethics Statement"
            or data["section_title"] == "Disclosure of Funding"
            or data["section_title"] == "Reproducibility Statement"
        ):
            return data

        try:
            section_number = int(section_number_part.split(".")[0])
            assert section_number == data["section_id"]
        except ValueError as e:
            raise ValueError(
                f"The section number part should be an integer, but it is {section_number_part}."
            ) from e
        except AssertionError as e:
            raise ValueError(
                f'The section number should match the section id {data["section_id"]}, but it is {section_number}.'
            ) from e

        return data


class ParsedPaper(BaseModel):
    """Pydantic model which stores parsed paper infomation.

    Attributes:
        abstract (str): The abstract of the paper.
        section_list (list[Section]): The list of sections in the paper.
    """

    abstract: str
    section_list: list[Section]

    @field_validator("section_list")
    @classmethod
    def section_list_not_empty(
        cls, v: list[Section], info: ValidationInfo
    ) -> list[Section]:
        """Check if the section list is not empty.

        Args:
            v (list[Section]): The section list.
            info (ValidationInfo): The validation information.

        Returns:
            list[Section]: The section list.
        """
        if len(v) == 0:
            raise ValueError("Section list should not be empty.")
        return v

    @classmethod
    def parse_mmd_text(cls, raw_mmd_text: str) -> "ParsedPaper":
        """Parse the raw Mathpix markdown(mmd) format text OCR-ed from the PDF.

        Args:
            raw_mmd_text (str): The raw mmd format text.

        Returns:
            ParsedPaper: The parsed document of the paper.
        """
        # Remove metadata contents before abstract
        _, contents = raw_mmd_text.split("\\begin{abstract}", 1)

        # Extract abstract
        abstract, contents_wo_abstract = contents.split("\n\\end{abstract}", 1)

        # Split sections
        raw_section_list = contents_wo_abstract.lstrip("\n").split("\\section*{")
        section_list: list[Section] = list()
        section_id = 1
        for i, each_section in enumerate(raw_section_list):
            if i == 0:
                continue
            section_title, raw_section_text = each_section.split("}\n", 1)
            section_text = raw_section_text.lstrip("\n")
            section_text = cls.simple_figure_table_remover(section_text)
            section_dict = {
                "section_id": section_id,
                "section_title": section_title,
                "section_text": section_text,
                "subsection_list": list(),
            }
            section_list.append(Section.model_validate(section_dict))
            section_id += 1

        # Split subsections
        for each_section_model in section_list:
            raw_subsection_list = each_section_model.section_text.split(
                "\\subsection*{"
            )
            # Go into next section if there is no subsection
            if len(raw_subsection_list) == 1:
                continue
            subsection_list: list[Subsection] = list()
            subsection_id = 0
            for each_subsection in raw_subsection_list:
                # If there is no text between \section{} and \subsection{}, skip it
                if len(each_subsection) == 0:
                    subsection_id += 1
                    each_section_model.section_text = ""
                    continue
                # If there is text between \section{} and \subsection{}, update section_text
                if subsection_id == 0 and len(each_subsection) != 0:
                    each_section_model.section_text = each_subsection
                    subsection_id += 1
                    continue

                subsection_title, raw_subsection_text = each_subsection.split("}\n", 1)
                subsection_text = raw_subsection_text.lstrip("\n")
                subsection_dict = {
                    "subsection_id": subsection_id,
                    "subsection_title": subsection_title,
                    "subsection_text": subsection_text,
                }
                subsection_list.append(Subsection.model_validate(subsection_dict))
                subsection_id += 1

            each_section_model.subsection_list = subsection_list

        parsed_document = {
            "abstract": abstract,
            "section_list": section_list,
        }

        return ParsedPaper.model_validate(parsed_document)

    @staticmethod
    def simple_figure_table_remover(text: str) -> str:
        """Remove figure and table from the mmd text.

        Args:
            text (str): The source mmd text.

        Returns:
            str: The mmd text without figure and table.
        """
        wo_table_text = re.sub(
            r"\\begin{tabular}(.*?)\\end{tabular}", "", text, flags=re.DOTALL
        )
        wo_fig_table_text = re.sub(
            r"!\[\]\((.*?)\)\n", "", wo_table_text, flags=re.DOTALL
        )
        wo_fig_table_text = re.sub(
            r"\nFigure(.*?)\n\n", "", wo_fig_table_text, flags=re.DOTALL
        )
        return wo_fig_table_text

    def structure_mmd_documents(
        self,
        text_splitter: TextSplitter,
        abstract_text: str | None = None,
    ) -> list[Document]:
        """Structure the parsed paper into documents.

        Args:
            text_splitter (TextSplitter): The text splitter object of langchain.
            abstract_text (str | None): The full abstract text.

        Returns:
            list[Document]: A list of structured documents.
        """
        # If full abstract is provided, use it instead of parsed one.
        documents = [
            Document(
                page_content=abstract_text if abstract_text else self.abstract,
                metadata={"section": "abstract"},
            )
        ]

        # Loop over section.
        for each_section in self.section_list:
            section_title = each_section.section_title
            if section_title == "References":
                continue

            section_id = each_section.section_id
            section_text = each_section.section_text
            if section_text:
                metadata = {
                    "section_id": f"{section_id}",
                    "section": f"{section_title}",
                }
                for each_section_text in text_splitter.split_text(section_text):
                    documents.append(
                        Document(page_content=each_section_text, metadata=metadata)
                    )

            # Loop over subsection.
            for each_subsection in each_section.subsection_list:
                subsection_title = each_subsection.subsection_title
                subsection_id = each_subsection.subsection_id
                subsection_text = each_subsection.subsection_text
                metadata = {
                    "section_id": f"{section_id}.{subsection_id}",
                    "section": f"{section_title}/{subsection_title}",
                }
                for each_subsection_text in text_splitter.split_text(subsection_text):
                    documents.append(
                        Document(page_content=each_subsection_text, metadata=metadata)
                    )

        return documents
