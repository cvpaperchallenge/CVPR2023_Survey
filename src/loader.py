from __future__ import annotations

import json
from typing import Any

from langchain.docstore.document import Document
from langchain.document_loaders import MathpixPDFLoader


class CustomMathpixLoader(MathpixPDFLoader):
    """Loader for mathpix.

    NOTE: This class extends `MathpixPDFLoader` class implemented in
    langchain to support request paramters.

    """

    def __init__(
        self,
        file_path: str,
        processed_file_format: str = "mmd",
        max_wait_time_seconds: int = 500,
        should_clean_pdf: bool = False,
        output_langchain_document: bool = True,
        other_request_parameters: dict = {},
        **kwargs: Any,
    ) -> None:
        self.output_langchain_document = output_langchain_document
        self.other_request_parameters = other_request_parameters
        super().__init__(
            file_path,
            processed_file_format,
            max_wait_time_seconds,
            should_clean_pdf,
            **kwargs,
        )

    @property
    def data(self) -> dict:
        options = {
            "conversion_formats": {self.processed_file_format: True},
            **self.other_request_parameters,
        }
        return {"options_json": json.dumps(options)}

    def load(self) -> list[Document] | str:
        pdf_id = self.send_pdf()
        contents = self.get_processed_pdf(pdf_id)
        if self.should_clean_pdf:
            contents = self.clean_pdf(contents)
        if self.output_langchain_document:
            metadata = {"source": self.source, "file_path": self.source}
            output = [Document(page_content=contents, metadata=metadata)]
        else:
            output = contents
        return output
