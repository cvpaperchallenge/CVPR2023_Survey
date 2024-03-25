from typing import Any

import requests
from langchain.docstore.document import Document
from langchain.document_loaders.pdf import MathpixPDFLoader


class CustomMathpixPDFLoader(MathpixPDFLoader):
    """Load `PDF` files using `Mathpix` service.

    This class extends `MathpixPDFLoader` class implemented in
    langchain to support mmd format conversion.

    """

    def __init__(
        self,
        file_path: str,
        processed_file_format: str = "md",
        max_wait_time_seconds: int = 500,
        should_clean_pdf: bool = False,
        extra_request_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            file_path,
            processed_file_format,
            max_wait_time_seconds,
            should_clean_pdf,
            extra_request_data,
            **kwargs,
        )

    def get_processed_pdf_in_mmd_format(self, pdf_id: str) -> str:
        self.wait_for_processing(pdf_id)
        url = f"{self.url}/{pdf_id}.mmd"
        response = requests.get(url, headers=self._mathpix_headers)
        return response.content.decode("utf-8")

    def load_mmd(self) -> list[Document]:
        pdf_id = self.send_pdf()
        contents = self.get_processed_pdf_in_mmd_format(pdf_id)
        if self.should_clean_pdf:
            contents = self.clean_pdf(contents)
        metadata = {"source": self.source, "file_path": self.source, "pdf_id": pdf_id}
        return [Document(page_content=contents, metadata=metadata)]
