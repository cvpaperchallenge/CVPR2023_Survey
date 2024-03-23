from enum import Enum
from typing import Final

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl


class ConferencePath(Enum):
    """The Enum class which stores the paths to the conference papers available on the cvf page."""
    CVPR = "/CVPR2023?day=all"
    ICCV = "/ICCV2023?day=all"


class Paper(BaseModel):
    """Pydantic model which stores single paper infomation."""

    title: str
    author: str
    abstract: str
    cvf: HttpUrl
    pdf: HttpUrl


def get_paper_page_urls(conference_path: ConferencePath) -> list[str]:
    """Return a list of CVF page URL.

    Return a list of CVF page URL based on the conference path.
    The number of accepted papers is different for each conference:
        - CVPR 2023: 2,359 papers
        - ICCV 2023: 2,156 papers

    Args:
        conference_path (ConferencePath): The paths to the conference papers
            available on the cvf page.

    Returns:
        list[str]: A list of CVF page URL of each paper.
    """

    cvf_root_url: Final = "https://openaccess.thecvf.com"
    cvf_all_paper_url: Final = cvf_root_url + conference_path.value

    html: Final = requests.get(cvf_all_paper_url).text
    bs: Final = BeautifulSoup(html, "html.parser")
    parsed_tags = bs.select(".ptitle > a")
    return [cvf_root_url + parsed_tag.get("href") for parsed_tag in parsed_tags]


def parse_paper_page(page_url: str) -> Paper:
    """Parse a paper page and return Paper object.

    Args:
        page_url (str): The URL of the paper page.

    Returns:
        Paper: The Paper object which stores the paper information.
    """

    html: Final = requests.get(page_url).text
    bs: Final = BeautifulSoup(html, "html.parser")

    title: Final = bs.select_one("#papertitle").text.strip()
    author: Final = bs.select_one("#authors b").text.strip()
    abstract: Final = bs.select_one("#abstract").text.strip()
    cvf: Final = page_url
    pdf: Final = (
        "https://openaccess.thecvf.com/content/CVPR2023/papers/"
        + page_url.removesuffix(".html").split("/")[-1]
        + (".pdf")
    )

    return Paper(
        title=title,
        author=author,
        abstract=abstract,
        cvf=cvf,  # type: ignore
        pdf=pdf,  # type: ignore
    )


if __name__ == "__main__":
    paper = parse_paper_page(
        "https://openaccess.thecvf.com/content/CVPR2023/html/Ci_GFPose_Learning_3D_Human_Pose_Prior_With_Gradient_Fields_CVPR_2023_paper.html"
    )
    print(paper.json())
