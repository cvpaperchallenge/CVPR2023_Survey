from typing import Final

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl


class Paper(BaseModel):
    """Pydantic model which stores single paper infomation."""

    title: str
    author: str
    abstract: str
    cvf: HttpUrl
    pdf: HttpUrl


def validate_conference(conference: str, year: int) -> str:
    """Validate the user specified conference name and year and return the unique conference name with year.

    Args:
        conference (str): The conference name.
        year (int): The year of the conference.

    Returns:
        str: The unique conference name with year.
    """
    if conference == "cvpr":
        if year not in range(2013, 2025):
            raise ValueError(
                "CVPR conference is held from 2013 to 2023. \
                Please specify the year in the range."
            )
        return f"CVPR{year}"
    elif conference == "iccv":
        if year not in range(2013, 2024, 2):
            raise ValueError(
                "ICCV conference is held from 2013 to 2023 every two years. \
                Please specify the year in the range."
            )
        return f"ICCV{year}"
    else:
        raise ValueError(
            f"You specified the conference name as {conference}, \
            but our code does not support the conference."
        )


def get_paper_page_urls(conference: str, year: int) -> list[str]:
    """Return a list of CVF page URL.

    The number of accepted papers is different for each conference:
        - CVPR 2023: 2,359 papers
        - ICCV 2023: 2,156 papers

    Args:
        conference (str): The conference name.
        year (int): The year of the conference.

    Returns:
        list[str]: A list of CVF page URL of each paper.
    """
    cvf_root_url: Final[str] = "https://openaccess.thecvf.com"
    conference_name: Final[str] = validate_conference(conference, year)
    cvf_all_paper_url: Final = cvf_root_url + f"/{conference_name}?day=all"

    html: Final = requests.get(cvf_all_paper_url).text
    bs: Final = BeautifulSoup(html, "html.parser")
    parsed_tags = bs.select(".ptitle > a")
    return [cvf_root_url + parsed_tag.get("href") for parsed_tag in parsed_tags]


def parse_paper_page(page_url: str) -> Paper:
    """Parse a paper page and return Paper object.

    Args:
        page_url (str): The URL of the paper page. The page url structure is like
            https://openaccess.thecvf.com/content/<conference_name><year>/html/<family_name>_<paper_title>_<conference_name>_<year>_paper.html

    Returns:
        Paper: The Paper object which stores the paper information.
    """
    html: Final[str] = requests.get(page_url).text
    bs: Final = BeautifulSoup(html, "html.parser")

    title: Final[str] = bs.select_one("#papertitle").text.strip()
    author: Final[str] = bs.select_one("#authors b").text.strip()
    abstract: Final[str] = bs.select_one("#abstract").text.strip()
    cvf: Final[str] = page_url

    # conference_path is like: https://openaccess.thecvf.com/content/<conference_name><year>
    conference_path: Final[str] = page_url.rsplit("/", 2)[0]
    # paper_name is like: <family_name>_<paper_title>_<conference_name>_<year>_paper
    paper_name: Final[str] = page_url.rsplit("/", 1)[1].removesuffix(".html")
    pdf: Final[str] = conference_path + "/papers/" + paper_name + ".pdf"

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
