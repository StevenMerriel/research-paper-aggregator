from typing import TypedDict


class ResearchPaper(TypedDict):
    title: str
    authors: list[str]
    abstract: str
    publication_date: str
    journal: str
    doi: str
