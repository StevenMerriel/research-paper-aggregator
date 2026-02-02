from typing import TypedDict


class ResearchPaperContent(TypedDict):
    sections: list[str]
    figures: list[str]
    tables: list[str]
    abstract: str
    introduction: str
    methods: str
    results: str
    discussion: str
    conclusion: str
    references: list[str]


class ResearchPaper(TypedDict):
    title: str
    authors: list[str]
    abstract: str
    publication_date: str
    journal: str
    doi: str
    arxiv_id: str
    content: ResearchPaperContent
