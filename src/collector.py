import arxiv
from pathlib import Path
from typing import Generator, Dict, Optional


class ArxivPaperCollector:

    def __init__(self, pdf_cache_dir: str = "./pdf_cache"):
        # PDF cache directory
        self.pdf_cache_dir = Path(pdf_cache_dir)
        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_papers(
        self, query: str, max_results: int = 10
    ) -> Generator[Dict, None, None]:
        """Generator that fetches papers from Arxiv"""
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        for paper in search.results():
            yield {
                "id": paper.entry_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "published": paper.published.isoformat(),
                "categories": paper.categories,
                "pdf_url": paper.pdf_url,
                "arxiv_id": paper.get_short_id(),
                "paper_object": paper,
            }

    def download_pdf(self, paper: Dict) -> Optional[Path]:
        """Download PDF to cache directory"""
        arxiv_id = paper["arxiv_id"]
        pdf_path = self.pdf_cache_dir / f"{arxiv_id}.pdf"

        # Check if already downloaded
        if pdf_path.exists():
            print("Using cached PDF")
            return pdf_path

        try:
            print("Downloading PDF...")
            paper["paper_object"].download_pdf(
                dirpath=self.pdf_cache_dir, filename=f"{arxiv_id}.pdf"
            )
            return pdf_path
        except Exception as e:
            print(f"  ❌ Error downloading PDF: {e}")
            return None
