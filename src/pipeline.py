from src.knowledgebase import AISafetyKnowledgeBase
from src.collector import ArxivPaperCollector
from src.summarizer import PaperSummarizer
from src.zotero import ZoteroAdapter


def summarize_papers_pipeline(max_papers: int):
    knowledge_base = AISafetyKnowledgeBase()
    collector = ArxivPaperCollector()
    summarizer = PaperSummarizer()
    zotero_adapter = ZoteroAdapter()

    papers = collector.fetch_papers(query="AI safety", max_results=max_papers)
    for paper in papers:
        print(f"Processing paper: {paper['title']}")
        print(f"Paper entry ID: {paper['id']}")

        if knowledge_base.is_processed(paper["id"]):
            print("Already processed, skipping.")
            continue

        pdf_path = collector.download_pdf(paper)
        if not pdf_path:
            print("Failed to download PDF, skipping paper.")
            continue

        summary_result = summarizer.summarize_paper_full_text(paper, pdf_path)
        zotero_key = zotero_adapter.add_paper(
            paper, pdf_path, summary_result["summary"]
        )
        knowledge_base.store_paper(
            paper,
            summary_result["summary"],
            summary_result["method"],
            zotero_key=zotero_key,
        )
        print(f"Summary: {summary_result['summary']}")
        print("Paper processed and stored.")
        print("-" * 40)
