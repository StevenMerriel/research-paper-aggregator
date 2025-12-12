import typer


app = typer.Typer()


@app.command()
def summarize_new_papers(max_papers: int = 2):
    """Summarize new AI safety papers from arXiv and store them in the knowledge base."""
    from src.pipeline import summarize_papers_pipeline

    summarize_papers_pipeline(max_papers=max_papers)


@app.command()
def search_papers(query: str, n_results: int = 5):
    """Search for papers in the knowledge base matching the query."""
    from src.knowledgebase import AISafetyKnowledgeBase

    knowledge_base = AISafetyKnowledgeBase()
    results = knowledge_base.search_papers(query, n_results=n_results)
    knowledge_base.print_search_results(results)


if __name__ == "__main__":
    app()
