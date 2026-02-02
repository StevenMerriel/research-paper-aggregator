import typer

from src.structs import ResearchPaper


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


@app.command()
def generate_podcast_feed():
    """Generate podcast feed for all papers with audio."""
    from src.podcast import PodcastFeedGenerator
    from src.knowledgebase import AISafetyKnowledgeBase

    knowledge_base = AISafetyKnowledgeBase()

    podcast_generator = PodcastFeedGenerator(
        title="AI Safety Research Papers",
        link="http://localhost:8080/podcast",
        description="A podcast summarizing the latest AI safety research papers.",
        author_name="AI Research Bot",
        author_email="ai-research-bot@example.com",
    )
    count = 0
    for paper in sorted(knowledge_base.get_all_papers()["metadatas"], key=lambda x: x.get('processed_at', ''), reverse=True):
        print(f"  🎧  Adding episode for paper: {paper['title']}")
        research_paper = ResearchPaper(
            title=paper["title"],
            authors=paper["authors"].split(", "),
            abstract=paper.get("abstract", ""),
            publication_date=paper["published"],
            journal=paper.get("journal", ""),
            doi=paper.get("doi", ""),
            arxiv_id=paper["arxiv_id"],
        )
        # research_paper = ResearchPaper(**paper)
        print(f"  🎧  Parsed ResearchPaper: {research_paper}")
        print(f"  🎧  Generating episode for paper title: {research_paper['title']}")
        podcast_generator.add_episode(research_paper)
        count += 1
        if count >= 10:  # For testing, only generate one episode
            break

    podcast_generator.generate_feed(knowledge_base.get_all_papers())

@app.command()
def serve_podcast_feed():
    """Serve the podcast feed and audio files over HTTP."""
    from src.serve_podcast_feed import serve_audio

    serve_audio()

if __name__ == "__main__":
    app()
