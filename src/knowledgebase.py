import chromadb
from datetime import datetime
import hashlib
from typing import Dict, Optional


class AISafetyKnowledgeBase:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AISafetyKnowledgeBase, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        db_path: str = "./chroma_db",
    ):
        # Initialize ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        # Create or get papers collection
        self.papers_collection = self.chroma_client.get_or_create_collection(
            name="ai_safety_papers",
            metadata={"description": "Arxiv AI Safety Research Papers with summaries"},
        )

    def _paper_id_to_doc_id(self, paper_id: str) -> str:
        """Convert Arxiv paper ID to ChromaDB document ID"""
        return hashlib.md5(paper_id.encode()).hexdigest()

    def is_processed(self, paper_id: str) -> bool:
        """Check if paper exists in the database"""
        doc_id = self._paper_id_to_doc_id(paper_id)
        try:
            result = self.papers_collection.get(ids=[doc_id])
            return len(result["ids"]) > 0
        except Exception:
            return False

    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Retrieve a specific paper from the database"""
        doc_id = self._paper_id_to_doc_id(paper_id)
        try:
            result = self.papers_collection.get(ids=[doc_id])
            if len(result["ids"]) > 0:
                return {
                    "id": doc_id,
                    "summary": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            print(f"Error retrieving paper: {e}")
        return None

    def store_paper(
        self,
        paper: Dict,
        summary: str,
        summary_method: str,
        zotero_key: Optional[str] = None,
    ):
        """Store paper in ChromaDB"""
        doc_id = self._paper_id_to_doc_id(paper["id"])

        self.papers_collection.upsert(
            documents=[summary],
            ids=[doc_id],
            metadatas=[
                {
                    "paper_id": paper["id"],
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "authors": ", ".join(paper["authors"]),
                    "published": paper["published"],
                    "categories": ", ".join(paper["categories"]),
                    "pdf_url": paper["pdf_url"],
                    "abstract": paper["abstract"][:500],
                    "summary_method": summary_method,
                    "processed_at": datetime.now().isoformat(),
                    "zotero_key": zotero_key,
                }
            ],
        )

    def search_papers(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        date_from: Optional[str] = None,
    ) -> chromadb.QueryResult:
        """Semantic search over paper summaries"""
        where_clause = {}

        if category_filter:
            where_clause["categories"] = {"$contains": category_filter}
        if date_from:
            where_clause["published"] = {"$gte": date_from}

        kwargs = {"query_texts": [query], "n_results": n_results}
        if where_clause:
            kwargs["where"] = where_clause

        return self.papers_collection.query(**kwargs)

    def print_search_results(self, results: chromadb.QueryResult):
        """Pretty print search results"""
        if results["documents"] is None or len(results["documents"][0]) == 0:
            print("No results found.")
            return

        for i, (doc, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            print(f"\n{'='*80}")
            print(f"Result {i+1} (Similarity: {1 - distance:.3f})")
            print(f"{'='*80}")
            print(f"Title: {metadata['title']}")
            print(f"Authors: {metadata['authors']}")
            print(f"Published: {metadata['published']}")
            print(f"ArXiv: {metadata['arxiv_id']}")
            print(f"PDF: {metadata['pdf_url']}")
            print(f"Summary Method: {metadata.get('summary_method', 'unknown')}")
            print(f"\nSummary:\n{doc}")
