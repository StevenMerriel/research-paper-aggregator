from pathlib import Path
from typing import Dict, Optional
from pyzotero import zotero
import os


class ZoteroAdapter:
    def __init__(
        self,
        library_id: Optional[str] = None,
        api_key: Optional[str] = None,
        library_type: Optional[str] = "user",
    ):
        # Initialize Zotero client
        self.client = None
        self.libray_id = library_id or os.environ.get("ZOTERO_LIBRARY_ID")
        print(f"Zotero Library ID: {self.libray_id}")

        self.api_key = api_key or os.environ.get("ZOTERO_API_KEY")
        print(f"Zotero API Key: {self.api_key}")
        print(f"Zotero Library ID: {self.libray_id}")

        if self.libray_id is None or self.api_key is None:
            print("Zotero credentials not provided, Zotero integration disabled.")
            # return
            raise ValueError("Zotero credentials not provided")

        self.client = zotero.Zotero(
            library_id=self.libray_id,
            library_type=library_type,
            api_key=self.api_key,
        )

    def is_already_added(self, arxiv_id: str) -> Optional[str]:
        """Check if paper already exists in Zotero, return item key if found"""
        if not self.client:
            return None

        try:
            # Search for items with this arXiv ID
            items = self.client.items(q=arxiv_id)

            for item in items:
                # Check if arXiv ID matches in extra field or URL
                if "data" in item:
                    extra = item["data"].get("extra", "")
                    url = item["data"].get("url", "")
                    if arxiv_id in extra or arxiv_id in url:
                        return item["key"]

            return None
        except Exception as e:
            print(f"Error checking Zotero: {e}")
            return None

    def add_paper(self, paper: Dict, pdf_path: Path, summary: str) -> Optional[str]:
        """
        Add paper to Zotero with PDF attachment
        Returns the Zotero item key if successful
        """
        if not self.client:
            print("Zotero client not initialized")
            return None

        # Check if already in Zotero
        existing_key = self.is_already_added(paper["arxiv_id"])
        if existing_key:
            print(f"  📚 Already in Zotero (key: {existing_key})")
            return existing_key

        try:
            print("Adding to Zotero...")

            # Create the item template
            template = self.client.item_template("journalArticle")

            # Fill in the metadata
            template["title"] = paper["title"]
            template["abstractNote"] = summary  # Use our comprehensive summary!
            template["date"] = paper["published"][:10]  # YYYY-MM-DD format
            template["url"] = paper["id"]
            template["publicationTitle"] = "arXiv"

            # Add arXiv ID and DOI to extra field
            extra_parts = [f"arXiv: {paper['arxiv_id']}"]
            if paper.get("doi"):
                extra_parts.append(f"DOI: {paper['doi']}")
            template["extra"] = "\n".join(extra_parts)

            # Add authors
            template["creators"] = [
                {"creatorType": "author", "name": author} for author in paper["authors"]
            ]

            # Add tags from categories
            template["tags"] = [{"tag": cat} for cat in paper["categories"]]

            # Create the item in Zotero
            created_item = self.client.create_items([template])

            if not created_item or "successful" not in created_item:
                print("Failed to create Zotero item")
                return None

            item_key = created_item["successful"]["0"]["key"]
            print("Created Zotero item (key: {item_key})")

            # Upload the PDF attachment
            if pdf_path and pdf_path.exists():
                print("Uploading PDF to Zotero...")
                try:
                    self.client.attachment_simple([str(pdf_path)], item_key)
                    print("PDF attached")
                except Exception as e:
                    print(f"PDF upload failed: {e}")

            return item_key

        except Exception as e:
            print(f"Error adding to Zotero: {e}")
            return None
