from anthropic import Anthropic
import os
from pathlib import Path
import re
import pymupdf
import tiktoken
from typing import List, Dict, Optional

from src.structs import ResearchPaperContent


class PaperSummarizer:
    def __init__(
        self, anthropic_api_key: str | None = None, openai_api_key: str | None = None
    ):
        # Initialize LLM clients if API keys are provided
        self.client = self._initialize_client(anthropic_api_key, openai_api_key)

        # Initialize tokenizer for counting tokens
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def _initialize_client(
        self, anthropic_api_key: str | None = None, openai_api_key: str | None = None
    ):
        """Initialize LLM client if not already done"""
        self.anthropic_api_key = anthropic_api_key or os.environ.get(
            "ANTHROPIC_API_KEY"
        )
        if self.anthropic_api_key:
            return Anthropic(api_key=self.anthropic_api_key)

        # elif self.openai_api_key:
        #     self.client = OpenAI(api_key=self.openai_api_key)

    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = pymupdf.open(pdf_path)
            text = ""

            for page_num, page in enumerate(doc):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()

            doc.close()
            return text
        except Exception as e:
            print(f"  ❌ Error extracting text: {e}")
            return None

    def detect_sections(self, text: str) -> Dict[str, str]:
        """Try to detect paper sections (heuristic approach)"""
        sections = {}

        # Common section headers (case-insensitive)
        section_patterns = [
            (r"\n\s*(?:1\.?\s+)?(?:INTRODUCTION|Introduction)\s*\n", "introduction"),
            (
                r"\n\s*(?:\d+\.?\s+)?(?:RELATED WORK|Related Work|Background)\s*\n",
                "related_work",
            ),
            (
                r"\n\s*(?:\d+\.?\s+)?(?:METHODOLOGY|Methodology|Methods|Approach)\s*\n",
                "methodology",
            ),
            (
                r"\n\s*(?:\d+\.?\s+)?(?:EXPERIMENTS?|Experiments?|Results|Evaluation)\s*\n",
                "results",
            ),
            (r"\n\s*(?:\d+\.?\s+)?(?:DISCUSSION|Discussion)\s*\n", "discussion"),
            (
                r"\n\s*(?:\d+\.?\s+)?(?:CONCLUSION|Conclusion|Conclusions)\s*\n",
                "conclusion",
            ),
            (r"\n\s*(?:REFERENCES|References)\s*\n", "references"),
        ]

        # Find all section positions
        positions = []
        for pattern, name in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                positions.append((matches[0].start(), name))

        # Sort by position
        positions.sort()

        # Extract sections
        for i, (start_pos, name) in enumerate(positions):
            if i < len(positions) - 1:
                end_pos = positions[i + 1][0]
                sections[name] = text[start_pos:end_pos].strip()
            else:
                sections[name] = text[start_pos:].strip()

        return sections

    def chunk_text(self, text: str, max_tokens: int = 15000) -> List[str]:
        """Split text into chunks that fit within token limit"""
        chunks = []
        current_chunk = ""

        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if self._count_tokens(potential_chunk) > max_tokens:
                if current_chunk:  # Save current chunk
                    chunks.append(current_chunk)
                    current_chunk = para
                else:  # Single paragraph is too long, split it
                    # Split by sentences
                    sentences = para.split(". ")
                    for sentence in sentences:
                        if self._count_tokens(current_chunk + sentence) > max_tokens:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            current_chunk += sentence + ". "
            else:
                current_chunk = potential_chunk

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def summarize_text_chunk(
        self, chunk: str, chunk_num: int, total_chunks: int, paper_title: str
    ) -> str:
        """Summarize a single chunk of text"""
        if not self.client:
            return chunk[:500]

        prompt = f"""You are summarizing part {chunk_num} of {total_chunks} from the paper: "{paper_title}"

Extract and summarize the key information from this section:

{chunk}

Focus on:
- Main findings and claims
- Methodology details
- Results and evidence
- Important context

Be concise but comprehensive. If this section contains references or acknowledgments, just note that briefly."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            print(f"  ❌ Error summarizing chunk {chunk_num}: {e}")
            return chunk[:500]

    def create_final_summary(self, paper: Dict, section_summaries: List[str]) -> str:
        """Create final comprehensive summary from section summaries"""
        if not self.client:
            return "\n\n".join(section_summaries)

        combined = "\n\n---\n\n".join(
            [
                f"Section {i+1}:\n{summary}"
                for i, summary in enumerate(section_summaries)
            ]
        )

        prompt = f"""You are creating a final comprehensive summary for the paper: "{paper['title']}"
By: {', '.join(paper['authors'])}

Here are summaries of different sections of the paper:

{combined}

Create a cohesive, comprehensive summary (3-5 paragraphs) that covers:
1. The research problem and motivation
2. The methodology and approach
3. Key findings and results
4. Significance and contributions
5. Limitations or future work (if mentioned)

Write as a unified narrative, not as separate sections."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as e:
            print(f"  ❌ Error creating final summary: {e}")
            return "\n\n".join(section_summaries)

    def summarize_paper_full_text(self, paper: Dict, pdf_path: Path) -> Dict[str, str]:
        """
        Download PDF, extract text, and create comprehensive summary
        Returns dict with 'summary' and 'method' keys
        """
        # Download PDF
        if not pdf_path:
            print("Falling back to abstract only")
            return {"summary": paper["abstract"], "method": "abstract_only"}

        # Extract text
        full_text = self.extract_text_from_pdf(pdf_path)
        if not full_text:
            print("Falling back to abstract only")
            return {"summary": paper["abstract"], "method": "abstract_only"}

        print(f"  📄 Extracted {len(full_text)} characters")

        # Detect sections
        sections = self.detect_sections(full_text)
        print(f"  🗂️ Detected sections: {list(sections.keys())}")

        content: ResearchPaperContent = {
            "sections": list(sections.keys()),
            "figures": [],  # Placeholder, implement figure extraction if needed
            "tables": [],   # Placeholder, implement table extraction if needed
            "abstract": sections.get("abstract", ""),
            "introduction": sections.get("introduction", ""),
            "methods": sections.get("methods", ""),
            "results": sections.get("results", ""),
            "discussion": sections.get("discussion", ""),
            "conclusion": sections.get("conclusion", ""),
            "references": sections.get("references", []),
        }

        # Remove references section to save tokens
        text_no_refs = re.split(
            r"\n\s*(?:REFERENCES|References)\s*\n", full_text, maxsplit=1
        )[0]

        token_count = self._count_tokens(text_no_refs)
        print(f"  🔢 Token count: {token_count:,}")

        # If short enough, summarize in one go
        if token_count < 50000:  # Well within Claude's context window
            print("Creating summary in single pass")
            if not self.client:
                return {"summary": text_no_refs, "method": "abstract_only", "content": content}

            prompt = f"""Analyze and summarize this research paper comprehensively.

Title: {paper['title']}
Authors: {', '.join(paper['authors'])}

Full Paper Text:
{text_no_refs}

Provide a comprehensive summary (3-5 paragraphs) covering:
1. Research problem and motivation
2. Methodology and approach
3. Key findings and results
4. Significance and contributions
5. Limitations or future work

Focus on the actual content and findings, not just the abstract."""

            try:
                message = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=3000,
                    messages=[{"role": "user", "content": prompt}],
                )
                return {
                    "summary": message.content[0].text,
                    "method": "full_text_single_pass",
                    "content": content,
                }
            except Exception as e:
                print(f"  ❌ Error in single pass: {e}")

        # For longer papers, use chunking approach
        print("Paper is long, using hierarchical summarization")
        chunks = self.chunk_text(text_no_refs, max_tokens=15000)
        print(f"Split into {len(chunks)} chunks")

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  🔄 Summarizing chunk {i+1}/{len(chunks)}...")
            summary = self.summarize_text_chunk(
                chunk, i + 1, len(chunks), paper["title"]
            )
            chunk_summaries.append(summary)

        # Create final summary
        print("Creating final summary from chunks...")
        final_summary = self.create_final_summary(paper, chunk_summaries)

        return {
            "summary": final_summary,
            "method": f"hierarchical_{len(chunks)}_chunks",
            "content": content,
        }
