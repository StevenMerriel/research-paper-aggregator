from datetime import datetime
import os
from feedgen.feed import FeedGenerator
from openai import OpenAI
from pathlib import Path
from typing import Optional
from anthropic import Anthropic

from src.structs import ResearchPaper


class PodcastFeedGenerator:
    def __init__(self, title: str, link: str, description: str, author_name: str, author_email: str, audio_model: str = "tts-1-hd", tts_voice: str = "alloy", tts_speed: float = 1.0, script_model: str = "claude-sonnet-4-5"):
        self.feed_title = title
        self.feed_url = link
        self.feed_description = description
        self.episodes = []
        self.author_name = author_name
        self.author_email = author_email
        self.audio_dir = Path("podcast_audio")
        self.audio_dir.mkdir(exist_ok=True)
        self.audio_model = audio_model
        self.tts_voice = tts_voice
        self.tts_speed = tts_speed
        self.script_model = script_model
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None

    def add_episode(self, paper: ResearchPaper):
        if self._check_if_audio_exists(paper['arxiv_id']):
            print(f"  🔊 Audio already exists for paper {paper['arxiv_id']}, skipping generation")
            return

        episode_title = paper['title']
        episode_link = f"{self.feed_url}/{paper['arxiv_id'].replace('/', '_')}.mp3"
        episode_description = paper['abstract'][:500] + "..."  # Truncate for description
        pub_date = datetime.fromisoformat(paper['publication_date'])

        audio_path = self.generate_audio(paper)
        if not audio_path:
            print(f"  ⚠️  Skipping episode generation for paper {paper['arxiv_id']} due to audio generation failure")
            return
        
        episode = {
            'title': episode_title,
            'link': episode_link,
            'description': episode_description,
            'pub_date': pub_date
        }
        self.episodes.append(episode)

    def _check_if_audio_exists(self, arxiv_id: str) -> bool:
        audio_filename = f"{arxiv_id.replace('/', '_')}.mp3"
        audio_path = self.audio_dir / audio_filename
        return audio_path.exists()

    def generate_feed(self, papers: dict) -> Path:
        """
        Generate RSS podcast feed for all papers with audio
        """
        print(f"\n🎙️  Generating podcast feed...")
        
        # Create feed
        fg = FeedGenerator()
        fg.load_extension('podcast')
        
        # Set feed metadata
        feed_link = self.feed_url or f"{self.audio_base_url}/feed.xml"
        fg.id(feed_link)
        fg.title(self.feed_title)
        fg.description(self.feed_description)
        fg.author({'name': self.author_name, 'email': self.author_email})
        fg.link(href=feed_link, rel='self')
        fg.language('en')
        fg.podcast.itunes_category('Science')
        fg.podcast.itunes_explicit('no')
        
        # Get all papers from database
        all_papers = papers
        
        # Sort by processed date (newest first)
        papers_with_dates = [
            (i, metadata.get('processed_at', ''))
            for i, metadata in enumerate(all_papers['metadatas'])
        ]
        papers_with_dates.sort(key=lambda x: x[1], reverse=True)
        
        # Add episodes
        episodes_added = 0
        for idx, _ in papers_with_dates:
            metadata = all_papers['metadatas'][idx]
            summary = all_papers['documents'][idx]
            
            arxiv_id = metadata.get('arxiv_id', '')
            if not arxiv_id:
                continue
            
            # Check if audio file exists
            audio_filename = f"{arxiv_id.replace('/', '_')}.mp3"
            audio_path = self.audio_dir / audio_filename
            
            if not audio_path.exists():
                print(f"  ⚠️  Audio file not found for paper {arxiv_id}, skipping...")
                continue

            # Create feed entry
            fe = fg.add_entry()
            fe.id(metadata.get('paper_id', ''))
            fe.title(metadata.get('title', 'Untitled Paper'))
            fe.description(summary[:500] + '...')  # Truncate for description
            
            # Audio enclosure
            audio_url = f"{self.feed_url}/{audio_filename}"
            audio_size = audio_path.stat().st_size
            fe.enclosure(audio_url, str(audio_size), 'audio/mpeg')
            
            # Episode metadata
            published_date = datetime.fromisoformat(metadata.get('published', datetime.now().isoformat()))
            fe.published(published_date)
            fe.podcast.itunes_summary(summary[:4000])  # iTunes limit
            
            # Authors and categories
            authors = metadata.get('authors', 'Unknown')
            fe.podcast.itunes_author(authors)
            
            episodes_added += 1

            # BREAK after adding 1 episode for demo purposes
            break
        
        # Write feed to file
        feed_path = self.audio_dir / 'feed.xml'
        fg.rss_file(str(feed_path))
        
        print(f"✅ Podcast feed generated: {feed_path}")
        print(f"   Episodes: {episodes_added}")
        print(f"   Feed URL: {feed_link}")
        
        return feed_path

    def generate_audio(self, paper: ResearchPaper) -> Optional[Path]:
        """
        Generate audio file from summary using OpenAI TTS
        Returns path to audio file
        """
        if not self.openai:
            print("⚠️  OpenAI API not configured, skipping audio generation")
            return None
        
        # Check if audio already exists
        arxiv_id = paper['arxiv_id']
        audio_filename = f"{arxiv_id.replace('/', '_')}.mp3"
        audio_path = self.audio_dir / audio_filename
        
        if audio_path.exists():
            print("  🔊 Using existing audio file")
            return audio_path
        
        try:
            print(f"  🎙️  Generating audio (voice: {self.tts_voice}, speed: {self.tts_speed})...")
            
            # Prepare script
            script = self.prepare_audio_script(paper)
            
            # Generate audio
            response = self.openai.audio.speech.create(
                model=self.audio_model,  # Use HD model for better quality
                voice=self.tts_voice,
                input=script,
                speed=self.tts_speed
            )
            
            # Save to file
            response.stream_to_file(audio_path)
            
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Audio generated ({file_size_mb:.2f} MB)")
            
            return audio_path
            
        except Exception as e:
            print(f"  ❌ Error generating audio: {e}")
            return None

    def prepare_audio_script(self, paper: ResearchPaper) -> str:
        """
        Generate a podcast script based on the paper details and summary.
        Adds intro and makes it more conversational.
        """
        first_author = paper['authors'][0].split()[0] if paper['authors'] else "Unknown Author"

        if len(paper['authors']) == 1:
            text_authors = paper['authors'][0]
        if len(paper['authors']) == 2:
            text_authors = f"{paper['authors'][0]} and {paper['authors'][1]}"
        else:
            text_authors = f"{first_author} et al."



        prompt = f"""
        You are an expert podcast scriptwriter. Given the following paper details and summary, create an engaging podcast script.

        Host Name: Steven Merriel
        Paper Title: {paper['title']}
        Authors: {text_authors}
        Abstract: {paper['abstract']}
        Summary: {paper.get('content', {}).get('summary', '')}
        Research Context: {paper.get('content', {}).get('introduction', '')}
        Methods: {paper.get('content', {}).get('methods', '')}
        Results: {paper.get('content', {}).get('results', '')}
        Discussion: {paper.get('content', {}).get('discussion', '')}
        Conclusion: {paper.get('content', {}).get('conclusion', '')}

        Write a podcast script that introduces the paper, discusses its significance, discusses each section of the paper,and summarizes the key points in an engaging manner. Make it conversational and accessible to a technical audience interested in AI safety research. Aim for a script length of about 3000 to 4000 words.
        """
        if self.script_model.startswith("gpt-"):
            openai_client = self.openai
            if not openai_client:
                raise ValueError("OpenAI API key not configured.")
            response = openai_client.chat.completions.create(
                model=self.script_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that writes podcast scripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=14000,
                temperature=0.7
            )
            script = response.choices[0].message.content.strip()

        else:
            client = self.anthropic
            if not client:
                raise ValueError("Anthropic API key not configured.")
            response = client.messages.create(
                model=self.script_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that writes podcast scripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=14000,
                temperature=0.2
            )
            script = response['content'][0]["text"].strip()

        with open("./podcast_audio/" + paper['arxiv_id'].replace('/', '_') + "_script.txt", "w") as script_file:
            script_file.write(script)

        return script