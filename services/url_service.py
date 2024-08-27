# -*- coding: utf-8 -*-

# url_service.py

import logging
import re
import subprocess
import sys
import time
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import numpy as np
from bs4 import BeautifulSoup, Comment
from langchain.schema import Document
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from sklearn.feature_extraction.text import TfidfVectorizer


def ensure_playwright_installed():
    """Ensure Playwright and its browsers are installed."""
    try:
        import playwright
    except ImportError:
        logging.info("Playwright not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
        import playwright

    try:
        # Check if browsers are installed
        with sync_playwright() as p:
            p.chromium.launch()
    except Exception:
        logging.info("Installing Playwright browsers...")
        subprocess.check_call(
            [sys.executable, "-m", "playwright", "install", "chromium"]
        )


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.delay = 60.0 / requests_per_minute
        self.last_request = 0

    def wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()


class ContentCleaner:
    def __init__(self):
        # Define common class names for ads and navigation elements
        self.common_ad_classes = ["ad", "advertisement", "banner", "sponsored"]
        self.common_nav_classes = ["nav", "navigation", "menu", "sidebar"]
        # Define tags that are typically irrelevant to main content
        self.irrelevant_tags = [
            "script",
            "style",
            "noscript",
            "iframe",
            "header",
            "footer",
        ]
        # Initialize TF-IDF vectorizer for text analysis
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def clean(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Main cleaning method that applies all cleaning techniques."""
        # Apply basic cleaning techniques
        self.remove_comments(soup)
        self.remove_irrelevant_tags(soup)
        self.remove_ads(soup)
        self.remove_navigation(soup)
        self.remove_empty_tags(soup)
        self.clean_text(soup)

        # Find the main content wrapper
        main_content = self.find_content_wrapper(soup)
        if main_content:
            soup = BeautifulSoup(str(main_content), "html.parser")

        # Extract paragraphs and remove boilerplate content
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        cleaned_paragraphs = self.remove_boilerplate(paragraphs)

        # Reconstruct the soup with cleaned paragraphs
        new_soup = BeautifulSoup("<div></div>", "html.parser")
        for para in cleaned_paragraphs:
            new_p = new_soup.new_tag("p")
            new_p.string = para
            new_soup.div.append(new_p)

        return new_soup

    def remove_comments(self, soup: BeautifulSoup):
        """Remove HTML comments from the soup."""
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()

    def remove_irrelevant_tags(self, soup: BeautifulSoup):
        """Remove irrelevant tags from the soup."""
        for tag in self.irrelevant_tags:
            for element in soup.find_all(tag):
                element.decompose()

    def remove_ads(self, soup: BeautifulSoup):
        """Remove elements with common ad class names from the soup."""
        for ad_class in self.common_ad_classes:
            for element in soup.find_all(class_=re.compile(ad_class, re.I)):
                element.decompose()

    def remove_navigation(self, soup: BeautifulSoup):
        """Remove elements with common navigation class names from the soup."""
        for nav_class in self.common_nav_classes:
            for element in soup.find_all(class_=re.compile(nav_class, re.I)):
                element.decompose()

    def remove_empty_tags(self, soup: BeautifulSoup):
        """Remove empty tags from the soup."""
        for element in soup.find_all():
            if len(element.get_text(strip=True)) == 0:
                element.extract()

    def clean_text(self, soup: BeautifulSoup):
        """Normalize whitespace in text nodes."""
        for element in soup.find_all(text=True):
            text = element.strip()
            if text:
                cleaned_text = re.sub(r"\s+", " ", text)
                element.replace_with(cleaned_text)

    def remove_boilerplate(self, paragraphs: List[str]) -> List[str]:
        """Remove boilerplate content using TF-IDF analysis."""
        if not paragraphs:
            return []

        # Calculate TF-IDF scores for each paragraph
        tfidf_matrix = self.vectorizer.fit_transform(paragraphs)

        # Calculate the average TF-IDF score for each paragraph
        avg_scores = np.mean(tfidf_matrix.toarray(), axis=1)

        # Keep paragraphs with above-average TF-IDF scores
        threshold = np.mean(avg_scores)
        return [
            para for para, score in zip(paragraphs, avg_scores) if score > threshold
        ]

    def score_readability(self, text):
        """Calculate a basic readability score for a given text."""
        words = len(re.findall(r"\w+", text))
        sentences = len(re.findall(r"\w+[.!?]", text)) or 1
        return words / sentences  # A simple words per sentence ratio

    def find_content_wrapper(self, soup):
        """Identify the main content wrapper based on readability and structure."""
        candidates = soup.find_all(["div", "article", "main", "section"])
        if not candidates:
            return None

        scored_candidates = []
        for candidate in candidates:
            text = candidate.get_text()
            score = self.score_readability(text)
            text_length = len(text)
            p_density = len(candidate.find_all("p")) / (len(candidate.find_all()) + 1)

            combined_score = score * text_length * p_density
            scored_candidates.append((candidate, combined_score))

        return max(scored_candidates, key=lambda x: x[1])[0]


class WebScraper:
    def __init__(self, requests_per_minute: int = 20):
        ensure_playwright_installed()
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.robot_parsers = {}
        self.content_cleaner = ContentCleaner()
        logging.info(
            f"WebScraper initialized with {requests_per_minute} requests per minute"
        )

    def can_fetch(self, url: str) -> bool:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        if base_url not in self.robot_parsers:
            logging.info(f"Fetching robots.txt for {base_url}")
            rp = RobotFileParser()
            rp.set_url(urljoin(base_url, "/robots.txt"))
            rp.read()
            self.robot_parsers[base_url] = rp

        can_fetch = self.robot_parsers[base_url].can_fetch("*", url)
        logging.info(f"Can fetch {url}: {can_fetch}")
        return can_fetch

    def scrape_website(self, url: str, timeout: int = 30000) -> List[Document]:
        logging.info(f"Starting to scrape website: {url}")
        if not self.can_fetch(url):
            logging.warning(f"Scraping not allowed for {url} according to robots.txt")
            return []

        self.rate_limiter.wait()
        logging.info(f"Rate limiter delay applied for {url}")

        try:
            with sync_playwright() as p:
                logging.info("Launching browser")
                browser = p.chromium.launch()
                context = browser.new_context(
                    user_agent="YourBot/1.0 (+https://yourwebsite.com/bot)"
                )
                page = context.new_page()

                try:
                    logging.info(f"Navigating to {url}")
                    page.goto(url, timeout=timeout)
                    logging.info("Waiting for network idle")
                    page.wait_for_load_state("networkidle", timeout=timeout)
                except PlaywrightTimeoutError:
                    logging.warning(f"Timeout occurred while loading {url}")

                logging.info("Extracting page content")
                content = page.content()
                browser.close()
                logging.info("Browser closed")
        except Exception as e:
            logging.error(f"Error occurred while scraping {url}: {str(e)}")
            return []

        logging.info("Extracting and processing content")
        return self.extract_content(content, url)

    def extract_content(self, html_content: str, url: str) -> List[Document]:
        logging.info("Cleaning and parsing HTML content")
        soup = BeautifulSoup(html_content, "html.parser")
        cleaned_soup = self.content_cleaner.clean(soup)

        logging.info("Extracting main content")
        main_content = self.extract_main_content(cleaned_soup)

        logging.info("Extracting sections")
        sections = self.extract_sections(cleaned_soup)

        documents = []

        logging.info("Creating main content document")
        documents.append(
            Document(
                page_content=main_content,
                metadata={
                    "url": url,
                    "title": self.extract_title(soup),
                    "type": "main_content",
                },
            )
        )

        logging.info(f"Creating {len(sections)} section documents")
        for section in sections:
            documents.append(
                Document(
                    page_content=section["content"],
                    metadata={"url": url, "title": section["title"], "type": "section"},
                )
            )

        logging.info(f"Extracted {len(documents)} documents in total")
        return documents

    def extract_main_content(self, soup: BeautifulSoup) -> str:
        logging.info("Extracting main content from paragraphs")
        content = " ".join(
            p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
        )
        logging.info(f"Extracted {len(content)} characters of main content")
        return content

    def extract_title(self, soup: BeautifulSoup) -> str:
        title = soup.title.string if soup.title else ""
        logging.info(f"Extracted title: {title}")
        return title

    def extract_sections(self, soup: BeautifulSoup) -> List[dict]:
        logging.info("Extracting sections from headers")
        sections = []
        for header in soup.find_all(["h1", "h2", "h3"]):
            content = []
            for sibling in header.find_next_siblings():
                if sibling.name in ["h1", "h2", "h3"]:
                    break
                if sibling.name in ["p", "ul", "ol"]:
                    content.append(sibling.text)
            if content:
                sections.append({"title": header.text, "content": " ".join(content)})
        logging.info(f"Extracted {len(sections)} sections")
        return sections
