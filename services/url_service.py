# url_service.py

import logging
from typing import List

from crawl4ai import WebCrawler
from crawl4ai.chunking_strategy import NlpSentenceChunking
from crawl4ai.crawler_strategy import LocalSeleniumCrawlerStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy, NoExtractionStrategy
from langchain.schema import Document


def scrape_website(
    url: str,
    use_llm: bool = False,
) -> List[Document]:
    """Scrape a website using Crawl4AI with advanced strategies and return the data in LangChain Document format."""

    # Set the crawler strategy
    strategy = LocalSeleniumCrawlerStrategy(
        js_code=["console.log('Crawling started!');"]
    )

    # Set the chunking strategy
    chunking_strategy = NlpSentenceChunking()

    # Set the extraction strategy
    if use_llm:
        extraction_strategy = LLMExtractionStrategy(
            provider="openai",
            api_token="your_api_token",
            instruction="Extract relevant content.",
        )
    else:
        extraction_strategy = NoExtractionStrategy()

    # Initialize the WebCrawler
    crawler = WebCrawler(crawler_strategy=strategy)
    crawler.warmup()

    # Execute the crawl
    result = crawler.run(
        url=url,
        word_count_threshold=10,
        extraction_strategy=extraction_strategy,
        chunking_strategy=chunking_strategy,
        bypass_cache=True,
    )

    # Convert Crawl4AI results into LangChain Document format
    documents = [
        Document(
            page_content=page["content"],
            metadata={"url": page["url"], "title": page.get("title")},
        )
        for page in result.pages
    ]

    logging.info(f"Scraping complete, {len(documents)} documents loaded.")
    return documents
