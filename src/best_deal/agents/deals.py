"""
Deal discovery models and RSS scraping for Best Deal.

Description:
    This module contains the external deal feed scraper and the validated
    domain models used by the scanner and planner.

Responsibilities:
    - Fetch recent deal entries from DealNews RSS feeds.
    - Extract usable product descriptions from deal pages.
    - Represent deals and priced opportunities with Pydantic models.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, ClassVar, Self

import feedparser
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

FEEDS = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://www.dealnews.com/c238/Automotive/?rss=1",
    "https://www.dealnews.com/c196/Home-Garden/?rss=1",
]


class ScrapedDeal:
    """
    Represent a raw deal scraped from an RSS feed and deal page.

    Returns:
        A parsed deal object containing title, description, and page details.
    """

    timeout_seconds: ClassVar[int] = 20

    def __init__(self, entry: dict[str, Any]) -> None:
        """
        Build a scraped deal from one feed entry.

        Args:
            entry: FeedParser entry mapping.

        Returns:
            None.
        """
        self.title = str(entry.get("title", "")).strip()
        self.summary = self.extract(str(entry.get("summary", "")))
        links = entry.get("links") or []
        self.url = links[0].get("href", "") if links else str(entry.get("link", ""))
        response = requests.get(self.url, timeout=self.timeout_seconds)
        response.raise_for_status()
        content = self._extract_content(response.text)
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details, self.features = content, ""
        self.category = ""
        self.truncate()

    @staticmethod
    def extract(html_snippet: str) -> str:
        """
        Extract readable deal text from an RSS HTML snippet.

        Args:
            html_snippet: HTML fragment from the RSS summary.

        Returns:
            Cleaned summary text.
        """
        soup = BeautifulSoup(html_snippet, "html.parser")
        snippet_div = soup.find("div", class_="snippet summary")
        description = snippet_div.get_text(strip=True) if snippet_div else soup.get_text(" ", strip=True)
        return re.sub(r"<[^<]+?>", "", description).replace("\n", " ").strip()

    @staticmethod
    def _extract_content(html: str) -> str:
        """
        Extract the main content section from a deal page.

        Args:
            html: Deal page HTML.

        Returns:
            Main textual content.
        """
        soup = BeautifulSoup(html, "html.parser")
        content = soup.find("div", class_="content-section")
        if content is None:
            return soup.get_text(" ", strip=True)
        return content.get_text(" ", strip=True).replace("more", " ")

    def truncate(self) -> None:
        """
        Limit scraped text to a safe prompt size.

        Returns:
            None.
        """
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def describe(self) -> str:
        """
        Convert the scraped deal into a compact model input string.

        Returns:
            Text representation of the deal.
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    def __repr__(self) -> str:
        """
        Return a compact representation for debugging.

        Returns:
            A short title representation.
        """
        return f"<{self.title}>"

    @classmethod
    def fetch(cls, feeds: list[str] | None = None, show_progress: bool = False, max_workers: int = 10) -> list[Self]:
        """
        Fetch and parse recent deals concurrently from RSS feeds.

        Args:
            feeds: RSS feed URLs. Defaults to configured DealNews feeds.
            show_progress: Reserved for compatibility with the notebook flow.
            max_workers: Maximum concurrent page requests.

        Returns:
            Successfully scraped deal objects.
        """
        del show_progress
        source_feeds = feeds or FEEDS
        entries: list[dict[str, Any]] = []
        for feed_url in source_feeds:
            feed = feedparser.parse(feed_url)
            entries.extend(feed.entries[:10])
        deals: list[Self] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(cls, entry) for entry in entries]
            for future in as_completed(futures):
                try:
                    deals.append(future.result())
                except Exception as exc:
                    logging.getLogger().warning("Skipping deal entry: %s", exc)
        return deals


class Deal(BaseModel):
    """
    Represent a structured deal selected by the scanner.

    Returns:
        A validated deal domain object.
    """

    product_description: str = Field(description="Concise product-focused summary.")
    price: float = Field(description="Current advertised deal price in USD.")
    url: str = Field(description="URL of the deal.")


class DealSelection(BaseModel):
    """
    Represent the scanner's selected deal set.

    Returns:
        A validated list of selected deals.
    """

    deals: list[Deal] = Field(description="Selected high-quality deals.")


class Opportunity(BaseModel):
    """
    Represent a deal after fair-value estimation.

    Returns:
        A validated opportunity with its price gap.
    """

    deal: Deal
    estimate: float
    discount: float
