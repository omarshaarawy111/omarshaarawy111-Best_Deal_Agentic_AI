"""
Deal scanner agent for Best Deal.

Description:
    This module fetches recent deals, removes opportunities already present in
    runtime memory, and asks an OpenAI model to select structured bargains.

Responsibilities:
    - Scrape candidate deals from RSS feeds.
    - Avoid repeated deals by URL.
    - Select and validate up to five high-quality deals.
"""

from __future__ import annotations

from typing import Iterable

from openai import OpenAI

from best_deal.agents.agent import Agent
from best_deal.agents.deals import DealSelection, ScrapedDeal
from best_deal.config import SCANNER_MODEL


class ScannerAgent(Agent):
    """
    Find and structure promising online deals.

    Returns:
        A configured scanner agent.
    """

    name = "Scanner Agent"
    color = Agent.CYAN

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in structured JSON with no explanation. Select only deals whose actual price is clear and greater than zero. Focus the product_description on the product itself, not discount terms."""
    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
Rephrase the description as a concise product summary. Be careful with claims such as "$XXX off" because that may not be the actual product price.

Deals:

"""

    def __init__(self, client: OpenAI | None = None) -> None:
        """
        Initialize the scanner and OpenAI client.

        Args:
            client: Optional OpenAI client for dependency injection and testing.

        Returns:
            None.
        """
        self.openai = client or OpenAI()
        self.log("Scanner Agent is ready")

    def fetch_deals(self, memory: Iterable) -> list[ScrapedDeal]:
        """
        Fetch deals that have not already been processed.

        Args:
            memory: Previously stored opportunities.

        Returns:
            Newly scraped deals.
        """
        seen_urls = {opportunity.deal.url for opportunity in memory}
        scraped = ScrapedDeal.fetch()
        result = [deal for deal in scraped if deal.url not in seen_urls]
        self.log(f"Scanner Agent received {len(result)} unseen deals")
        return result

    def make_user_prompt(self, scraped: Iterable[ScrapedDeal]) -> str:
        """
        Create the scanner model's candidate prompt.

        Args:
            scraped: Scraped deal objects.

        Returns:
            User prompt containing all candidate deals.
        """
        return self.USER_PROMPT_PREFIX + "\n\n".join(deal.describe() for deal in scraped) + "\n\nInclude no more than 5 deals."

    def scan(self, memory: Iterable = ()) -> DealSelection | None:
        """
        Scrape the internet and select structured deals with OpenAI.

        Args:
            memory: Previously processed opportunities.

        Returns:
            A DealSelection object, or None when no unseen deals exist.
        """
        scraped = self.fetch_deals(memory)
        if not scraped:
            return None
        response = self.openai.chat.completions.parse(
            model=SCANNER_MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.make_user_prompt(scraped)},
            ],
            response_format=DealSelection,
            reasoning_effort="minimal",
        )
        selection = response.choices[0].message.parsed
        selection.deals = [deal for deal in selection.deals if deal.price > 0][:5]
        self.log(f"Scanner Agent selected {len(selection.deals)} deals")
        return selection

    def test_scan(self, memory: Iterable = ()) -> DealSelection:
        """
        Return deterministic scanner fixtures for local UI testing.

        Args:
            memory: Unused compatibility argument.

        Returns:
            A deterministic DealSelection fixture.
        """
        del memory
        results = {
            "deals": [
                {
                    "product_description": "The Hisense R6 Series 55R6030N is a 55-inch 4K UHD Roku Smart TV with 3840x2160 resolution, Dolby Vision HDR and HDR10 compatibility, Roku streaming, voice control, and three HDMI ports.",
                    "price": 178,
                    "url": "https://www.dealnews.com/products/Hisense/Hisense-R6-Series-55-R6030-N-55-4-K-UHD-Roku-Smart-TV/484824.html?iref=rss-c142",
                },
                {
                    "product_description": "The Poly Studio P21 is a 21.5-inch 1080p personal meeting display for remote work with a webcam, privacy shutter, stereo speakers, ambient light sensor, and wireless phone charging.",
                    "price": 30,
                    "url": "https://www.dealnews.com/products/Poly-Studio-P21-21-5-1080-p-LED-Personal-Meeting-Display/378335.html?iref=rss-c39",
                },
                {
                    "product_description": "The Lenovo IdeaPad Slim 5 laptop uses a 7th generation AMD Ryzen 5 8645HS 6-core CPU and includes a 16-inch touch display, 16GB RAM, and 512GB SSD.",
                    "price": 446,
                    "url": "https://www.dealnews.com/products/Lenovo/Lenovo-Idea-Pad-Slim-5-7-th-Gen-Ryzen-5-16-Touch-Laptop/485068.html?iref=rss-c39",
                },
                {
                    "product_description": "The Dell G15 gaming laptop uses an AMD Ryzen 5 7640HS CPU and combines a 15.6-inch 1080p 120Hz display with 16GB RAM, a 1TB NVMe SSD, and Nvidia GeForce RTX 3050 graphics.",
                    "price": 650,
                    "url": "https://www.dealnews.com/products/Dell/Dell-G15-Ryzen-5-15-6-Gaming-Laptop-w-Nvidia-RTX-3050/485067.html?iref=rss-c39",
                },
            ]
        }
        return DealSelection.model_validate(results)
