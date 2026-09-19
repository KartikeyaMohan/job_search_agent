"""Base scraper contract and shared helpers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import time
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class RawJob:
    title: str
    company: str
    location: str
    url: str
    description: str = ""
    requirements: str = ""
    job_type: Optional[str] = None
    salary_range: Optional[str] = None
    posted_date: Optional[datetime] = None
    source: str = "unknown"


class BaseScraper(ABC):
    """Abstract base for all job scrapers."""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    @abstractmethod
    def search(
        self,
        role: str,
        location: str,
        skills: list[str],
        max_results: int = 25,
        max_days: int = 7,
    ) -> list[RawJob]:
        """Return a list of raw job postings no older than max_days."""

    @staticmethod
    def _polite_delay(min_s: float = 1.5, max_s: float = 4.0) -> None:
        """Random delay to avoid rate limiting."""
        time.sleep(random.uniform(min_s, max_s))

    @staticmethod
    def _clean_text(text: str) -> str:
        """Collapse whitespace and strip."""
        import re
        text = re.sub(r"\s+", " ", text or "")
        return text.strip()
