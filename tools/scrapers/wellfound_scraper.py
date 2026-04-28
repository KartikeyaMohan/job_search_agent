"""Wellfound (formerly AngelList Talent) job scraper."""
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, RawJob

logger = logging.getLogger(__name__)


class WellfoundScraper(BaseScraper):
    """Scrapes Wellfound startup job listings."""

    BASE_URL = "https://wellfound.com/jobs"

    def search(
        self,
        role: str,
        location: str,
        skills: list[str],
        max_results: int = 25,
    ) -> list[RawJob]:
        try:
            return self._scrape_with_playwright(role, location, max_results)
        except ImportError:
            logger.warning("Playwright not installed — trying requests fallback.")
            return self._scrape_with_requests(role, location, max_results)
        except Exception as e:
            logger.error(f"Wellfound scrape failed: {e}")
            return []

    def _scrape_with_playwright(
        self, role: str, location: str, max_results: int
    ) -> list[RawJob]:
        from playwright.sync_api import sync_playwright
        import urllib.parse

        role_slug = role.lower().replace(" ", "-")
        url = f"{self.BASE_URL}?role={urllib.parse.quote(role)}&location={urllib.parse.quote(location)}"
        jobs: list[RawJob] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=self.DEFAULT_HEADERS["User-Agent"])
            page = ctx.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(3000)

                for _ in range(min(max_results // 8, 4)):
                    page.keyboard.press("End")
                    page.wait_for_timeout(2000)

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")
                jobs = self._parse_listings(soup, max_results)

            finally:
                browser.close()

        return jobs

    def _scrape_with_requests(
        self, role: str, location: str, max_results: int
    ) -> list[RawJob]:
        import urllib.parse
        url = f"{self.BASE_URL}?role={urllib.parse.quote(role)}"
        try:
            resp = requests.get(url, headers=self.DEFAULT_HEADERS, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            return self._parse_listings(soup, max_results)
        except Exception as e:
            logger.error(f"Wellfound requests fallback failed: {e}")
            return []

    def _parse_listings(self, soup: BeautifulSoup, max_results: int) -> list[RawJob]:
        jobs = []
        # Wellfound uses various selectors depending on layout version
        cards = soup.select("div[data-test='StartupResult']") or soup.select(".styles_component__")
        if not cards:
            # Generic fallback: look for job link patterns
            cards = soup.select("a[href*='/jobs/']")

        for card in cards[:max_results]:
            try:
                job = self._parse_card(card)
                if job:
                    jobs.append(job)
            except Exception as e:
                logger.debug(f"Wellfound card error: {e}")

        logger.info(f"Wellfound: found {len(jobs)} jobs")
        return jobs

    def _parse_card(self, card) -> Optional[RawJob]:
        title = self._clean_text(
            (card.select_one("h2") or card.select_one("h3") or card).get_text()
        )
        company_el = card.select_one("[class*='company'], [class*='startup']")
        company = self._clean_text(company_el.get_text() if company_el else "Unknown")
        location_el = card.select_one("[class*='location'], [class*='remote']")
        location = self._clean_text(location_el.get_text() if location_el else "Remote")

        link = card.get("href") or (card.select_one("a") or {}).get("href", "")
        if not link:
            return None
        if not link.startswith("http"):
            link = f"https://wellfound.com{link}"

        return RawJob(
            title=title or "Software Engineer",
            company=company,
            location=location,
            url=link,
            source="wellfound",
        )
