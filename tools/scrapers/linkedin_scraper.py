"""LinkedIn job scraper using Playwright.

WARNING: Scraping LinkedIn may violate their Terms of Service.
Use this only for personal, non-commercial purposes and at your own risk.
Consider using the official LinkedIn Jobs API for production use.
"""
import logging
from datetime import datetime
from typing import Optional

from .base_scraper import BaseScraper, RawJob

logger = logging.getLogger(__name__)


class LinkedInScraper(BaseScraper):
    """Scrapes LinkedIn Jobs (public search, no login required for basic listings)."""

    BASE_URL = "https://www.linkedin.com/jobs/search/"

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
            logger.warning("Playwright not installed. Skipping LinkedIn scrape.")
            return []
        except Exception as e:
            logger.error(f"LinkedIn scrape failed: {e}")
            return []

    def _scrape_with_playwright(
        self, role: str, location: str, max_results: int
    ) -> list[RawJob]:
        from playwright.sync_api import sync_playwright
        import urllib.parse

        query = urllib.parse.urlencode({
            "keywords": role,
            "location": location,
            "f_TPR": "r604800",  # past week
            "sortBy": "R",       # relevance
        })
        url = f"{self.BASE_URL}?{query}"
        jobs: list[RawJob] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=self.DEFAULT_HEADERS["User-Agent"],
                viewport={"width": 1280, "height": 800},
            )
            page = ctx.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(2000)

                # Scroll to load more results
                for _ in range(min(max_results // 10, 5)):
                    page.keyboard.press("End")
                    page.wait_for_timeout(1500)

                cards = page.query_selector_all("div.base-card")
                logger.info(f"LinkedIn: found {len(cards)} cards")

                for card in cards[:max_results]:
                    try:
                        job = self._parse_card(card, page)
                        if job:
                            jobs.append(job)
                    except Exception as e:
                        logger.debug(f"Card parse error: {e}")
                    self._polite_delay(0.5, 1.5)

            finally:
                browser.close()

        return jobs

    def _parse_card(self, card, page) -> Optional[RawJob]:
        title_el = card.query_selector("h3.base-search-card__title")
        company_el = card.query_selector("h4.base-search-card__subtitle")
        location_el = card.query_selector("span.job-search-card__location")
        link_el = card.query_selector("a.base-card__full-link")

        title = self._clean_text(title_el.inner_text() if title_el else "")
        company = self._clean_text(company_el.inner_text() if company_el else "")
        location = self._clean_text(location_el.inner_text() if location_el else "")
        url = link_el.get_attribute("href") if link_el else ""

        if not title or not url:
            return None

        # Strip tracking params from URL
        url = url.split("?")[0] if "?" in url else url

        # Fetch job description by clicking the card
        description = ""
        try:
            card.click()
            page.wait_for_timeout(1500)
            desc_el = page.query_selector("div.description__text")
            if desc_el:
                description = self._clean_text(desc_el.inner_text())
        except Exception:
            pass

        return RawJob(
            title=title,
            company=company,
            location=location,
            url=url,
            description=description,
            source="linkedin",
        )
