"""Generic careers page scraper — works on most company career portals."""
import logging
import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, RawJob

logger = logging.getLogger(__name__)

# Common job-related CSS selectors and patterns
_JOB_LINK_PATTERNS = re.compile(
    r"/(jobs?|careers?|openings?|positions?|roles?|opportunities|apply)/",
    re.IGNORECASE,
)
_SALARY_PATTERN = re.compile(
    r"\$[\d,]+(?:\s*[-–]\s*\$[\d,]+)?(?:\s*/\s*(?:yr|year|mo|month|hr|hour))?",
    re.IGNORECASE,
)


class GenericCareersScraper(BaseScraper):
    """Scrapes job listings from any careers page URL."""

    def search(
        self,
        role: str,
        location: str,
        skills: list[str],
        max_results: int = 25,
    ) -> list[RawJob]:
        # Generic scraper requires explicit URLs — used programmatically
        return []

    def scrape_url(self, careers_url: str, role_filter: str = "", max_results: int = 20) -> list[RawJob]:
        """Scrape a company careers page and return matching jobs."""
        try:
            html = self._fetch(careers_url)
            if not html:
                return []
            soup = BeautifulSoup(html, "html.parser")
            job_links = self._find_job_links(soup, careers_url)

            if not job_links:
                # Try to parse jobs directly from the listing page
                return self._parse_inline_listings(soup, careers_url, role_filter, max_results)

            jobs = []
            for link in job_links[:max_results]:
                if role_filter and role_filter.lower() not in link.lower():
                    # Quick title filter from URL
                    pass  # Still visit — title in URL is unreliable
                try:
                    job = self._scrape_job_page(link)
                    if job:
                        if not role_filter or self._matches_role(job.title, role_filter):
                            jobs.append(job)
                    self._polite_delay(1.0, 2.5)
                except Exception as e:
                    logger.debug(f"Job page error {link}: {e}")

            logger.info(f"Generic scraper: found {len(jobs)} jobs from {careers_url}")
            return jobs

        except Exception as e:
            logger.error(f"Generic scrape failed for {careers_url}: {e}")
            return []

    def _fetch(self, url: str, timeout: int = 15) -> Optional[str]:
        try:
            resp = requests.get(url, headers=self.DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"Fetch failed for {url}: {e}")
            return None

    def _find_job_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href)
            if _JOB_LINK_PATTERNS.search(full_url) and urlparse(full_url).netloc == urlparse(base_url).netloc:
                links.add(full_url)
        return list(links)

    def _scrape_job_page(self, url: str) -> Optional[RawJob]:
        html = self._fetch(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = ""
        for selector in ["h1", "h2.job-title", "[class*='job-title']", "[class*='position-title']"]:
            el = soup.select_one(selector)
            if el:
                title = self._clean_text(el.get_text())
                break

        if not title:
            title = soup.title.string if soup.title else "Unknown Role"
            title = self._clean_text(title)

        # Extract company from domain
        domain = urlparse(url).netloc
        company = domain.replace("www.", "").split(".")[0].title()

        # Extract location
        location = "Remote"
        for selector in ["[class*='location']", "[data-location]", "[class*='city']"]:
            el = soup.select_one(selector)
            if el:
                location = self._clean_text(el.get_text())
                break

        # Extract description (largest text block)
        description = ""
        for selector in [
            "[class*='job-description']", "[class*='description']",
            "[class*='job-details']", "article", "main",
        ]:
            el = soup.select_one(selector)
            if el and len(el.get_text()) > 200:
                description = self._clean_text(el.get_text())
                break

        if not description:
            body = soup.find("body")
            description = self._clean_text(body.get_text()) if body else ""

        # Extract salary if mentioned
        salary_match = _SALARY_PATTERN.search(description)
        salary = salary_match.group(0) if salary_match else None

        return RawJob(
            title=title,
            company=company,
            location=location,
            url=url,
            description=description[:4000],  # cap length
            salary_range=salary,
            source="careers_page",
        )

    def _parse_inline_listings(
        self, soup: BeautifulSoup, base_url: str, role_filter: str, max_results: int
    ) -> list[RawJob]:
        """Fallback: parse job cards directly from a listing page."""
        jobs = []
        domain = urlparse(base_url).netloc
        company = domain.replace("www.", "").split(".")[0].title()

        for a in soup.find_all("a", href=True)[:max_results * 3]:
            text = self._clean_text(a.get_text())
            if len(text) < 5 or len(text) > 120:
                continue
            href = urljoin(base_url, a["href"])
            if not role_filter or self._matches_role(text, role_filter):
                jobs.append(RawJob(
                    title=text,
                    company=company,
                    location="See posting",
                    url=href,
                    source="careers_page",
                ))
            if len(jobs) >= max_results:
                break
        return jobs

    @staticmethod
    def _matches_role(title: str, role_filter: str) -> bool:
        return any(word.lower() in title.lower() for word in role_filter.split())
