"""JSearch scraper via RapidAPI — aggregates Indeed, Glassdoor, LinkedIn, ZipRecruiter.

Get a free API key at: https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
Free tier: 200 requests/month.
"""
import logging
from datetime import datetime
from typing import Optional

import requests

from .base_scraper import BaseScraper, RawJob
from config import get_settings

logger = logging.getLogger(__name__)

JSEARCH_HOST = "jsearch.p.rapidapi.com"
JSEARCH_BASE = "https://jsearch.p.rapidapi.com"


class JSearchScraper(BaseScraper):
    """Job search via JSearch RapidAPI (aggregates multiple job boards)."""

    def search(
        self,
        role: str,
        location: str,
        skills: list[str],
        max_results: int = 25,
    ) -> list[RawJob]:
        settings = get_settings()
        if not settings.rapidapi_key:
            logger.info("RAPIDAPI_KEY not set — skipping JSearch.")
            return []

        headers = {
            "X-RapidAPI-Key": settings.rapidapi_key,
            "X-RapidAPI-Host": JSEARCH_HOST,
        }

        # Build query: role + top skills
        skill_str = " ".join(skills[:3]) if skills else ""
        query = f"{role} {skill_str}".strip()

        jobs: list[RawJob] = []
        page = 1

        while len(jobs) < max_results:
            try:
                params = {
                    "query": f"{query} in {location}" if location else query,
                    "page": str(page),
                    "num_pages": "1",
                    "date_posted": "week",
                }
                resp = requests.get(
                    f"{JSEARCH_BASE}/search",
                    headers=headers,
                    params=params,
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                results = data.get("data", [])
                if not results:
                    break

                for item in results:
                    job = self._parse_result(item)
                    if job:
                        jobs.append(job)
                    if len(jobs) >= max_results:
                        break

                page += 1
                if page > 3:
                    break
                self._polite_delay(0.5, 1.0)

            except Exception as e:
                logger.error(f"JSearch API error: {e}")
                break

        logger.info(f"JSearch: found {len(jobs)} jobs for '{query}' in '{location}'")
        return jobs

    def _parse_result(self, item: dict) -> Optional[RawJob]:
        title = item.get("job_title", "")
        company = item.get("employer_name", "")
        location_parts = [
            item.get("job_city", ""),
            item.get("job_state", ""),
            item.get("job_country", ""),
        ]
        location = ", ".join(p for p in location_parts if p) or "Remote"
        url = item.get("job_apply_link") or item.get("job_google_link", "")
        description = item.get("job_description", "")
        job_type = item.get("job_employment_type", "")

        # Salary
        min_sal = item.get("job_min_salary")
        max_sal = item.get("job_max_salary")
        period = item.get("job_salary_period", "")
        salary = None
        if min_sal and max_sal:
            salary = f"${min_sal:,.0f} - ${max_sal:,.0f} {period}".strip()

        # Posted date
        posted_ts = item.get("job_posted_at_timestamp")
        posted_date = datetime.fromtimestamp(posted_ts) if posted_ts else None

        if not title or not url:
            return None

        return RawJob(
            title=title,
            company=company,
            location=location,
            url=url,
            description=description,
            job_type=job_type.lower().replace("_", "-") if job_type else None,
            salary_range=salary,
            posted_date=posted_date,
            source="jsearch",
        )
