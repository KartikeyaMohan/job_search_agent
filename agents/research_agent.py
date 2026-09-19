"""Research Agent — discovers job postings from multiple sources."""
import logging
import re
from typing import Any

from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from config import get_settings
from models import CandidateProfile
from tools.scrapers import LinkedInScraper, WellfoundScraper, JSearchScraper
from tools.scrapers.base_scraper import RawJob
from database import get_db

logger = logging.getLogger(__name__)

# Words that appear in virtually every engineering job title and carry no discriminating signal
_GENERIC_ROLE_WORDS = frozenset({
    "engineer", "developer", "senior", "junior", "lead", "staff", "principal",
    "associate", "software", "specialist", "manager", "head", "director",
    "architect", "consultant", "ii", "iii", "iv",
})


def _title_relevant_to_role(title: str, role: str) -> bool:
    """Return True if the job title shares a meaningful keyword with the searched role."""
    role_words = set(role.lower().split()) - _GENERIC_ROLE_WORDS
    if not role_words:
        return True  # Role is entirely generic; don't filter
    title_lower = title.lower()
    return any(
        re.search(r"\b" + re.escape(word) + r"\b", title_lower)
        for word in role_words
    )


def _location_relevant(job_location: str, target_location: str) -> bool:
    """Return True if the job location matches the target or is remote."""
    loc_lower = job_location.lower()
    if "remote" in loc_lower:
        return True
    target_lower = target_location.lower()
    if target_lower in ("remote", "anywhere", "worldwide", ""):
        return True
    # Match any significant word from the target location (len > 3 avoids noise like "san", "new")
    target_words = [w for w in target_lower.split() if len(w) > 3]
    return any(w in loc_lower for w in target_words)


# ── Tool Input Schemas ────────────────────────────────────────────────────────

class JobSearchInput(BaseModel):
    role: str = Field(description="Job role/title to search for")
    location: str = Field(description="Target location or 'Remote'")
    skills: list[str] = Field(default_factory=list, description="Key skills to filter by")
    max_results: int = Field(default=20, description="Max results per source")
    sources: list[str] = Field(
        default=["jsearch", "linkedin", "wellfound"],
        description="Sources to search: jsearch, linkedin, wellfound",
    )


# ── CrewAI Tools ─────────────────────────────────────────────────────────────

class SearchJobsTool(BaseTool):
    name: str = "search_jobs"
    description: str = (
        "Search for job postings across LinkedIn, Wellfound, and JSearch (Indeed/Glassdoor). "
        "Returns a list of raw job postings with title, company, location, and description."
    )
    args_schema: type[BaseModel] = JobSearchInput

    def __init__(self, max_job_age_days: int = 7, **kwargs):
        super().__init__(**kwargs)
        self._max_job_age_days = max_job_age_days

    def _run(self, role: str, location: str, skills: list[str],
             max_results: int = 20, sources: list[str] = None) -> str:
        if sources is None:
            sources = ["jsearch", "linkedin", "wellfound"]

        all_jobs: list[RawJob] = []
        scrapers = {
            "jsearch": JSearchScraper(),
            "linkedin": LinkedInScraper(),
            "wellfound": WellfoundScraper(),
        }

        for source in sources:
            scraper = scrapers.get(source)
            if not scraper:
                continue
            try:
                jobs = scraper.search(role, location, skills, max_results=max_results, max_days=self._max_job_age_days)
                all_jobs.extend(jobs)
                logger.info(f"{source}: {len(jobs)} jobs found")
            except Exception as e:
                logger.warning(f"{source} scraper error: {e}")

        # Post-scrape filtering: drop jobs whose title doesn't match the searched role
        # or whose location is unrelated to the target (LinkedIn's algorithm can return
        # off-target results, e.g. Android jobs when searching for ML roles).
        original_count = len(all_jobs)
        title_filtered = [j for j in all_jobs if _title_relevant_to_role(j.title, role)]
        if title_filtered:
            all_jobs = title_filtered
        else:
            logger.warning("Title filter removed all results for role='%s'; using unfiltered", role)

        if location.lower() not in ("remote", "anywhere", "worldwide", ""):
            loc_filtered = [j for j in all_jobs if _location_relevant(j.location, location)]
            if loc_filtered:
                all_jobs = loc_filtered
            else:
                logger.warning("Location filter removed all results for location='%s'; skipping", location)

        logger.info("Post-scrape filter: %d → %d jobs kept", original_count, len(all_jobs))

        # Save to DB and deduplicate by URL
        db = get_db()
        seen_urls: set[str] = set()
        saved_ids: list[str] = []

        for job in all_jobs:
            if job.url in seen_urls:
                continue
            seen_urls.add(job.url)
            job_dict = {
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "url": job.url,
                "description": job.description,
                "requirements": job.requirements,
                "job_type": job.job_type,
                "salary_range": job.salary_range,
                "posted_date": job.posted_date,
                "source": job.source,
            }
            job_id = db.upsert_job(job_dict)
            saved_ids.append(job_id)

        summary_lines = []
        for job in all_jobs[:30]:  # Cap output length
            summary_lines.append(f"- [{job.source}] {job.title} @ {job.company} | {job.location}")

        return (
            f"Found {len(all_jobs)} jobs ({len(saved_ids)} saved to database).\n\n"
            + "\n".join(summary_lines)
        )


# ── Agent Factory ─────────────────────────────────────────────────────────────

def build_research_agent(llm, profile: CandidateProfile) -> Agent:
    return Agent(
        role="Senior Job Research Specialist",
        goal=(
            "Discover the most relevant job openings across multiple platforms for the candidate. "
            "Focus on roles matching their technical skills, experience level, and location preferences. "
            "Cast a wide net initially, then refine based on seniority and tech stack alignment."
        ),
        backstory=(
            "You are a seasoned talent acquisition expert who knows how to efficiently search "
            "job boards and company career pages. You understand tech industry hiring patterns, "
            "know which platforms have the best listings for different roles, and can identify "
            "promising opportunities that others might miss. You always document your findings "
            "thoroughly for downstream analysis."
        ),
        tools=[SearchJobsTool(max_job_age_days=profile.max_job_age_days)],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_research_task(agent: Agent, profile: CandidateProfile, sources: list[str]) -> Task:
    roles_str = ", ".join(profile.target_roles) if profile.target_roles else "Software Engineer"
    locations_str = ", ".join(profile.target_locations) if profile.target_locations else "Remote"
    skills_str = ", ".join(profile.tech_stack[:10])

    return Task(
        description=(
            f"Search for job openings matching this candidate's profile:\n\n"
            f"**Target Roles:** {roles_str}\n"
            f"**Locations:** {locations_str}\n"
            f"**Key Skills:** {skills_str}\n"
            f"**Years of Experience:** {profile.years_of_experience or 'Not specified'}\n"
            f"**Max Job Age:** {profile.max_job_age_days} days\n\n"
            f"Search across these sources: {', '.join(sources)}.\n"
            f"For each target role and location combination, run a search. "
            f"Aim for at least {get_settings().max_jobs_per_source} results total.\n"
            f"Save all findings to the database for analysis."
        ),
        expected_output=(
            "A summary of all job postings found, organized by source, "
            "including job title, company, location, and URL. "
            "Include the total count found and saved."
        ),
        agent=agent,
    )
