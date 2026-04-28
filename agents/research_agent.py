"""Research Agent — discovers job postings from multiple sources."""
import logging
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
                jobs = scraper.search(role, location, skills, max_results=max_results)
                all_jobs.extend(jobs)
                logger.info(f"{source}: {len(jobs)} jobs found")
            except Exception as e:
                logger.warning(f"{source} scraper error: {e}")

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

def build_research_agent(llm) -> Agent:
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
        tools=[SearchJobsTool()],
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
            f"**Years of Experience:** {profile.years_of_experience or 'Not specified'}\n\n"
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
