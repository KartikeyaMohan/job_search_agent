"""Main orchestration layer — wires together all agents into CrewAI workflows."""
import logging
from pathlib import Path
from typing import Optional

from crewai import Crew, Process, LLM

from config import get_settings
from models import CandidateProfile
from tools.rag.document_processor import DocumentProcessor
from tools.rag.vector_store import get_vector_store
from database import get_db

from agents.research_agent import build_research_agent, build_research_task
from agents.matching_agent import build_matching_agent, build_matching_task
from agents.resume_agent import build_resume_agent, build_resume_task
from agents.outreach_agent import build_outreach_agent, build_outreach_task

logger = logging.getLogger(__name__)


def _build_llm() -> LLM:
    settings = get_settings()
    return LLM(
        model=f"anthropic/{settings.claude_model}",
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
        max_retries=6,
    )


class JobSearchOrchestrator:
    """Coordinates the multi-agent job search pipeline."""

    def __init__(self, profile: CandidateProfile) -> None:
        self.profile = profile
        self.settings = get_settings()
        self.db = get_db()
        self.vs = get_vector_store()
        self.doc_processor = DocumentProcessor()
        self._llm = _build_llm()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def initialize_profile(self, resume_path: Optional[str] = None) -> None:
        """Parse resume and index it into the vector store."""
        if resume_path:
            logger.info(f"Parsing resume: {resume_path}")
            resume_text, chunks = self.doc_processor.process_resume(resume_path)
            self.profile.resume_text = resume_text
            self.profile.resume_path = resume_path
            self.vs.index_resume(chunks, resume_id="candidate")
            logger.info(f"Indexed {len(chunks)} resume chunks")
        elif self.profile.resume_text:
            chunks = self.doc_processor.process_job_description(self.profile.resume_text)
            self.vs.index_resume(chunks, resume_id="candidate")

        # Save updated profile
        self.profile.save(self.settings.profile_path)
        logger.info("Candidate profile saved")

    # ── Pipeline Stages ───────────────────────────────────────────────────────

    def run_research(
        self,
        sources: list[str] | None = None,
    ) -> str:
        """Stage 1: Discover jobs from all configured sources."""
        if sources is None:
            sources = ["jsearch", "linkedin", "wellfound"]
            if not self.settings.rapidapi_key:
                sources = ["linkedin", "wellfound"]

        logger.info(f"Starting research stage with sources: {sources}")

        agent = build_research_agent(self._llm, self.profile)
        task = build_research_task(agent, self.profile, sources)

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        logger.info("Research stage complete")
        return str(result)

    def run_matching(self) -> str:
        """Stage 2: Score all discovered jobs for relevance."""
        job_count = self.db.job_count()
        if job_count == 0:
            return "No jobs to score. Run research stage first."

        logger.info(f"Starting matching stage for {job_count} jobs")

        agent = build_matching_agent(self._llm, self.profile)
        task = build_matching_task(agent, self.profile)

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        logger.info("Matching stage complete")
        return str(result)

    def run_full_research_and_matching(self, sources: list[str] | None = None) -> str:
        """Run research and matching as a sequential 2-agent crew."""
        if sources is None:
            sources = ["jsearch", "linkedin", "wellfound"]
            if not self.settings.rapidapi_key:
                sources = ["linkedin", "wellfound"]

        research_agent = build_research_agent(self._llm, self.profile)
        matching_agent = build_matching_agent(self._llm, self.profile)

        research_task = build_research_task(research_agent, self.profile, sources)
        matching_task = build_matching_task(matching_agent, self.profile)

        # Context dependency: matching reads research output
        matching_task.context = [research_task]

        crew = Crew(
            agents=[research_agent, matching_agent],
            tasks=[research_task, matching_task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        return str(result)

    def tailor_resume_for_job(self, job_id: str) -> str:
        """Stage 3a: Tailor resume for a specific job."""
        job = self.db.get_job(job_id)
        if not job:
            return f"Job {job_id} not found."

        logger.info(f"Tailoring resume for: {job['title']} at {job['company']}")

        agent = build_resume_agent(self._llm, self.profile)
        task = build_resume_task(agent, job_id, job["title"], job["company"])

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()

        # Index tailored resume as a past application for future RAG
        app = self.db.get_application_by_job(job_id)
        if app and app.get("tailored_resume_text"):
            chunks = self.doc_processor.process_past_application(app["tailored_resume_text"])
            self.vs.index_application(
                chunks,
                job_id=job_id,
                job_title=job["title"],
                outcome="pending",
            )

        return str(result)

    def generate_outreach(
        self,
        job_id: str,
        message_types: list[str] | None = None,
        recruiter_name: str = "",
    ) -> str:
        """Stage 3b: Generate outreach messages for a specific job."""
        job = self.db.get_job(job_id)
        if not job:
            return f"Job {job_id} not found."

        if message_types is None:
            message_types = ["linkedin_connection", "cover_letter"]

        logger.info(f"Generating outreach for: {job['title']} at {job['company']}")

        agent = build_outreach_agent(self._llm, self.profile)
        task = build_outreach_task(
            agent, job_id, job["title"], job["company"], message_types
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        return str(result)

    def run_full_pipeline(
        self,
        sources: list[str] | None = None,
        top_n: int = 5,
        message_types: list[str] | None = None,
    ) -> dict:
        """Run the complete pipeline: research → match → tailor + outreach for top N jobs."""
        if message_types is None:
            message_types = ["linkedin_connection", "cover_letter"]

        results = {
            "research": "",
            "matching": "",
            "tailored": [],
            "outreach": [],
        }

        # Stage 1 + 2: Research and matching
        logger.info("=== STAGE 1+2: Research & Matching ===")
        results["research_matching"] = self.run_full_research_and_matching(sources)

        # Get top N qualified jobs
        threshold = self.settings.relevance_threshold
        top_jobs = self.db.list_jobs(min_score=threshold, limit=top_n)

        if not top_jobs:
            logger.warning(f"No jobs above threshold {threshold}. Lowering to 0.4.")
            top_jobs = self.db.list_jobs(min_score=0.4, limit=top_n)

        logger.info(f"Processing top {len(top_jobs)} jobs")

        # Stage 3: Tailor + outreach for each top job
        for job in top_jobs:
            job_id = job["id"]
            logger.info(f"=== Processing: {job['title']} @ {job['company']} ===")

            tailor_result = self.tailor_resume_for_job(job_id)
            results["tailored"].append({
                "job_id": job_id,
                "title": job["title"],
                "company": job["company"],
                "result": tailor_result,
            })

            outreach_result = self.generate_outreach(job_id, message_types)
            results["outreach"].append({
                "job_id": job_id,
                "title": job["title"],
                "company": job["company"],
                "result": outreach_result,
            })

        return results

    # ── Feedback Loop ─────────────────────────────────────────────────────────

    def record_feedback(
        self,
        job_id: str,
        got_response: bool,
        response_days: int | None = None,
    ) -> str:
        """Record application outcome to improve future matching."""
        app = self.db.get_application_by_job(job_id)
        if not app:
            return f"No application found for job {job_id}"

        self.db.record_feedback(
            job_id=job_id,
            application_id=app["id"],
            got_response=got_response,
            response_days=response_days,
        )

        # Update the indexed application's outcome for future RAG
        if app.get("tailored_resume_text"):
            chunks = self.doc_processor.process_past_application(app["tailored_resume_text"])
            outcome = "success" if got_response else "no_response"
            job = self.db.get_job(job_id)
            if job:
                self.vs.index_application(
                    chunks,
                    job_id=job_id,
                    job_title=job["title"],
                    outcome=outcome,
                )

        status = "got_response" if got_response else "ghosted"
        self.db.update_application(app["id"], status=status)

        stats = self.db.get_feedback_stats()
        return (
            f"Feedback recorded for job {job_id}.\n"
            f"Overall response rate: {stats['response_rate']:.0%} "
            f"({stats['got_response']}/{stats['total_feedback']} applications)"
        )
