"""Matching Agent — scores job relevance against candidate profile using RAG + ATS."""
import json
import logging
from typing import Any

from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from config import get_settings
from models import CandidateProfile
from tools.ats_scorer import score_resume_against_jd, top_missing_keywords
from tools.rag.vector_store import get_vector_store
from database import get_db

logger = logging.getLogger(__name__)


class ScoreJobsInput(BaseModel):
    min_score_threshold: float = Field(
        default=0.5, description="Minimum match score to keep (0-1)"
    )
    max_jobs_to_score: int = Field(
        default=50, description="Maximum number of jobs to score"
    )


class ScoreAllJobsTool(BaseTool):
    name: str = "score_all_jobs"
    description: str = (
        "Score all unscored jobs in the database for relevance to the candidate's profile. "
        "Uses RAG retrieval over the resume combined with ATS keyword matching. "
        "Returns a ranked list of jobs with match scores."
    )
    args_schema: type[BaseModel] = ScoreJobsInput

    def __init__(self, profile: CandidateProfile, **kwargs):
        super().__init__(**kwargs)
        self._profile = profile

    def _run(self, min_score_threshold: float = 0.5, max_jobs_to_score: int = 50) -> str:
        db = get_db()
        vs = get_vector_store()
        settings = get_settings()

        jobs = db.list_jobs(limit=max_jobs_to_score)
        if not jobs:
            return "No jobs found in database. Run the research agent first."

        resume_text = self._profile.resume_text or ""
        if not resume_text:
            resume_text = (
                f"Skills: {self._profile.skills_text}\n\n"
                f"Experience:\n{self._profile.experience_summary}"
            )

        results = []
        for job in jobs:
            jd_text = f"{job['title']}\n{job['description']}\n{job.get('requirements', '')}"

            # ATS scoring
            ats_result = score_resume_against_jd(resume_text, jd_text)

            # RAG-based relevance: query resume for job-relevant experience
            relevant_chunks = vs.query_resume(jd_text[:500], n_results=3)
            rag_score = min(1.0, len(relevant_chunks) / 3.0) if relevant_chunks else 0.0

            # Combined match score: 70% ATS + 30% RAG
            match_score = round(0.7 * ats_result.score + 0.3 * rag_score, 4)

            missing = top_missing_keywords(ats_result.missing_keywords, jd_text)
            db.update_job_scores(
                job["id"],
                match_score=match_score,
                ats_score=ats_result.score,
                keywords=ats_result.matched_keywords,
                missing_keywords=missing,
            )

            results.append({
                "id": job["id"],
                "title": job["title"],
                "company": job["company"],
                "match_score": match_score,
                "ats_score": ats_result.score,
                "keyword_coverage": ats_result.keyword_coverage,
                "missing_keywords": missing[:5],
            })

        # Sort by match score
        results.sort(key=lambda x: x["match_score"], reverse=True)
        qualified = [r for r in results if r["match_score"] >= min_score_threshold]

        output_lines = [
            f"Scored {len(results)} jobs. {len(qualified)} qualify (score >= {min_score_threshold}).\n",
            "Top matches:",
        ]
        for r in results[:15]:
            flag = "✓" if r["match_score"] >= min_score_threshold else "✗"
            output_lines.append(
                f"  {flag} [{r['match_score']:.0%} match] {r['title']} @ {r['company']} "
                f"(ATS: {r['ats_score']:.0%})"
            )
            if r["missing_keywords"]:
                output_lines.append(f"     Missing: {', '.join(r['missing_keywords'])}")

        return "\n".join(output_lines)


def build_matching_agent(llm, profile: CandidateProfile) -> Agent:
    return Agent(
        role="AI Job Relevance Analyst",
        goal=(
            "Accurately score and rank all discovered jobs by how well they match the candidate's "
            "skills, experience, and career goals. Identify which requirements the candidate meets "
            "and flag critical skill gaps. Prioritise quality over quantity."
        ),
        backstory=(
            "You are a specialized AI analyst combining NLP, ATS expertise, and deep understanding "
            "of tech industry hiring. You use semantic search over the candidate's resume to find "
            "relevant experience and apply keyword analysis to predict ATS pass rates. "
            "Your scores help the candidate focus on applications with the highest success probability."
        ),
        tools=[ScoreAllJobsTool(profile=profile)],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_matching_task(agent: Agent, profile: CandidateProfile) -> Task:
    threshold = get_settings().relevance_threshold
    return Task(
        description=(
            f"Score all jobs in the database for relevance to this candidate:\n\n"
            f"**Name:** {profile.name}\n"
            f"**Skills:** {profile.skills_text}\n"
            f"**Experience:** {profile.years_of_experience or 'N/A'} years\n"
            f"**Target Roles:** {', '.join(profile.target_roles)}\n\n"
            f"Use the score_all_jobs tool with a threshold of {threshold}. "
            f"After scoring, provide an analysis of:\n"
            f"1. Top 5 best-matching jobs and why they're a good fit\n"
            f"2. Common skill gaps across rejected jobs\n"
            f"3. Which companies/industries are best aligned with the candidate's profile"
        ),
        expected_output=(
            "A ranked list of all scored jobs with match percentages, ATS scores, and "
            "a strategic analysis identifying the top opportunities and key skill gaps."
        ),
        agent=agent,
    )
