"""Resume Agent — tailors the candidate's resume for a specific job posting."""
import json
import logging
from pathlib import Path
from datetime import datetime

import anthropic
from tools.retry import with_retry
from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from config import get_settings
from models import CandidateProfile
from tools.rag.vector_store import get_vector_store
from tools.ats_scorer import score_resume_against_jd, top_missing_keywords
from database import get_db

logger = logging.getLogger(__name__)


class TailorResumeInput(BaseModel):
    job_id: str = Field(description="Database ID of the job to tailor the resume for")


class TailorResumeTool(BaseTool):
    name: str = "tailor_resume"
    description: str = (
        "Tailor the candidate's resume for a specific job to maximize ATS score. "
        "Retrieves the job description, identifies keyword gaps, and uses Claude to "
        "rewrite the resume with improved keyword density and relevance. "
        "Saves the tailored resume to disk and returns it."
    )
    args_schema: type[BaseModel] = TailorResumeInput

    def __init__(self, profile: CandidateProfile, **kwargs):
        super().__init__(**kwargs)
        self._profile = profile

    def _run(self, job_id: str) -> str:
        settings = get_settings()
        db = get_db()
        vs = get_vector_store()

        job = db.get_job(job_id)
        if not job:
            return f"Job {job_id} not found in database."

        jd_text = f"{job['title']} at {job['company']}\n\n{job['description']}\n{job.get('requirements', '')}"

        # Get resume text
        resume_text = self._profile.resume_text or (
            f"Name: {self._profile.name}\n"
            f"Email: {self._profile.email}\n"
            f"Location: {self._profile.location}\n\n"
            f"Summary: {self._profile.summary or ''}\n\n"
            f"Skills: {self._profile.skills_text}\n\n"
            f"Experience:\n{self._profile.experience_summary}\n\n"
            f"Education:\n" + "\n".join(
                f"  {e.degree} in {e.field} from {e.institution} ({e.year or ''})"
                for e in self._profile.education
            )
        )

        # Pre-tailor ATS score
        pre_result = score_resume_against_jd(resume_text, jd_text)
        missing = top_missing_keywords(pre_result.missing_keywords, jd_text)

        # Retrieve relevant resume chunks via RAG
        relevant_chunks = vs.query_resume(jd_text[:600], n_results=5)
        relevant_context = "\n---\n".join(relevant_chunks) if relevant_chunks else ""

        # Retrieve similar past successful applications for style reference
        past_apps = vs.query_applications(jd_text[:400], n_results=2)
        past_context = ""
        if past_apps:
            past_context = "\n\nSuccessful past application excerpts for reference:\n"
            past_context += "\n---\n".join(a["text"] for a in past_apps)

        # Use Claude with prompt caching for the long resume + JD context
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        system_prompt = (
            "You are an expert resume writer with deep knowledge of ATS systems and tech industry hiring. "
            "Your task is to tailor a resume for a specific job posting to maximize ATS pass rate "
            "while keeping the content honest and authentic. "
            "Rules:\n"
            "1. Do NOT fabricate experience or skills the candidate doesn't have.\n"
            "2. Reframe existing experience using the job's exact terminology where accurate.\n"
            "3. Incorporate missing keywords naturally into existing bullet points.\n"
            "4. Prioritize and reorder bullet points to emphasize the most relevant experience first.\n"
            "5. Keep the resume to 1-2 pages in text form.\n"
            "6. Use strong action verbs and quantify achievements where possible.\n"
            "7. Output the tailored resume in clean Markdown format."
        )

        user_prompt = (
            f"## Job Posting\n{jd_text[:3000]}\n\n"
            f"## Original Resume\n{resume_text[:3000]}\n\n"
            f"## Most Relevant Experience Sections\n{relevant_context[:1000]}\n"
            f"{past_context[:500]}\n\n"
            f"## Missing Keywords to Incorporate (if genuinely applicable)\n"
            f"{', '.join(missing[:15])}\n\n"
            f"## Current ATS Score: {pre_result.score:.0%} | Target: >85%\n\n"
            f"Produce a tailored version of the resume optimized for this specific role. "
            f"Return ONLY the resume in Markdown format, no explanations."
        )

        response = with_retry(lambda: client.messages.create(
            model=settings.claude_model,
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        ))

        tailored_resume = response.content[0].text

        # Score the tailored resume
        post_result = score_resume_against_jd(tailored_resume, jd_text)

        # Save to disk
        safe_company = "".join(c if c.isalnum() else "_" for c in job["company"])
        safe_title = "".join(c if c.isalnum() else "_" for c in job["title"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"resume_{safe_company}_{safe_title}_{ts}.md"
        output_path = settings.applications_dir / filename
        output_path.write_text(tailored_resume, encoding="utf-8")

        # Update application record
        db.upsert_application({
            "job_id": job_id,
            "status": "ready",
            "match_score": job.get("match_score", 0.0),
            "ats_score": post_result.score,
            "tailored_resume_path": str(output_path),
            "tailored_resume_text": tailored_resume,
        })

        improvement = post_result.score - pre_result.score
        return (
            f"Resume tailored for: {job['title']} at {job['company']}\n"
            f"ATS Score: {pre_result.score:.0%} → {post_result.score:.0%} "
            f"({'↑' if improvement > 0 else '↓'}{abs(improvement):.0%} improvement)\n"
            f"Saved to: {output_path}\n\n"
            f"Remaining gaps: {', '.join(post_result.missing_keywords[:10])}"
        )


def build_resume_agent(llm, profile: CandidateProfile) -> Agent:
    return Agent(
        role="Expert Technical Resume Writer",
        goal=(
            "Produce highly optimized, ATS-friendly resume variants that authentically represent "
            "the candidate while maximizing relevance score for each specific job. "
            "Aim for >85% ATS match score on every tailored resume."
        ),
        backstory=(
            "You are a certified professional resume writer specializing in the tech industry, "
            "with expertise in ATS optimization for companies like Google, Meta, Amazon, and startups. "
            "You understand that the best resumes tell a coherent career story while strategically "
            "incorporating the exact language recruiters and ATS systems look for. "
            "You never fabricate experience — you reframe real achievements powerfully."
        ),
        tools=[TailorResumeTool(profile=profile)],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_resume_task(agent: Agent, job_id: str, job_title: str, company: str) -> Task:
    return Task(
        description=(
            f"Tailor the candidate's resume for this specific position:\n\n"
            f"**Job ID:** {job_id}\n"
            f"**Role:** {job_title} at {company}\n\n"
            f"Use the tailor_resume tool with job_id='{job_id}'. "
            f"After tailoring, provide:\n"
            f"1. The ATS score improvement achieved\n"
            f"2. Top 5 strategic changes made to the resume\n"
            f"3. Any remaining keyword gaps and suggestions to address them"
        ),
        expected_output=(
            "A confirmation that the tailored resume was saved, "
            "the before/after ATS score, key changes made, and the file path."
        ),
        agent=agent,
    )
