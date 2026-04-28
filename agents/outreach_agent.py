"""Outreach Agent — drafts personalized recruiter messages and cover letters."""
import logging
from datetime import datetime

import anthropic
from tools.retry import with_retry
from crewai import Agent, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from config import get_settings
from models import CandidateProfile
from tools.rag.vector_store import get_vector_store
from database import get_db

logger = logging.getLogger(__name__)


class OutreachInput(BaseModel):
    job_id: str = Field(description="Database ID of the job to write outreach for")
    message_type: str = Field(
        default="linkedin_connection",
        description="Type: 'linkedin_connection', 'linkedin_inmail', 'email', 'cover_letter'",
    )
    recruiter_name: str = Field(default="", description="Recruiter's first name if known")


class GenerateOutreachTool(BaseTool):
    name: str = "generate_outreach"
    description: str = (
        "Generate a personalized recruiter outreach message or cover letter for a specific job. "
        "Supports LinkedIn connection requests, InMails, cold emails, and cover letters. "
        "Uses RAG to pull the most relevant experience from the candidate's background."
    )
    args_schema: type[BaseModel] = OutreachInput

    def __init__(self, profile: CandidateProfile, **kwargs):
        super().__init__(**kwargs)
        self._profile = profile

    def _run(
        self,
        job_id: str,
        message_type: str = "linkedin_connection",
        recruiter_name: str = "",
    ) -> str:
        settings = get_settings()
        db = get_db()
        vs = get_vector_store()

        job = db.get_job(job_id)
        if not job:
            return f"Job {job_id} not found."

        jd_text = f"{job['title']} at {job['company']}\n\n{job['description']}"

        # Pull most relevant resume sections via RAG
        relevant_chunks = vs.query_resume(jd_text[:500], n_results=3)
        relevant_context = "\n".join(relevant_chunks) if relevant_chunks else ""

        # Message length limits by type
        limits = {
            "linkedin_connection": 300,
            "linkedin_inmail": 1500,
            "email": 2000,
            "cover_letter": 3000,
        }
        char_limit = limits.get(message_type, 1000)

        # Word count guidelines
        word_guides = {
            "linkedin_connection": "~50 words — very brief, one call to action",
            "linkedin_inmail": "150-300 words — professional, specific value prop",
            "email": "200-350 words — structured: hook, value, CTA",
            "cover_letter": "3 paragraphs, ~400 words — narrative, specific, compelling",
        }
        word_guide = word_guides.get(message_type, "~200 words")

        salutation = f"Hi {recruiter_name}," if recruiter_name else "Hi [Recruiter Name],"

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        system_prompt = (
            "You are an expert career coach and communications specialist who writes highly personalized, "
            "compelling job application messages. Your messages:\n"
            "- Feel authentic and human, never templated\n"
            "- Lead with specific value the candidate brings\n"
            "- Reference specific details from the job posting\n"
            "- Are concise and respect the reader's time\n"
            "- Include a clear, low-pressure call to action\n"
            "- Avoid buzzwords, clichés, and generic phrases\n"
            "Output ONLY the message text, no explanations or labels."
        )

        profile_summary = (
            f"Candidate: {self._profile.name}\n"
            f"Experience: {self._profile.years_of_experience or 'N/A'} years\n"
            f"Current/Recent Role: {self._profile.experience[0].title if self._profile.experience else 'N/A'}\n"
            f"Top Skills: {', '.join(self._profile.tech_stack[:8])}\n"
            f"Notable Experience:\n{relevant_context[:800]}\n"
            f"LinkedIn: {self._profile.linkedin_url or '[LinkedIn URL]'}\n"
            f"GitHub: {self._profile.github_url or ''}"
        )

        user_prompt = (
            f"Write a {message_type.replace('_', ' ')} for this job opportunity.\n\n"
            f"## Job Details\n"
            f"Role: {job['title']} at {job['company']}\n"
            f"Location: {job['location']}\n"
            f"Description snippet: {job['description'][:800]}\n\n"
            f"## Candidate Profile\n{profile_summary}\n\n"
            f"## Message Specs\n"
            f"- Salutation: {salutation}\n"
            f"- Length: {word_guide}\n"
            f"- Character limit: {char_limit}\n"
            f"- Type: {message_type.replace('_', ' ')}\n\n"
            f"Write the message now:"
        )

        response = with_retry(lambda: client.messages.create(
            model=settings.claude_model,
            max_tokens=1500,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        ))

        message = response.content[0].text.strip()

        # Save to database
        app = db.get_application_by_job(job_id)
        update_kwargs = {"outreach_message": message}
        if message_type == "cover_letter":
            update_kwargs = {"cover_letter": message}

        if app:
            db.update_application(app["id"], **update_kwargs)
        else:
            db.upsert_application({
                "job_id": job_id,
                "status": "ready",
                "match_score": job.get("match_score", 0.0),
                "ats_score": job.get("ats_score", 0.0),
                **update_kwargs,
            })

        # Save to file
        settings = get_settings()
        safe_company = "".join(c if c.isalnum() else "_" for c in job["company"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{message_type}_{safe_company}_{ts}.txt"
        out_path = settings.applications_dir / filename
        out_path.write_text(message, encoding="utf-8")

        return (
            f"Generated {message_type.replace('_', ' ')} for {job['title']} at {job['company']}\n"
            f"Character count: {len(message)}/{char_limit}\n"
            f"Saved to: {out_path}\n\n"
            f"--- MESSAGE PREVIEW ---\n{message[:500]}{'...' if len(message) > 500 else ''}"
        )


def build_outreach_agent(llm, profile: CandidateProfile) -> Agent:
    return Agent(
        role="Career Communications Strategist",
        goal=(
            "Craft highly personalized, compelling outreach messages that get responses from recruiters "
            "and hiring managers. Every message should reference specific job details and showcase "
            "the candidate's most relevant value proposition."
        ),
        backstory=(
            "You are a career coach who has helped hundreds of engineers land their dream jobs. "
            "You know that the average recruiter reads a message in under 10 seconds, so your "
            "messages hook immediately with specific value. You understand the psychology of "
            "hiring decisions and craft messages that make it easy for recruiters to say yes. "
            "Your LinkedIn connection request acceptance rate is over 60%."
        ),
        tools=[GenerateOutreachTool(profile=profile)],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_outreach_task(
    agent: Agent,
    job_id: str,
    job_title: str,
    company: str,
    message_types: list[str],
) -> Task:
    types_str = ", ".join(message_types)
    return Task(
        description=(
            f"Generate outreach materials for this job opportunity:\n\n"
            f"**Job ID:** {job_id}\n"
            f"**Role:** {job_title} at {company}\n\n"
            f"Generate the following message types: {types_str}\n"
            f"Call generate_outreach once for each message type.\n\n"
            f"After generating all messages, provide:\n"
            f"1. A brief critique of the strongest message\n"
            f"2. Tips for personalizing further if the recruiter's name is found\n"
            f"3. Best time to send and follow-up strategy"
        ),
        expected_output=(
            "All requested outreach messages saved to disk, with file paths, "
            "character counts, and strategic sending advice."
        ),
        agent=agent,
    )
