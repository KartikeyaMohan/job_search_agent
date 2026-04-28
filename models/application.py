from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class ApplicationStatus(str, Enum):
    FOUND = "found"
    ANALYZING = "analyzing"
    READY = "ready"
    APPLIED = "applied"
    INTERVIEWING = "interviewing"
    OFFERED = "offered"
    REJECTED = "rejected"
    GHOSTED = "ghosted"
    WITHDRAWN = "withdrawn"


class Application(BaseModel):
    id: Optional[str] = None
    job_id: str
    status: ApplicationStatus = ApplicationStatus.FOUND
    match_score: float = 0.0
    ats_score: float = 0.0
    tailored_resume_path: Optional[str] = None
    tailored_resume_text: Optional[str] = None
    outreach_message: Optional[str] = None
    cover_letter: Optional[str] = None
    applied_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None
    got_response: Optional[bool] = None  # feedback loop signal
    response_days: Optional[int] = None  # days until response (feedback)
