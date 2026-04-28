from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class JobSource(str, Enum):
    LINKEDIN = "linkedin"
    WELLFOUND = "wellfound"
    CAREERS_PAGE = "careers_page"
    JSEARCH = "jsearch"
    MANUAL = "manual"


class Job(BaseModel):
    id: Optional[str] = None
    title: str
    company: str
    location: str
    job_type: Optional[str] = None  # full-time, part-time, contract, remote
    description: str
    requirements: Optional[str] = None
    salary_range: Optional[str] = None
    url: str
    source: JobSource = JobSource.MANUAL
    posted_date: Optional[datetime] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)

    # Filled after analysis
    match_score: Optional[float] = None
    ats_score: Optional[float] = None
    keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        parts = [self.title, self.company, self.description]
        if self.requirements:
            parts.append(self.requirements)
        return "\n\n".join(parts)
