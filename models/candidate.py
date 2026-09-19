from pydantic import BaseModel, Field
from typing import Optional
import json
from pathlib import Path


class Experience(BaseModel):
    company: str
    title: str
    duration: str  # e.g. "Jan 2022 - Present"
    description: str
    technologies: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str
    field: str
    year: Optional[str] = None
    gpa: Optional[str] = None


class CandidateProfile(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    location: str
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    summary: Optional[str] = None
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    years_of_experience: Optional[int] = None
    target_roles: list[str] = Field(default_factory=list)
    target_locations: list[str] = Field(default_factory=list)
    target_salary_min: Optional[int] = None
    max_job_age_days: int = Field(default=7, description="Only fetch jobs posted within this many days. Edit in candidate_profile.json to change.")
    resume_path: Optional[str] = None
    resume_text: Optional[str] = None  # Parsed text from resume

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CandidateProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    @property
    def skills_text(self) -> str:
        return ", ".join(self.skills + self.tech_stack)

    @property
    def experience_summary(self) -> str:
        lines = []
        for exp in self.experience:
            lines.append(f"{exp.title} at {exp.company} ({exp.duration})")
            lines.append(f"  {exp.description}")
            if exp.technologies:
                lines.append(f"  Tech: {', '.join(exp.technologies)}")
        return "\n".join(lines)
