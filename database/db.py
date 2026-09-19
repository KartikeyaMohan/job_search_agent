import uuid
import json
from datetime import datetime
from functools import lru_cache
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, String, Float, DateTime,
    Boolean, Text, Integer, event
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from config import get_settings


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    company = Column(String, nullable=False)
    location = Column(String)
    job_type = Column(String)
    description = Column(Text)
    requirements = Column(Text)
    salary_range = Column(String)
    url = Column(String, unique=True, nullable=False)
    source = Column(String)
    posted_date = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    match_score = Column(Float)
    ats_score = Column(Float)
    keywords = Column(Text)         # JSON array
    missing_keywords = Column(Text)  # JSON array


class ApplicationRecord(Base):
    __tablename__ = "applications"

    id = Column(String, primary_key=True)
    job_id = Column(String, nullable=False)
    status = Column(String, default="found")
    match_score = Column(Float, default=0.0)
    ats_score = Column(Float, default=0.0)
    tailored_resume_path = Column(String)
    tailored_resume_text = Column(Text)
    outreach_message = Column(Text)
    cover_letter = Column(Text)
    applied_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    got_response = Column(Boolean)
    response_days = Column(Integer)


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id = Column(String, primary_key=True)
    job_id = Column(String, nullable=False)
    application_id = Column(String)
    match_score = Column(Float)
    ats_score = Column(Float)
    got_response = Column(Boolean)
    response_days = Column(Integer)
    keywords_hit = Column(Text)  # JSON list of keywords that matched
    recorded_at = Column(DateTime, default=datetime.utcnow)


class Database:
    def __init__(self) -> None:
        settings = get_settings()
        self._engine = create_engine(
            f"sqlite:///{settings.db_path}",
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for better concurrent reads
        @event.listens_for(self._engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")

        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)

    @contextmanager
    def _session(self):
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Jobs ──────────────────────────────────────────────────────────────────

    def upsert_job(self, job: dict) -> str:
        with self._session() as s:
            existing = s.query(JobRecord).filter_by(url=job["url"]).first()
            if existing:
                return existing.id
            record = JobRecord(
                id=str(uuid.uuid4()),
                title=job["title"],
                company=job["company"],
                location=job.get("location", ""),
                job_type=job.get("job_type"),
                description=job.get("description", ""),
                requirements=job.get("requirements"),
                salary_range=job.get("salary_range"),
                url=job["url"],
                source=job.get("source", "manual"),
                posted_date=job.get("posted_date"),
                scraped_at=job.get("scraped_at", datetime.utcnow()),
                keywords=json.dumps(job.get("keywords", [])),
                missing_keywords=json.dumps(job.get("missing_keywords", [])),
            )
            s.add(record)
            return record.id

    def update_job_scores(
        self,
        job_id: str,
        match_score: float,
        ats_score: float,
        keywords: list[str],
        missing_keywords: list[str],
    ) -> None:
        with self._session() as s:
            r = s.query(JobRecord).filter_by(id=job_id).first()
            if r:
                r.match_score = match_score
                r.ats_score = ats_score
                r.keywords = json.dumps(keywords)
                r.missing_keywords = json.dumps(missing_keywords)

    def get_job(self, job_id: str) -> dict | None:
        with self._session() as s:
            r = s.query(JobRecord).filter_by(id=job_id).first()
            return self._job_dict(r) if r else None

    def list_jobs(self, min_score: float = 0.0, limit: int = 100) -> list[dict]:
        with self._session() as s:
            q = s.query(JobRecord)
            if min_score > 0:
                q = q.filter(JobRecord.match_score >= min_score)
            records = (
                q.order_by(JobRecord.match_score.desc().nullslast())
                .limit(limit)
                .all()
            )
            return [self._job_dict(r) for r in records]

    def job_count(self) -> int:
        with self._session() as s:
            return s.query(JobRecord).count()

    def clear_jobs(self) -> int:
        """Delete all jobs and applications. Returns the number of jobs deleted."""
        with self._session() as s:
            count = s.query(JobRecord).count()
            s.query(FeedbackRecord).delete()
            s.query(ApplicationRecord).delete()
            s.query(JobRecord).delete()
            return count

    # ── Applications ─────────────────────────────────────────────────────────

    def upsert_application(self, app: dict) -> str:
        with self._session() as s:
            existing = s.query(ApplicationRecord).filter_by(
                job_id=app["job_id"]
            ).first()
            if existing:
                for k, v in app.items():
                    if hasattr(existing, k):
                        setattr(existing, k, v)
                existing.updated_at = datetime.utcnow()
                return existing.id
            record = ApplicationRecord(
                id=str(uuid.uuid4()),
                **{k: v for k, v in app.items() if hasattr(ApplicationRecord, k)},
            )
            s.add(record)
            return record.id

    def update_application(self, app_id: str, **kwargs) -> None:
        with self._session() as s:
            r = s.query(ApplicationRecord).filter_by(id=app_id).first()
            if r:
                for k, v in kwargs.items():
                    if hasattr(r, k):
                        setattr(r, k, v)
                r.updated_at = datetime.utcnow()

    def get_application(self, app_id: str) -> dict | None:
        with self._session() as s:
            r = s.query(ApplicationRecord).filter_by(id=app_id).first()
            return self._app_dict(r) if r else None

    def get_application_by_job(self, job_id: str) -> dict | None:
        with self._session() as s:
            r = s.query(ApplicationRecord).filter_by(job_id=job_id).first()
            return self._app_dict(r) if r else None

    def list_applications(self) -> list[dict]:
        with self._session() as s:
            records = (
                s.query(ApplicationRecord)
                .order_by(ApplicationRecord.created_at.desc())
                .all()
            )
            return [self._app_dict(r) for r in records]

    def application_count(self) -> int:
        with self._session() as s:
            return s.query(ApplicationRecord).count()

    # ── Feedback ─────────────────────────────────────────────────────────────

    def record_feedback(
        self,
        job_id: str,
        application_id: str,
        got_response: bool,
        response_days: int | None = None,
    ) -> None:
        with self._session() as s:
            app = s.query(ApplicationRecord).filter_by(id=application_id).first()
            if app:
                app.got_response = got_response
                app.response_days = response_days
                app.updated_at = datetime.utcnow()
            job = s.query(JobRecord).filter_by(id=job_id).first()
            record = FeedbackRecord(
                id=str(uuid.uuid4()),
                job_id=job_id,
                application_id=application_id,
                match_score=app.match_score if app else None,
                ats_score=app.ats_score if app else None,
                got_response=got_response,
                response_days=response_days,
                keywords_hit=job.keywords if job else "[]",
            )
            s.add(record)

    def get_feedback_stats(self) -> dict:
        with self._session() as s:
            total = s.query(FeedbackRecord).count()
            responded = (
                s.query(FeedbackRecord)
                .filter_by(got_response=True)
                .count()
            )
            return {
                "total_feedback": total,
                "got_response": responded,
                "response_rate": responded / total if total else 0.0,
            }

    # ── Serializers ───────────────────────────────────────────────────────────

    @staticmethod
    def _job_dict(r: JobRecord) -> dict:
        return {
            "id": r.id,
            "title": r.title,
            "company": r.company,
            "location": r.location,
            "job_type": r.job_type,
            "description": r.description,
            "requirements": r.requirements,
            "salary_range": r.salary_range,
            "url": r.url,
            "source": r.source,
            "posted_date": r.posted_date,
            "scraped_at": r.scraped_at,
            "match_score": r.match_score,
            "ats_score": r.ats_score,
            "keywords": json.loads(r.keywords or "[]"),
            "missing_keywords": json.loads(r.missing_keywords or "[]"),
        }

    @staticmethod
    def _app_dict(r: ApplicationRecord) -> dict:
        return {
            "id": r.id,
            "job_id": r.job_id,
            "status": r.status,
            "match_score": r.match_score,
            "ats_score": r.ats_score,
            "tailored_resume_path": r.tailored_resume_path,
            "tailored_resume_text": r.tailored_resume_text,
            "outreach_message": r.outreach_message,
            "cover_letter": r.cover_letter,
            "applied_at": r.applied_at,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "notes": r.notes,
            "got_response": r.got_response,
            "response_days": r.response_days,
        }


@lru_cache(maxsize=1)
def get_db() -> Database:
    return Database()
