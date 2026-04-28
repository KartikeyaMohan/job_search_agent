"""AI Job Search Agent — CLI entry point.

Usage:
    python main.py init --resume resume.pdf --name "Jane Doe" --email "jane@example.com"
    python main.py search --role "ML Engineer" --location "Remote"
    python main.py match
    python main.py status
    python main.py tailor <job_id>
    python main.py outreach <job_id>
    python main.py run        # full pipeline
    python main.py feedback <job_id> --response yes
    python main.py add-job    # manually add a job from URL or paste
"""
import json
import sys
import logging
from pathlib import Path
from typing import Optional, Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text
from rich.prompt import Prompt, Confirm

app = typer.Typer(
    name="job-search-agent",
    help="AI-powered autonomous job search agent",
    add_completion=False,
)
console = Console()

# Lazy imports to avoid slow startup
def _get_settings():
    from config import get_settings
    return get_settings()

def _load_profile() -> "CandidateProfile | None":
    from config import get_settings
    from models import CandidateProfile
    settings = get_settings()
    if not settings.profile_path.exists():
        return None
    return CandidateProfile.load(settings.profile_path)

def _get_orchestrator(profile=None):
    if profile is None:
        profile = _load_profile()
        if profile is None:
            console.print("[red]No candidate profile found. Run 'python main.py init' first.[/red]")
            raise typer.Exit(1)
    from orchestrator import JobSearchOrchestrator
    return JobSearchOrchestrator(profile)

# ── init ─────────────────────────────────────────────────────────────────────

@app.command()
def init(
    resume: Annotated[Optional[Path], typer.Option("--resume", "-r", help="Path to PDF/DOCX resume")] = None,
    name: Annotated[Optional[str], typer.Option(help="Full name")] = None,
    email: Annotated[Optional[str], typer.Option(help="Email address")] = None,
    location: Annotated[str, typer.Option(help="Current location")] = "Remote",
    profile_json: Annotated[Optional[Path], typer.Option("--profile", help="Load from JSON profile file")] = None,
):
    """Initialize the agent with your resume and profile."""
    from models import CandidateProfile, Experience, Education

    console.print(Panel.fit("[bold cyan]AI Job Search Agent — Setup[/bold cyan]"))

    if profile_json and profile_json.exists():
        profile = CandidateProfile.load(profile_json)
        console.print(f"[green]Loaded profile from {profile_json}[/green]")
    else:
        # Interactive setup
        if not name:
            name = Prompt.ask("Full name")
        if not email:
            email = Prompt.ask("Email")
        location = Prompt.ask("Location", default=location)
        phone = Prompt.ask("Phone (optional)", default="")
        linkedin = Prompt.ask("LinkedIn URL (optional)", default="")
        github = Prompt.ask("GitHub URL (optional)", default="")
        years_exp = Prompt.ask("Years of experience", default="0")
        summary = Prompt.ask("Brief professional summary (optional)", default="")

        # Skills
        console.print("\n[yellow]Enter your skills (comma-separated):[/yellow]")
        console.print("  e.g. Python, Machine Learning, SQL, Docker")
        skills_input = Prompt.ask("Skills")
        skills = [s.strip() for s in skills_input.split(",") if s.strip()]

        console.print("\n[yellow]Tech stack (comma-separated):[/yellow]")
        console.print("  e.g. PyTorch, FastAPI, PostgreSQL, Kubernetes")
        tech_input = Prompt.ask("Tech stack")
        tech_stack = [t.strip() for t in tech_input.split(",") if t.strip()]

        console.print("\n[yellow]Target job titles (comma-separated):[/yellow]")
        console.print("  e.g. ML Engineer, Senior Python Developer, Data Scientist")
        roles_input = Prompt.ask("Target roles")
        target_roles = [r.strip() for r in roles_input.split(",") if r.strip()]

        console.print("\n[yellow]Target locations (comma-separated):[/yellow]")
        console.print("  e.g. Remote, New York, London")
        locs_input = Prompt.ask("Target locations", default="Remote")
        target_locations = [l.strip() for l in locs_input.split(",") if l.strip()]

        profile = CandidateProfile(
            name=name,
            email=email,
            phone=phone or None,
            location=location,
            linkedin_url=linkedin or None,
            github_url=github or None,
            summary=summary or None,
            skills=skills,
            tech_stack=tech_stack,
            years_of_experience=int(years_exp) if years_exp.isdigit() else None,
            target_roles=target_roles,
            target_locations=target_locations,
        )

    # Process resume
    if resume and resume.exists():
        console.print(f"\n[cyan]Processing resume: {resume}[/cyan]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Parsing and indexing resume...", total=None)
            orch = _get_orchestrator(profile)
            orch.initialize_profile(str(resume))
            progress.update(task, completed=True)
        console.print(f"[green]✓ Resume indexed successfully[/green]")
    else:
        # Save profile without resume
        settings = _get_settings()
        profile.save(settings.profile_path)
        if resume:
            console.print(f"[yellow]Warning: Resume file not found at {resume}[/yellow]")

    console.print(f"\n[bold green]Profile initialized for {profile.name}![/bold green]")
    console.print(f"  Skills: {len(profile.skills + profile.tech_stack)} technologies")
    console.print(f"  Target roles: {', '.join(profile.target_roles)}")
    console.print(f"  Target locations: {', '.join(profile.target_locations)}")
    console.print("\n[dim]Next: run 'python main.py search' to find jobs[/dim]")


# ── search ────────────────────────────────────────────────────────────────────

@app.command()
def search(
    role: Annotated[Optional[str], typer.Option("--role", "-r", help="Job title to search")] = None,
    location: Annotated[Optional[str], typer.Option("--location", "-l", help="Target location")] = None,
    sources: Annotated[str, typer.Option(help="Comma-separated sources: jsearch,linkedin,wellfound")] = "jsearch,linkedin,wellfound",
    match: Annotated[bool, typer.Option("--match/--no-match", help="Also run matching after search")] = True,
):
    """Search for jobs across LinkedIn, Wellfound, and JSearch."""
    profile = _load_profile()
    if not profile:
        console.print("[red]No profile found. Run 'python main.py init' first.[/red]")
        raise typer.Exit(1)

    # Override profile with CLI args
    if role:
        profile.target_roles = [r.strip() for r in role.split(",")]
    if location:
        profile.target_locations = [l.strip() for l in location.split(",")]

    source_list = [s.strip() for s in sources.split(",")]

    console.print(Panel.fit(
        f"[bold]Searching for:[/bold] {', '.join(profile.target_roles)}\n"
        f"[bold]Locations:[/bold] {', '.join(profile.target_locations)}\n"
        f"[bold]Sources:[/bold] {', '.join(source_list)}",
        title="[cyan]Job Search[/cyan]"
    ))

    orch = _get_orchestrator(profile)

    if match:
        with console.status("[cyan]Running research + matching pipeline...[/cyan]"):
            result = orch.run_full_research_and_matching(source_list)
    else:
        with console.status("[cyan]Running job search...[/cyan]"):
            result = orch.run_research(source_list)

    console.print("\n[bold green]Search Complete[/bold green]")
    console.print(result)
    console.print("\n[dim]Run 'python main.py status' to see results[/dim]")


# ── match ─────────────────────────────────────────────────────────────────────

@app.command()
def match():
    """Score all discovered jobs for relevance to your profile."""
    orch = _get_orchestrator()
    db = orch.db
    job_count = db.job_count()

    if job_count == 0:
        console.print("[yellow]No jobs found. Run 'python main.py search' first.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[cyan]Scoring {job_count} jobs...[/cyan]")
    with console.status("[cyan]Running matching agent...[/cyan]"):
        result = orch.run_matching()
    console.print(result)


# ── status / dashboard ────────────────────────────────────────────────────────

@app.command()
def status(
    min_score: Annotated[float, typer.Option(help="Minimum match score to show")] = 0.0,
    limit: Annotated[int, typer.Option(help="Max rows to show")] = 25,
):
    """Show the job search dashboard with match scores."""
    settings = _get_settings()
    from database import get_db
    db = get_db()

    # Summary panel
    job_count = db.job_count()
    app_count = db.application_count()
    feedback_stats = db.get_feedback_stats()

    summary = (
        f"[bold]Jobs found:[/bold] {job_count}  |  "
        f"[bold]Applications:[/bold] {app_count}  |  "
        f"[bold]Response rate:[/bold] {feedback_stats['response_rate']:.0%}"
    )
    console.print(Panel(summary, title="[bold cyan]Job Search Dashboard[/bold cyan]"))

    # Jobs table
    jobs = db.list_jobs(min_score=min_score, limit=limit)
    if not jobs:
        console.print("[yellow]No jobs found yet. Run 'python main.py search' to start.[/yellow]")
        return

    table = Table(
        title=f"Top Jobs (showing {len(jobs)})",
        box=box.ROUNDED,
        show_lines=False,
        highlight=True,
    )
    table.add_column("ID", style="dim", width=8)
    table.add_column("Title", style="bold white", max_width=30)
    table.add_column("Company", style="cyan", max_width=20)
    table.add_column("Location", max_width=15)
    table.add_column("Match", justify="right", width=8)
    table.add_column("ATS", justify="right", width=8)
    table.add_column("Source", style="dim", width=10)
    table.add_column("Status", width=12)

    for job in jobs:
        app = db.get_application_by_job(job["id"])
        app_status = app["status"] if app else "-"
        match_score = job.get("match_score") or 0.0
        ats_score = job.get("ats_score") or 0.0

        # Color code match scores
        if match_score >= 0.8:
            score_str = f"[green]{match_score:.0%}[/green]"
        elif match_score >= 0.6:
            score_str = f"[yellow]{match_score:.0%}[/yellow]"
        elif match_score > 0:
            score_str = f"[red]{match_score:.0%}[/red]"
        else:
            score_str = "[dim]—[/dim]"

        ats_str = f"{ats_score:.0%}" if ats_score > 0 else "[dim]—[/dim]"

        # Status badge
        status_colors = {
            "applied": "green",
            "interviewing": "bright_green",
            "offered": "bold green",
            "rejected": "red",
            "ghosted": "dim red",
            "ready": "cyan",
        }
        color = status_colors.get(app_status, "white")
        status_text = f"[{color}]{app_status}[/{color}]"

        table.add_row(
            job["id"][:8],
            job["title"],
            job["company"],
            job["location"][:15],
            score_str,
            ats_str,
            job.get("source", "—"),
            status_text,
        )

    console.print(table)
    console.print(f"\n[dim]Use the 8-char ID prefix with tailor/outreach commands[/dim]")
    console.print("[dim]Commands: tailor <id> | outreach <id> | feedback <id>[/dim]")


# ── tailor ────────────────────────────────────────────────────────────────────

@app.command()
def tailor(
    job_id: Annotated[str, typer.Argument(help="Job ID (from 'status' command)")],
):
    """Tailor your resume for a specific job to maximize ATS score."""
    from database import get_db
    db = get_db()

    # Allow partial ID match
    if len(job_id) < 36:
        jobs = db.list_jobs(limit=500)
        matches = [j for j in jobs if j["id"].startswith(job_id)]
        if not matches:
            console.print(f"[red]No job found with ID prefix: {job_id}[/red]")
            raise typer.Exit(1)
        if len(matches) > 1:
            console.print(f"[yellow]Multiple matches for '{job_id}'. Be more specific.[/yellow]")
            for m in matches:
                console.print(f"  {m['id']} — {m['title']} @ {m['company']}")
            raise typer.Exit(1)
        job_id = matches[0]["id"]

    job = db.get_job(job_id)
    if not job:
        console.print(f"[red]Job {job_id} not found.[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold]Tailoring resume for:[/bold]\n{job['title']} at {job['company']}\n"
        f"Match score: {job.get('match_score', 0):.0%}",
        title="[cyan]Resume Tailor[/cyan]"
    ))

    orch = _get_orchestrator()
    with console.status("[cyan]AI is tailoring your resume...[/cyan]"):
        result = orch.tailor_resume_for_job(job_id)

    console.print("\n[bold green]Done![/bold green]")
    console.print(result)


# ── outreach ──────────────────────────────────────────────────────────────────

@app.command()
def outreach(
    job_id: Annotated[str, typer.Argument(help="Job ID (from 'status' command)")],
    types: Annotated[str, typer.Option(help="Message types: linkedin_connection,cover_letter,email")] = "linkedin_connection,cover_letter",
    recruiter: Annotated[str, typer.Option(help="Recruiter's first name")] = "",
):
    """Generate personalized recruiter messages and cover letters."""
    from database import get_db
    db = get_db()

    if len(job_id) < 36:
        jobs = db.list_jobs(limit=500)
        matches = [j for j in jobs if j["id"].startswith(job_id)]
        if not matches:
            console.print(f"[red]No job found with ID prefix: {job_id}[/red]")
            raise typer.Exit(1)
        job_id = matches[0]["id"]

    job = db.get_job(job_id)
    if not job:
        console.print(f"[red]Job {job_id} not found.[/red]")
        raise typer.Exit(1)

    message_types = [t.strip() for t in types.split(",")]

    console.print(Panel.fit(
        f"[bold]Generating outreach for:[/bold]\n{job['title']} at {job['company']}\n"
        f"Message types: {', '.join(message_types)}",
        title="[cyan]Outreach Generator[/cyan]"
    ))

    orch = _get_orchestrator()
    with console.status("[cyan]AI is crafting your messages...[/cyan]"):
        result = orch.generate_outreach(job_id, message_types, recruiter)

    console.print("\n[bold green]Done![/bold green]")
    console.print(result)


# ── run (full pipeline) ───────────────────────────────────────────────────────

@app.command()
def run(
    sources: Annotated[str, typer.Option(help="Comma-separated sources")] = "jsearch,linkedin,wellfound",
    top_n: Annotated[int, typer.Option(help="Process top N matching jobs")] = 5,
    message_types: Annotated[str, typer.Option(help="Outreach message types")] = "linkedin_connection,cover_letter",
):
    """Run the full autonomous pipeline: search → match → tailor → outreach."""
    console.print(Panel(
        "[bold cyan]Running Full Job Search Pipeline[/bold cyan]\n\n"
        "Stage 1: Research — discover jobs from all sources\n"
        "Stage 2: Match   — score relevance for your profile\n"
        "Stage 3: Tailor  — customize resume for top jobs\n"
        "Stage 4: Outreach — draft recruiter messages\n",
        title="[bold]AI Job Search Agent[/bold]"
    ))

    if not Confirm.ask("Start the full pipeline?"):
        raise typer.Exit(0)

    source_list = [s.strip() for s in sources.split(",")]
    msg_types = [m.strip() for m in message_types.split(",")]

    orch = _get_orchestrator()

    with console.status("[cyan]Running full pipeline...[/cyan]", spinner="dots"):
        results = orch.run_full_pipeline(
            sources=source_list,
            top_n=top_n,
            message_types=msg_types,
        )

    console.print("\n[bold green]Pipeline Complete![/bold green]")
    console.print(f"\nProcessed {len(results.get('tailored', []))} jobs:")
    for item in results.get("tailored", []):
        console.print(f"  ✓ {item['title']} @ {item['company']}")

    console.print("\n[dim]Run 'python main.py status' to see the full dashboard[/dim]")


# ── feedback ──────────────────────────────────────────────────────────────────

@app.command()
def feedback(
    job_id: Annotated[str, typer.Argument(help="Job ID to record feedback for")],
    response: Annotated[str, typer.Option("--response", help="'yes' or 'no'")] = "",
    days: Annotated[Optional[int], typer.Option(help="Days until response received")] = None,
):
    """Record application outcome to improve future matching."""
    from database import get_db
    db = get_db()

    if len(job_id) < 36:
        jobs = db.list_jobs(limit=500)
        matches = [j for j in jobs if j["id"].startswith(job_id)]
        if matches:
            job_id = matches[0]["id"]

    if not response:
        response = Prompt.ask("Did you get a response?", choices=["yes", "no"])

    got_response = response.lower() in ("yes", "y", "true", "1")

    if got_response and days is None:
        days_str = Prompt.ask("How many days after applying? (optional)", default="")
        days = int(days_str) if days_str.isdigit() else None

    orch = _get_orchestrator()
    result = orch.record_feedback(job_id, got_response, days)
    console.print(f"\n[green]{result}[/green]")


# ── add-job ───────────────────────────────────────────────────────────────────

@app.command(name="add-job")
def add_job(
    url: Annotated[Optional[str], typer.Option(help="Job posting URL")] = None,
    file: Annotated[Optional[Path], typer.Option(help="JSON file with job data")] = None,
):
    """Manually add a job posting (paste URL or provide JSON)."""
    from database import get_db
    from tools.scrapers.generic_scraper import GenericCareersScraper

    db = get_db()

    if file and file.exists():
        job_data = json.loads(file.read_text())
        job_id = db.upsert_job(job_data)
        console.print(f"[green]Job added: {job_id}[/green]")
        return

    if not url:
        url = Prompt.ask("Job posting URL")

    console.print(f"[cyan]Scraping: {url}[/cyan]")
    scraper = GenericCareersScraper()
    job = scraper.scrape_url(url)

    if not job:
        # Manual entry
        console.print("[yellow]Could not auto-scrape. Enter details manually:[/yellow]")
        title = Prompt.ask("Job title")
        company = Prompt.ask("Company name")
        location = Prompt.ask("Location", default="Remote")
        desc = Prompt.ask("Paste job description (or press Enter to skip)", default="")

        from tools.scrapers.base_scraper import RawJob
        job_raw = RawJob(title=title, company=company, location=location, url=url,
                         description=desc, source="manual")
        job_id = db.upsert_job({
            "title": job_raw.title, "company": job_raw.company,
            "location": job_raw.location, "url": job_raw.url,
            "description": job_raw.description, "source": job_raw.source,
        })
    else:
        job_id = db.upsert_job({
            "title": job[0].title, "company": job[0].company,
            "location": job[0].location, "url": job[0].url,
            "description": job[0].description, "source": job[0].source,
        })

    console.print(f"[green]✓ Job saved with ID: {job_id[:8]}[/green]")
    console.print(f"[dim]Run 'python main.py tailor {job_id[:8]}' to tailor your resume[/dim]")


# ── view ──────────────────────────────────────────────────────────────────────

@app.command()
def view(
    job_id: Annotated[str, typer.Argument(help="Job ID to view details for")],
):
    """View full details for a specific job."""
    from database import get_db
    db = get_db()

    jobs = db.list_jobs(limit=500)
    matches = [j for j in jobs if j["id"].startswith(job_id)]
    if not matches:
        console.print(f"[red]Job not found: {job_id}[/red]")
        raise typer.Exit(1)
    job = matches[0]

    console.print(Panel(
        f"[bold]{job['title']}[/bold] at [cyan]{job['company']}[/cyan]\n\n"
        f"[dim]ID:[/dim] {job['id']}\n"
        f"[dim]Location:[/dim] {job['location']}\n"
        f"[dim]Type:[/dim] {job.get('job_type') or '—'}\n"
        f"[dim]Salary:[/dim] {job.get('salary_range') or '—'}\n"
        f"[dim]Source:[/dim] {job.get('source', '—')}\n"
        f"[dim]URL:[/dim] {job['url']}\n\n"
        f"[bold]Match Score:[/bold] {job.get('match_score', 0):.0%}  "
        f"[bold]ATS Score:[/bold] {job.get('ats_score', 0):.0%}\n\n"
        f"[bold]Matched Keywords:[/bold] {', '.join(job.get('keywords', [])[:15]) or '—'}\n"
        f"[bold]Missing Keywords:[/bold] {', '.join(job.get('missing_keywords', [])[:15]) or '—'}",
        title=f"[bold]Job Details[/bold]",
    ))

    if job.get("description"):
        console.print("\n[bold]Description:[/bold]")
        console.print(job["description"][:1500] + ("..." if len(job["description"]) > 1500 else ""))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Silence noisy third-party loggers
    for name in ["httpx", "httpcore", "chromadb", "urllib3", "playwright"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    app()
